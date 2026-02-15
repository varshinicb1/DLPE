import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import xgboost as xgb
import os

# --- Configuration ---
st.set_page_config(
    page_title="Distress Watch: DLPE Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data (Cached) ---
@st.cache_data
def load_data():
    # Load ranking
    ranking_df = pd.read_csv("district_vulnerability_ranking.csv")
    
    # Load temporal features (for historical charts)
    temporal_df = pd.read_csv("district_feature_store_v3_temporal.csv")
    
    return ranking_df, temporal_df

@st.cache_resource
def load_graph():
    return nx.read_gml("district_crop_risk_graph.gml")

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("early_warning_model.json")
    return model

try:
    df_rank, df_temporal = load_data()
    G = load_graph()
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading data/models: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("📡 Distress Watch")
st.sidebar.markdown("District-Level Predictive Engine (DLPE)")
page = st.sidebar.radio("Navigation", ["National Overview", "District Profiler", "Early Warning Simulator"])

# --- 1. National Overview ---
if page == "National Overview":
    st.title("🇮🇳 National Distress Vulnerability Map")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    avg_score = df_rank['Vulnerability_Score'].mean()
    high_risk_count = df_rank[df_rank['Vulnerability_Score'] > 0.3].shape[0]
    
    col1.metric("Avg National Vulnerability", f"{avg_score:.2f}")
    col2.metric("High Risk Districts (>0.3)", high_risk_count, delta_color="inverse")
    col3.metric("Total Districts Tracked", df_rank.shape[0])
    
    # Choropleth (Simulated with Bar Chart for now as we don't have GeoJSON loaded)
    st.subheader("Top 20 Most Vulnerable Districts")
    top_20 = df_rank.head(20)
    fig = px.bar(top_20, x='Vulnerability_Score', y='District_ID', orientation='h', 
                 color='Vulnerability_Score', color_continuous_scale='Reds',
                 title="Districts with Highest Structural Risk")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.dataframe(df_rank)

# --- 2. District Profiler ---
elif page == "District Profiler":
    st.title("🔍 District Risk Profiler")
    
    # Selectors
    all_districts = sorted(df_rank['District_ID'].unique())
    selected_dist = st.selectbox("Select District", all_districts)
    
    # District Data
    dist_data = df_rank[df_rank['District_ID'] == selected_dist].iloc[0]
    
    # Header
    score = dist_data['Vulnerability_Score']
    color = "red" if score > 0.3 else "orange" if score > 0.15 else "green"
    st.markdown(f"### Vulnerability Score: <span style='color:{color}'>{score:.3f}</span>", unsafe_allow_html=True)
    
    # 2. Risk Graph (Subgraph)
    st.subheader("🕸️ Crop Portfolio Risk Graph")
    col_graph, col_stats = st.columns([2, 1])
    
    with col_graph:
        neighbors = list(G.neighbors(selected_dist))
        subgraph = G.subgraph([selected_dist] + neighbors)
        
        # Simple Plotly Network Viz (Nodes + Edges)
        pos = nx.spring_layout(subgraph, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in subgraph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = px.line(x=edge_x, y=edge_y).data[0]
        edge_trace.line.color = '#888'
        edge_trace.line.width = 0.5
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            if node == selected_dist:
                node_color.append('red') # District
            else:
                node_color.append('green') # Crop

        fig_net = px.scatter(x=node_x, y=node_y, text=node_text, color=node_color, size=[20]*len(node_x))
        fig_net.update_traces(textposition='top center')
        fig_net.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig_net, use_container_width=True)

    with col_stats:
        st.write("**Risk Portfolio:**")
        risk_portfolio = []
        for n in neighbors:
            edge_data = G.get_edge_data(selected_dist, n)
            risk_portfolio.append({'Crop': n, 'Risk': edge_data['weight'], 'Area': edge_data['area']})
        
        df_risk = pd.DataFrame(risk_portfolio).sort_values(by='Risk', ascending=False)
        st.dataframe(df_risk, height=400)

# --- 3. Early Warning Simulator ---
elif page == "Early Warning Simulator":
    st.title("🔮 Early Warning Simulator")
    st.markdown("Predict **Next Year's MnREGA Demand** based on agricultural shocks.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User Inputs
        st.subheader("Scenario Parameters")
        input_rain = st.slider("Rainfall (Avg Annual mm)", 0, 3000, 1000)
        input_yield_change = st.slider("Yield Change (%) from Last Year", -100, 100, 0) / 100.0
        
        # Derived Shock
        is_shock = 1 if input_yield_change < -0.2 else 0
        if is_shock:
            st.error("⚠️ Severe Yield Shock Detected!")
        
        # State Selection for Context (Generic for now)
        states = [c.replace('State_Name_', '') for c in model.get_booster().feature_names if 'State_Name_' in c]
        selected_state = st.selectbox("State", states)
        
    with col2:
        # Prepare Input Vector
        # Features: ['Yield_Lag1', 'Production_Lag1', 'Yield_Pct_Change', 'Yield_Shock', 'Avg_Annual_Rainfall', 'State_Name...', 'Crop...']
        # We'll use a dummy vector with zeros and fill our inputs
        
        features = model.get_booster().feature_names
        input_data = pd.DataFrame(0, index=[0], columns=features)
        
        # Fill inputs
        # Assuming Yield_Lag1 is average yield (heuristic: 1.5)
        input_data['Yield_Lag1'] = 1.5 
        input_data['Production_Lag1'] = 5000
        input_data['Yield_Pct_Change'] = input_yield_change
        input_data['Yield_Shock'] = is_shock
        input_data['Avg_Annual_Rainfall'] = input_rain
        
        state_col = f"State_Name_{selected_state}"
        if state_col in input_data.columns:
            input_data[state_col] = 1
            
        # Predict
        prediction = model.predict(input_data)[0]
        
        st.subheader("Prediction")
        st.metric("Predicted MNREGA Job Cards", f"{int(prediction):,}")
        
        if prediction > 100000:
            st.warning("High Distress Predicted. Recommend: Pre-allocate MNREGA funds.")
        else:
            st.success("Normal Demand Predicted.")

st.sidebar.markdown("---")
st.sidebar.info("v1.0 | DLPE Prototype")
