import pandas as pd
import networkx as nx
import logging
import matplotlib.pyplot as plt

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "district_feature_store_v3_temporal.csv"
OUTPUT_GRAPH = "district_crop_risk_graph.gml"
OUTPUT_RANKING = "district_vulnerability_ranking.csv"

def build_knowledge_graph():
    logging.info(f"📂 Loading Feature Store: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        logging.error("File not found.")
        return

    # 1. Aggregate Risk Metrics per (District, Crop) Pair
    # We want to know: "How often does this Crop fail in this District?"
    # Shock = Yield Drop > 20% (defined in V3 features)
    
    logging.info("⚙️ Aggregating Crop Risk Profiles...")
    
    # Group by District+Crop using the boolean 'Yield_Shock'
    risk_profile = df.groupby(['State_Name', 'District_Name', 'Crop']).agg(
        Total_Years=('Crop_Year', 'count'),
        Shock_Years=('Yield_Shock', 'sum'),
        Avg_Area=('Area', 'mean')
    ).reset_index()
    
    risk_profile['Risk_Probability'] = risk_profile['Shock_Years'] / risk_profile['Total_Years']
    
    # Filter out insignificant crops (e.g., grown for < 3 years or tiny area) 
    # to keep the graph clean
    risk_profile = risk_profile[risk_profile['Total_Years'] >= 3]
    
    # 2. Build Bipartite Graph (District <-> Crop)
    logging.info(f"🕸️ Building Graph from {len(risk_profile)} associations...")
    G = nx.Graph()
    
    for _, row in risk_profile.iterrows():
        dist_node = f"{row['District_Name']}_{row['State_Name']}" # Unique ID
        crop_node = row['Crop']
        
        # Add Nodes
        G.add_node(dist_node, type='District', state=row['State_Name'])
        G.add_node(crop_node, type='Crop')
        
        # Add Edge (Weighted by Risk)
        # Weight = Risk Probability (0.0 to 1.0)
        # Attribute = Area (Importance)
        G.add_edge(dist_node, crop_node, 
                   weight=row['Risk_Probability'], 
                   area=row['Avg_Area'])

    logging.info(f"✅ Graph Created: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
    
    # 3. Calculate Structural Vulnerability
    # Logic: A District is vulnerable if it depends on High-Risk Crops.
    # Score = Sum(Crop_Risk * Crop_Area) / Total_District_Area
    
    logging.info("🧮 Calculating District Vulnerability Scores...")
    dist_vulnerability = []
    
    districts = [n for n, attr in G.nodes(data=True) if attr['type'] == 'District']
    
    for dist in districts:
        edges = G.edges(dist, data=True)
        total_risk_exposure = 0
        total_area = 0
        
        for _, crop, data in edges:
            risk = data['weight']
            area = data['area']
            
            total_risk_exposure += (risk * area)
            total_area += area
            
        if total_area > 0:
            vuln_score = total_risk_exposure / total_area
        else:
            vuln_score = 0
            
        dist_vulnerability.append({
            'District_ID': dist,
            'Vulnerability_Score': vuln_score,
            'Total_Area': total_area
        })
        
    # Save Ranking
    vuln_df = pd.DataFrame(dist_vulnerability).sort_values(by='Vulnerability_Score', ascending=False)
    vuln_df.to_csv(OUTPUT_RANKING, index=False)
    logging.info(f"💾 Saved Vulnerability Ranking to {OUTPUT_RANKING}")
    
    # Top 10 Most Vulnerable Districts
    logging.info("\n🚨 Top 10 Most Vulnerable Districts (based on Crop Portfolio Risk):")
    logging.info(vuln_df.head(10).to_string(index=False))
    
    # Save Graph
    nx.write_gml(G, OUTPUT_GRAPH)
    logging.info(f"💾 Saved Knowledge Graph to {OUTPUT_GRAPH}")

if __name__ == "__main__":
    build_knowledge_graph()
