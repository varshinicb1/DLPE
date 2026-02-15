import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import logging
import seaborn as sns

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_GRAPH = "district_crop_risk_graph.gml"
INPUT_RANKING = "district_vulnerability_ranking.csv"
OUTPUT_PLOT_RANKING = "vulnerability_ranking_plot.png"
OUTPUT_PLOT_GRAPH = "district_risk_subgraph.png"

def visualize_results():
    # 1. Visualize Vulnerability Ranking
    logging.info(f"📊 Visualizing Vulnerability Ranking from {INPUT_RANKING}...")
    try:
        df = pd.read_csv(INPUT_RANKING)
        top_10 = df.head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Vulnerability_Score', y='District_ID', data=top_10, palette='viridis')
        plt.xlabel('Structural Vulnerability Score (Risk-Weighted Crop Portfolio)')
        plt.ylabel('District')
        plt.title('Top 10 Most Vulnerable Districts (based on Crop Risk)')
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_RANKING)
        logging.info(f"💾 Saved Ranking Plot to {OUTPUT_PLOT_RANKING}")
    except FileNotFoundError:
        logging.error("Ranking CSV not found.")

    # 2. Visualize Graph (Subgraph for Top Risk District)
    logging.info(f"🕸️ Visualizing Subgraph from {INPUT_GRAPH}...")
    try:
        G = nx.read_gml(INPUT_GRAPH)
        
        # Pick the most vulnerable district
        top_district = df.iloc[0]['District_ID'] # e.g., GARHWA_JHARKHAND
        logging.info(f"   Focusing on District: {top_district}")
        
        # Get neighbors (Crops)
        neighbors = list(G.neighbors(top_district))
        subgraph_nodes = [top_district] + neighbors
        subG = G.subgraph(subgraph_nodes)
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subG, k=0.5, seed=42)
        
        # Draw District Node
        nx.draw_networkx_nodes(subG, pos, nodelist=[top_district], node_color='red', node_size=3000, label='District')
        
        # Draw Crop Nodes
        # Color by Risk (Edge Weight)
        edge_weights = [subG[u][v]['weight'] for u, v in subG.edges()]
        
        nx.draw_networkx_nodes(subG, pos, nodelist=neighbors, node_color='green', node_size=1500, label='Crop')
        
        # Edges
        edges = nx.draw_networkx_edges(subG, pos, edge_color=edge_weights, edge_cmap=plt.cm.Reds, width=2, edge_vmin=0, edge_vmax=1)
        
        # Labels
        nx.draw_networkx_labels(subG, pos, font_size=10, font_weight='bold')
        
        plt.colorbar(edges, label='Crop Failure Probability')
        plt.title(f'Risk Profile: {top_district}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_GRAPH)
        logging.info(f"💾 Saved Subgraph Plot to {OUTPUT_PLOT_GRAPH}")
        
    except FileNotFoundError:
        logging.error("Graph GML not found.")
    except Exception as e:
        logging.error(f"Graph visualization failed: {e}")

if __name__ == "__main__":
    visualize_results()
