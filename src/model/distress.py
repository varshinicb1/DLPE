import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "district_feature_store_v2_mnrega.csv"

def train_distress_model():
    logging.info(f"📂 Loading Feature Store: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        logging.error("File not found.")
        return

    # Drop rows where Target is missing or 0 (assuming 0 might be missing data for this analysis)
    # Actually, let's keep 0 if it's substantial, but pure missing should be dropped.
    df = df.dropna(subset=['MNREGA_JobCards_Issued', 'Yield', 'Avg_Annual_Rainfall'])
    
    # Target: MNREGA Demand (Job Cards Issued)
    target_col = 'MNREGA_JobCards_Issued'
    
    # Features: Agriculture & Climate
    feature_cols = ['Area', 'Production', 'Yield', 'Avg_Annual_Rainfall', 'Crop', 'State_Name', 'Season']
    
    logging.info(f"🎯 Target Variable: {target_col}")
    logging.info(f"✨ Features: {feature_cols}")
    
    # One-Hot Encoding for Categorical
    X = pd.get_dummies(df[feature_cols], columns=['Crop', 'State_Name', 'Season'], drop_first=True)
    y = df[target_col]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info(f"🧠 Training XGBoost Regressor on {len(X_train)} samples...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    logging.info(f"✅ Model Results:")
    logging.info(f"   MSE: {mse:,.2f}")
    logging.info(f"   R² Score: {r2:.4f}")
    
    if r2 > 0.1:
        logging.info("🚀 POSITIVE SIGNAL: Agricultural features have predictive power for MNREGA demand!")
    else:
        logging.warning("⚠️ WEAK SIGNAL: Baseline model needs more temporal features (Lagged Rainfall, etc).")

    # Feature Importance
    importance = model.feature_importances_
    feats = X.columns
    feat_imp = pd.DataFrame({'Feature': feats, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(10)
    logging.info("\n🔝 Top 10 Drivers of Distress (MNREGA Demand):")
    logging.info(feat_imp.to_string(index=False))

    # --- Visualization ---
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=preds, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual MNREGA Job Cards Issued')
        plt.ylabel('Predicted Demand (Based on Crop/Rainfall)')
        plt.title(f'Agri-Distress Model: Actual vs Predicted MNREGA Demand\nR² = {r2:.4f}')
        plt.tight_layout()
        plt.savefig('mnrega_prediction_plot.png')
        logging.info("📊 Saved visualization to mnrega_prediction_plot.png")
    except ImportError:
        logging.warning("⚠️ Matplotlib/Seaborn not found. Skipping plot.")

if __name__ == "__main__":
    train_distress_model()
