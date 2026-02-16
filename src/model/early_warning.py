import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "district_feature_store_v3_temporal.csv"

def train_early_warning_model():
    logging.info(f"📂 Loading Temporal Feature Store: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        logging.error("File not found.")
        return

    # Scenario: "Predict Distress in Year T using ONLY Data from Year T-1"
    # Target: MNREGA_JobCards_Issued (Year T) - Wait, our MNREGA is static snapshot.
    # Limitation: Since MNREGA is static in this dataset, we can't truly predict "Future MNREGA" from "Past Yield".
    # BUT, we can test: "Does Past Yield (T-1) correlate with Current Static Distress better than Current Yield?"
    # Probably not, but it tests the "Lag Effect" hypothesis.
    
    # In a real dynamic dataset: Target = MNREGA(T), Features = Yield(T-1).
    # Here: Target = MNREGA(Static), Features = Yield(T-1).
    
    target_col = 'MNREGA_JobCards_Issued'
    
    # Features: Lagged Agronomy + Static Climate
    feature_cols = ['Yield_Lag1', 'Production_Lag1', 'Yield_Pct_Change', 'Yield_Shock', 
                    'Avg_Annual_Rainfall', 'State_Name', 'Crop']
    
    logging.info(f"🎯 Target: {target_col}")
    logging.info(f"⏳ Features (Temporal): {feature_cols}")
    
    df = df.dropna(subset=[target_col, 'Yield_Lag1'])
    
    # Clean Infinite values (from Pct Change division by zero)
    import numpy as np
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True) # Safety fill for any other NaNs
    
    X = pd.get_dummies(df[feature_cols], columns=['State_Name', 'Crop'], drop_first=True)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info(f"🔮 Training Early Warning Model (XGBoost)...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    logging.info(f"✅ Early Warning Model Results:")
    logging.info(f"   R² Score: {r2:.4f} (Prediction based on PAST year's yield)")
    
    # Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(10)
    logging.info("\n🔝 Top Predictive Signals (Leading Indicators):")
    logging.info(feat_imp.to_string(index=False))
    
    # Visualization
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=preds, alpha=0.3, color='orange')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual MNREGA Demand')
        plt.ylabel('Predicted Demand (using T-1 Signals)')
        plt.title(f'Early Warning System: Predicting Distress using Lagged Signals\nR² = {r2:.4f}')
        plt.tight_layout()
        plt.savefig('early_warning_plot.png')
        logging.info("📊 Saved plot to early_warning_plot.png")
    except Exception as e:
        logging.warning(f"Plotting failed: {e}")

    # Save Model for Dashboard
    model.save_model("early_warning_model.json")
    logging.info("💾 Saved XGBoost model to early_warning_model.json")

if __name__ == "__main__":
    train_early_warning_model()
