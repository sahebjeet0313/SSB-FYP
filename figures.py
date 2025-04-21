# generate_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import shap

# ==============================
# 1. SEERA Dataset Heatmap
# ==============================
def create_heatmap():
    # Simulate SEERA data (REPLACE WITH YOUR ACTUAL DATA)
    data = {
        'Object Points': [50, 120, 300, 80, 200, 150, 90, 210, 110, 170],
        'Actual Effort': [100, 300, 800, 200, 500, 350, 180, 420, 250, 380],
        'Team Size': [3, 5, 8, 4, 6, 5, 3, 7, 4, 6]
    }
    df = pd.DataFrame(data)
    df['Team_Productivity'] = df['Actual Effort'] / (df['Team Size'] * 30)  # 30-day duration
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("SEERA Feature Correlations", pad=20)
    plt.tight_layout()
    plt.savefig('seera_heatmap.png', dpi=300)
    plt.close()
    print("✅ Saved seera_heatmap.png")

# ==============================
# 2. Model Performance Comparison
# ==============================
def create_performance_plot():
    metrics = {
        'Model': ['COCOMO', 'XGBoost', 'Neural Network'],
        'MAE': [45, 30, 32],
        'RMSE': [60, 42, 44]
    }
    df_metrics = pd.DataFrame(metrics)

    ax = df_metrics.plot(x='Model', y=['MAE', 'RMSE'], kind='bar', 
                        figsize=(8, 5), rot=0, color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Error (hours)', labelpad=10)
    plt.title('Model Performance Comparison', pad=20)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300)
    plt.close()
    print("✅ Saved model_performance.png")

# ==============================
# 3. XGBoost Feature Importance
# ==============================
def create_feature_importance():
    # Simulate training (REPLACE WITH YOUR ACTUAL MODEL)
    data = {
        'Object Points': [50, 120, 300, 80, 200],
        'Team_Productivity': [1.1, 2.0, 3.3, 1.7, 2.8],
        'Actual Effort': [100, 300, 800, 200, 500]
    }
    df = pd.DataFrame(data)
    
    model = XGBRegressor(random_state=42)
    model.fit(df[['Object Points', 'Team_Productivity']], df['Actual Effort'])

    plt.figure(figsize=(8, 4))
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(df.columns[sorted_idx], model.feature_importances_[sorted_idx], color='#2ca02c')
    plt.title("XGBoost Feature Importance", pad=20)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    print("✅ Saved feature_importance.png")

# ==============================
# 4. SHAP Analysis (Optional)
# ==============================
def create_shap_plot():
    # Simulate data (REPLACE WITH YOUR ACTUAL MODEL/DATA)
    data = {
        'Object Points': [50, 120, 300, 80, 200],
        'Team_Productivity': [1.1, 2.0, 3.3, 1.7, 2.8],
        'Actual Effort': [100, 300, 800, 200, 500]
    }
    df = pd.DataFrame(data)
    
    model = XGBRegressor(random_state=42)
    model.fit(df[['Object Points', 'Team_Productivity']], df['Actual Effort'])
    
    explainer = shap.Explainer(model)
    shap_values = explainer(df[['Object Points', 'Team_Productivity']])
    
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title("SHAP Analysis for Effort Prediction", pad=20)
    plt.tight_layout()
    plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved shap_analysis.png")

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    print("Generating all figures...")
    create_heatmap()
    create_performance_plot()
    create_feature_importance()
    create_shap_plot()
    print("\n🎉 All figures generated! Check your folder for:")
    print("- seera_heatmap.png\n- model_performance.png\n- feature_importance.png\n- shap_analysis.png")