# Figure 1: Feature Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Simulate SEERA data (replace with your actual data)
data = {
    'Object Points': [50, 120, 300, 80, 200],
    'Actual Effort': [100, 300, 800, 200, 500],
    'Team Size': [3, 5, 8, 4, 6]
}
df = pd.DataFrame(data)
df['Team_Productivity'] = df['Actual Effort'] / (df['Team Size'] * 30)  # Assuming 30-day duration

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("SEERA Feature Correlations")
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')

# Figure 2: Model Performance Comparison
metrics = {
    'Model': ['COCOMO', 'XGBoost', 'Neural Network'],
    'MAE': [45, 30, 32],
    'RMSE': [60, 42, 44]
}
df_metrics = pd.DataFrame(metrics)

df_metrics.plot(x='Model', y=['MAE', 'RMSE'], kind='bar', figsize=(8,5))
plt.ylabel('Error (hours)')
plt.title('Model Performance Comparison')
plt.savefig('model_performance.png', dpi=300)

# Figure 3: Feature Importance (XGBoost Example)
from xgboost import XGBRegressor
model = XGBRegressor().fit(df[['Object Points', 'Team_Productivity']], df['Actual Effort'])

plt.figure(figsize=(8,6))
sorted_idx = model.feature_importances_.argsort()
plt.barh(df.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.title("XGBoost Feature Importance")
plt.savefig('feature_importance.png', dpi=300)