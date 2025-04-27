import pandas as pd
import numpy as np
from datetime import datetime

def create_final_summary(df, ml_predictions, cocomo_effort, dynamic_effort, hybrid_estimates):
    """
    Create a comprehensive summary CSV with all features, efforts, and metrics
    """
    # Create summary DataFrame with original features
    summary_df = df.copy()
    
    # Add model efforts
    summary_df['COCOMO_Effort'] = cocomo_effort
    summary_df['Neural_Network_Effort'] = ml_predictions['Neural Network']
    summary_df['XGBoost_Effort'] = ml_predictions['XGBoost']
    summary_df['Dynamic_Effort'] = dynamic_effort
    summary_df['Hybrid_Effort'] = hybrid_estimates
    
    # Add engineered features
    summary_df['Complexity_to_Effort_Ratio'] = df['Object points'] / df['Actual effort']
    summary_df['Team_Productivity'] = df['Actual effort'] / (df['Team size'] * df['Actual duration'])
    summary_df['Project_Size_Category'] = pd.cut(df['Object points'], 
                                               bins=[0, 100, 500, 1000], 
                                               labels=['Small', 'Medium', 'Large'])
    
    # Calculate metrics for each model
    metrics_dict = {}
    actual_effort = df['Actual effort']
    
    models = {
        'COCOMO': cocomo_effort,
        'Neural_Network': ml_predictions['Neural Network'],
        'XGBoost': ml_predictions['XGBoost'],
        'Dynamic': dynamic_effort,
        'Hybrid': hybrid_estimates
    }
    
    for model_name, predictions in models.items():
        mae = np.mean(np.abs(actual_effort - predictions))
        mse = np.mean((actual_effort - predictions) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((actual_effort - predictions) ** 2) / 
                  np.sum((actual_effort - actual_effort.mean()) ** 2))
        
        metrics_dict[f'{model_name}_MAE'] = mae
        metrics_dict[f'{model_name}_RMSE'] = rmse
        metrics_dict[f'{model_name}_R2'] = r2
    
    # Add metrics to a separate DataFrame
    metrics_df = pd.DataFrame([metrics_dict])
    
    # Calculate cost estimates (assuming $50 per hour)
    hourly_rate = 50
    summary_df['XGBoost_Cost_Estimate'] = summary_df['XGBoost_Effort'] * hourly_rate
    summary_df['Hybrid_Cost_Estimate'] = summary_df['Hybrid_Effort'] * hourly_rate
    
    # Save summary to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"final_summary_{timestamp}.csv"
    metrics_file = f"model_metrics_{timestamp}.csv"
    
    summary_df.to_csv(summary_file, index=False)
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"Summary saved as {summary_file}")
    print(f"Metrics saved as {metrics_file}")
    
    return summary_file, metrics_file 