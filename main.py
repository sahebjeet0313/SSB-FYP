import os
import logging
import pandas as pd
from preprocessing import preprocess_data
from cocomo_model import apply_cocomo_model
from machine_learning import train_and_evaluate_models
from dynamic_estimation import dynamic_cost_estimation
from hybrid_estimation import hybrid_estimation, analyze_estimation_patterns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from configuration import Config, Path
from CommonEnums import ErrorMessages

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging only once in main.py
logging.basicConfig(
    filename=Path.LOG_FILE.value,  # Logs will be saved to logs/project.log
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_comparison_plots(results_df, df_index):
    """
    Create separate comparison plots for each estimation method vs actual effort
    """
    plt.style.use('seaborn')
    methods = {
        'COCOMO': 'COCOMO',
        'Neural Network': 'Neural Network',
        'XGBoost': 'XGBoost',
        'Dynamic': 'Dynamic',
        'Hybrid Estimate': 'Hybrid'
    }
    
    for method_key, method_name in methods.items():
        plt.figure(figsize=(12, 6))
        
        # Plot actual effort with solid line
        plt.plot(df_index, results_df['Actual Effort'], 
                label='Actual Effort', 
                color='blue', 
                linewidth=2, 
                linestyle='-')
        
        # Plot estimation method with dashed line
        plt.plot(df_index, results_df[method_key], 
                label=f'{method_name} Estimation', 
                color='red', 
                linewidth=2, 
                linestyle='--',
                alpha=0.7)
        
        plt.xlabel('Project Index')
        plt.ylabel('Effort (person-hours)')
        plt.title(f'Actual Effort vs {method_name} Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient and metrics
        correlation = np.corrcoef(results_df['Actual Effort'], results_df[method_key])[0,1]
        mae = mean_absolute_error(results_df['Actual Effort'], results_df[method_key])
        rmse = np.sqrt(mean_squared_error(results_df['Actual Effort'], results_df[method_key]))
        r2 = r2_score(results_df['Actual Effort'], results_df[method_key])
        
        # Add metrics text box
        metrics_text = f'Correlation: {correlation:.3f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}'
        plt.text(0.02, 0.98, metrics_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.savefig(f'comparison_{method_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot for each method
        plt.figure(figsize=(8, 8))
        plt.scatter(results_df['Actual Effort'], results_df[method_key], 
                   alpha=0.5, color='blue')
        
        # Add perfect prediction line
        max_val = max(results_df['Actual Effort'].max(), results_df[method_key].max())
        min_val = min(results_df['Actual Effort'].min(), results_df[method_key].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Effort')
        plt.ylabel(f'{method_name} Estimated Effort')
        plt.title(f'{method_name} Estimation vs Actual Effort')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics text box to scatter plot
        plt.text(0.02, 0.98, metrics_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save scatter plot
        plt.savefig(f'scatter_{method_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    try:
        # Step 1: Preprocess the data
        file_path = Config.DATASET_PATH
        logger.info("Starting data preprocessing from main.py...")
        df, important_features, scaler, label_encoders = preprocess_data(file_path)
        logger.info(f"DataFrame shape after preprocessing: {df.shape} in main.py")
        if df.empty:
            logger.error("Error: DataFrame is empty after preprocessing in main.py.")
            return
        logger.info("Data preprocessing completed successfully from main.py.")

        # Step 2: Apply COCOMO model
        logger.info("Applying COCOMO model from main.py...")
        df = apply_cocomo_model(df)
        logger.info(f"DataFrame shape after applying COCOMO model: {df.shape} in main.py")
        logger.info("COCOMO model applied successfully from main.py.")

        # Step 3: Train and evaluate machine learning models
        logger.info("Training and evaluating machine learning models from main.py...")
        evaluation_file_path = train_and_evaluate_models(df, important_features)
        if evaluation_file_path:
            logger.info(f"Model evaluation results saved as {evaluation_file_path} in main.py")
        else:
            logger.error("Failed to save model evaluation results from main.py.")
            raise ValueError("Model evaluation file path is None (main.py).")

        # Load ML predictions
        try:
            predictions_df = pd.read_excel(evaluation_file_path, sheet_name=1)  # Load the second sheet
            ml_predictions = {
                'Neural Network': predictions_df['Neural Network Predicted Effort'].values,
                'XGBoost': predictions_df['XGBoost Predicted Effort'].values
            }
            logger.info("ML predictions loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load predictions from {evaluation_file_path}: {e} (main.py)")
            return

        # Step 4: Dynamic cost estimation
        team_size_col = Config.TEAM_SIZE_COL
        duration_col = Config.DURATION_COL
        daily_hours_col = Config.DAILY_HOURS_COL
        logger.info("Performing dynamic cost estimation...(main.py)")
        df = dynamic_cost_estimation(df, team_size_col, duration_col, daily_hours_col)
        logger.info(f"DataFrame shape after dynamic cost estimation: {df.shape} (main.py)")
        logger.info("Dynamic cost estimation completed successfully from main.py.")

        # Step 5: Apply Hybrid Estimation
        logger.info("Starting hybrid estimation...")
        results_df, hybrid_output_file = hybrid_estimation(
            df,
            ml_predictions,
            df['COCOMO Effort (person-hours)'].values,
            df['Dynamic Effort'].values
        )

        if results_df is not None:
            # Analyze estimation patterns
            pattern_analysis = analyze_estimation_patterns(results_df)
            if pattern_analysis:
                logger.info("Estimation Pattern Analysis:")
                for project_type, metrics in pattern_analysis.items():
                    logger.info(f"{project_type}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.2f}")

            # Create separate comparison plots
            create_comparison_plots(results_df, df.index)
            logger.info("Individual comparison plots have been created successfully")

        else:
            logger.error("Hybrid estimation failed")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()