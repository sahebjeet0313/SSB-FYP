import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime
from configuration import Config, Path
from CommonEnums import ModelType

# Configure logging
logging.basicConfig(
    filename=Path.LOG_FILE.value,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def calculate_weights(df, ml_predictions, cocomo_effort, dynamic_effort):
    """
    Calculate adaptive weights for each estimation method based on their performance
    with enhanced weighting for XGBoost
    """
    actual_effort = df['Actual effort']
    
    # Calculate error metrics for each method
    r2_nn = r2_score(actual_effort, ml_predictions['Neural Network'])
    r2_xgb = r2_score(actual_effort, ml_predictions['XGBoost'])
    r2_cocomo = r2_score(actual_effort, cocomo_effort)
    r2_dynamic = r2_score(actual_effort, dynamic_effort)
    
    # Calculate RMSE for each method
    rmse_nn = np.sqrt(mean_squared_error(actual_effort, ml_predictions['Neural Network']))
    rmse_xgb = np.sqrt(mean_squared_error(actual_effort, ml_predictions['XGBoost']))
    rmse_cocomo = np.sqrt(mean_squared_error(actual_effort, cocomo_effort))
    rmse_dynamic = np.sqrt(mean_squared_error(actual_effort, dynamic_effort))
    
    # Combine R² and RMSE into a composite score (higher is better)
    composite_nn = (r2_nn / (1 + rmse_nn)) if rmse_nn > 0 else r2_nn
    composite_xgb = (r2_xgb / (1 + rmse_xgb)) if rmse_xgb > 0 else r2_xgb
    composite_cocomo = (r2_cocomo / (1 + rmse_cocomo)) if rmse_cocomo > 0 else r2_cocomo
    composite_dynamic = (r2_dynamic / (1 + rmse_dynamic)) if rmse_dynamic > 0 else r2_dynamic
    
    # Apply XGBoost performance boost
    xgboost_boost_factor = 1.5  # Increase weight for XGBoost
    
    # Convert composite scores to weights
    weights = {
        'Neural Network': max(composite_nn, 0),
        'XGBoost': max(composite_xgb * xgboost_boost_factor, 0),  # Apply boost factor
        'COCOMO': max(composite_cocomo, 0),
        'Dynamic': max(composite_dynamic, 0)
    }
    
    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    else:
        # If all weights are 0, give higher weight to XGBoost
        weights = {
            'Neural Network': 0.2,
            'XGBoost': 0.5,  # Higher default weight for XGBoost
            'COCOMO': 0.15,
            'Dynamic': 0.15
        }
    
    return weights

def project_complexity_score(row):
    """
    Calculate a complexity score for the project based on various factors
    """
    # Normalize and combine different factors
    size_factor = row['Object points'] / 1000  # Normalize size
    team_factor = row['Team size'] / 10  # Normalize team size
    duration_factor = row['Actual duration'] / 12  # Normalize duration (assuming months)
    
    # Combine factors with weights
    complexity_score = (size_factor * 0.4 + 
                       team_factor * 0.3 + 
                       duration_factor * 0.3)
    
    return complexity_score

def adjust_weights_by_complexity(weights, complexity_score):
    """
    Adjust weights based on project complexity with enhanced XGBoost preference
    """
    if complexity_score > 0.8:  # High complexity
        # Favor ML models, especially XGBoost for complex projects
        weights['Neural Network'] *= 1.1
        weights['XGBoost'] *= 1.4  # Increased boost for complex projects
        weights['COCOMO'] *= 0.7
        weights['Dynamic'] *= 0.7
    elif complexity_score < 0.3:  # Low complexity
        # Still maintain XGBoost influence in simple projects
        weights['Neural Network'] *= 0.9
        weights['XGBoost'] *= 1.2  # Maintain strong XGBoost presence
        weights['COCOMO'] *= 1.0
        weights['Dynamic'] *= 1.0
    
    # Renormalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    return weights

def hybrid_estimation(df, ml_predictions, cocomo_effort, dynamic_effort):
    """
    Combine different estimation approaches using adaptive weighting with XGBoost emphasis
    """
    try:
        logger.info("Starting hybrid estimation with XGBoost emphasis...")
        
        # Initialize results dataframe
        results_df = pd.DataFrame()
        results_df['Actual Effort'] = df['Actual effort']
        results_df['Neural Network'] = ml_predictions['Neural Network']
        results_df['XGBoost'] = ml_predictions['XGBoost']
        results_df['COCOMO'] = cocomo_effort
        results_df['Dynamic'] = dynamic_effort
        
        # Calculate initial weights with XGBoost emphasis
        initial_weights = calculate_weights(df, {
            'Neural Network': ml_predictions['Neural Network'],
            'XGBoost': ml_predictions['XGBoost']
        }, cocomo_effort, dynamic_effort)
        
        # Calculate hybrid estimates for each project
        hybrid_estimates = []
        pure_xgboost = []  # Track XGBoost estimates for comparison
        
        for idx, row in df.iterrows():
            complexity = project_complexity_score(row)
            weights = adjust_weights_by_complexity(initial_weights.copy(), complexity)
            
            # Calculate weighted estimate
            estimate = (
                weights['Neural Network'] * ml_predictions['Neural Network'][idx] +
                weights['XGBoost'] * ml_predictions['XGBoost'][idx] +
                weights['COCOMO'] * cocomo_effort[idx] +
                weights['Dynamic'] * dynamic_effort[idx]
            )
            
            hybrid_estimates.append(estimate)
            pure_xgboost.append(ml_predictions['XGBoost'][idx])
        
        results_df['Hybrid Estimate'] = hybrid_estimates
        
        # Compare hybrid with XGBoost
        mae_hybrid = mean_absolute_error(results_df['Actual Effort'], results_df['Hybrid Estimate'])
        mae_xgboost = mean_absolute_error(results_df['Actual Effort'], pure_xgboost)
        
        rmse_hybrid = np.sqrt(mean_squared_error(results_df['Actual Effort'], results_df['Hybrid Estimate']))
        rmse_xgboost = np.sqrt(mean_squared_error(results_df['Actual Effort'], pure_xgboost))
        
        r2_hybrid = r2_score(results_df['Actual Effort'], results_df['Hybrid Estimate'])
        r2_xgboost = r2_score(results_df['Actual Effort'], pure_xgboost)
        
        logger.info(f"Hybrid vs XGBoost Comparison:")
        logger.info(f"Hybrid  - MAE: {mae_hybrid:.2f}, RMSE: {rmse_hybrid:.2f}, R²: {r2_hybrid:.2f}")
        logger.info(f"XGBoost - MAE: {mae_xgboost:.2f}, RMSE: {rmse_xgboost:.2f}, R²: {r2_xgboost:.2f}")
        
        # Save results to Excel with comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"hybrid_estimation_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file) as writer:
            results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Save performance metrics with comparison
            metrics_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'R²'],
                'Hybrid': [mae_hybrid, rmse_hybrid, r2_hybrid],
                'XGBoost': [mae_xgboost, rmse_xgboost, r2_xgboost]
            })
            metrics_df.to_excel(writer, sheet_name='Performance Comparison', index=False)
            
            # Save weights
            weights_df = pd.DataFrame(list(initial_weights.items()), columns=['Method', 'Weight'])
            weights_df.to_excel(writer, sheet_name='Model Weights', index=False)
        
        logger.info(f"Hybrid estimation results saved as {output_file}")
        return results_df, output_file
        
    except Exception as e:
        logger.error(f"Error in hybrid estimation: {e}", exc_info=True)
        return None, None

def analyze_estimation_patterns(results_df):
    """
    Analyze patterns in estimation accuracy across different project characteristics
    """
    try:
        patterns = {
            'small_projects': results_df[results_df['Actual Effort'] < results_df['Actual Effort'].median()],
            'large_projects': results_df[results_df['Actual Effort'] >= results_df['Actual Effort'].median()]
        }
        
        analysis = {}
        for project_type, data in patterns.items():
            analysis[project_type] = {
                'mae': mean_absolute_error(data['Actual Effort'], data['Hybrid Estimate']),
                'rmse': np.sqrt(mean_squared_error(data['Actual Effort'], data['Hybrid Estimate'])),
                'r2': r2_score(data['Actual Effort'], data['Hybrid Estimate'])
            }
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error in pattern analysis: {e}", exc_info=True)
        return None 