import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import shap
import os
from datetime import datetime
import logging
from configuration import Config, Path
from CommonEnums import ErrorMessages

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Use the logger without reconfiguring it
logger = logging.getLogger(__name__)

def preprocess_data(file_path):
    try:
        logger.info("Starting data preprocessing...")

        # Load the dataset
        logger.info(f"Loading dataset from {file_path}...")
        df = pd.read_excel(file_path, sheet_name='SEERA dataset', header=1)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

        # Print column names for debugging
        logger.info("Column names in the dataset:")
        logger.info(df.columns.tolist())

        # Ensure the required columns exist
        required_columns = Config.REQUIRED_COLUMNS
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"'{col}' not found in dataset. Creating default column with value 0.")
                df[col] = 0  # Default value to ensure presence

        # Handle missing values
        logger.info("Handling missing values...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        logger.info("Missing values handled successfully.")

        # Label encode categorical columns
        logger.info("Label encoding categorical columns...")
        label_encoders = {}
        categorical_columns = df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        logger.info("Categorical columns label encoded successfully.")

        # Train a Random Forest model for SHAP analysis
        logger.info("Training Random Forest model for SHAP analysis...")
        target_variable = Config.TARGET_VARIABLE
        X = df[numeric_columns].drop(columns=[target_variable])
        y = df[target_variable]

        model = RandomForestRegressor(random_state=Config.RANDOM_STATE)
        model.fit(X, y)
        logger.info("Random Forest model trained successfully.")

        # Compute SHAP values
        logger.info("Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        logger.info("SHAP values computed successfully.")

        # Get the mean absolute SHAP values for feature importances
        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Get the sorted indices of feature importances
        sorted_idx = np.argsort(mean_shap_values)[::-1]

        # Convert columns to a numpy array to handle multi-dimensional indexing
        important_features = np.array(numeric_columns)[sorted_idx]

        # Ensure required columns are always included in important features
        for col in required_columns:
            if col not in important_features:
                important_features = np.append(important_features, col)

        logger.info("Most important features based on SHAP values:")
        logger.info(important_features.tolist())

        # Save the processed file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"processed_dataset_{timestamp}.xlsx"
        df.to_excel(output_file, index=False)
        logger.info(f"Processed dataset saved as {output_file}")

        return df, important_features, None, label_encoders  # Return None for scaler

    except Exception as e:
        logger.error(f"An error occurred in preprocess_data: {e}", exc_info=True)
        raise