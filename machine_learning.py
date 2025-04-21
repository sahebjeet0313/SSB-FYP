import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from CommonEnums import ModelType
from configuration import Config, Path

# Configure logging
logging.basicConfig(
    filename=Path.LOG_FILE.value,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def train_and_evaluate_models(df, important_features):
    try:
        # Validate required columns
        if 'Actual effort' not in df.columns:
            logger.error("Error: 'Actual effort' column not found in the dataset.")
            return None

        # Feature Engineering
        logger.info("Starting feature engineering...")
        df['Complexity_to_Effort_Ratio'] = df['Object points'] / df['Actual effort']
        df['Team_Productivity'] = df['Actual effort'] / (df['Team size'] * df['Actual duration'])
        df['Project_Size_Category'] = pd.cut(df['Object points'], bins=[0, 100, 500, 1000], labels=['Small', 'Medium', 'Large'])

        # Convert important_features (NumPy array) to a list
        important_features_list = important_features.tolist()

        # Prepare data for ML models
        X = df[important_features_list + ['Complexity_to_Effort_Ratio', 'Team_Productivity', 'Project_Size_Category']]
        y = df['Actual effort']

        # Convert categorical features to numerical (one-hot encoding)
        X = pd.get_dummies(X, columns=['Project_Size_Category'], drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_scaled = scaler.transform(X)  # Normalize the entire dataset

        # Train and evaluate a Deep Neural Network
        def train_deep_neural_network(X_train, y_train, X_test, y_test):
            try:
                logger.info("Training Neural Network...")
                model = Sequential()
                model.add(Input(shape=(X_train.shape[1],)))  # Input layer
                model.add(Dense(128, activation='relu'))  # First hidden layer
                model.add(Dropout(0.2))  # Dropout layer
                model.add(Dense(64, activation='relu'))  # Second hidden layer
                model.add(Dropout(0.2))  # Dropout layer
                model.add(Dense(32, activation='relu'))  # Third hidden layer
                model.add(Dropout(0.2))  # Dropout layer
                model.add(Dense(16, activation='relu'))  # Fourth hidden layer
                model.add(Dense(1, activation='linear'))  # Output layer

                model.compile(optimizer=Adam(learning_rate=Config.NN_LEARNING_RATE), loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=Config.NN_EPOCHS, batch_size=Config.NN_BATCH_SIZE, validation_split=0.2, verbose=0)

                y_pred_test = model.predict(X_test)
                y_pred_all = model.predict(X_scaled)  # Predict for the entire dataset
                logger.info("Neural Network training completed successfully.")
                return y_pred_test, y_pred_all
            except Exception as e:
                logger.error(f"Error training Neural Network: {e}", exc_info=True)
                raise

        # Train and evaluate XGBoost with Hyperparameter Tuning
        def train_xgboost(X_train, y_train, X_test, y_test):
            try:
                logger.info("Training XGBoost...")
                param_grid = {
                    'n_estimators': Config.XGB_N_ESTIMATORS,
                    'max_depth': Config.XGB_MAX_DEPTH,
                    'learning_rate': Config.XGB_LEARNING_RATE,
                    'subsample': Config.XGB_SUBSAMPLE
                }
                model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(X_train, y_train)
                y_pred_test = grid_search.predict(X_test)
                y_pred_all = grid_search.predict(X)  # Predict for the entire dataset
                logger.info("XGBoost training completed successfully.")
                return y_pred_test, y_pred_all
            except Exception as e:
                logger.error(f"Error training XGBoost: {e}", exc_info=True)
                raise

        # Get predictions from both models
        y_pred_nn_test, y_pred_nn_all = train_deep_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        y_pred_xgb_test, y_pred_xgb_all = train_xgboost(X_train, y_train, X_test, y_test)

        # Validate predictions
        if y_pred_nn_all is None or y_pred_xgb_all is None:
            logger.error("Error: Model predictions are None.")
            return None

        # Evaluate models and store results
        results = []

        def evaluate_model(y_true, y_pred, model_name):
            try:
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                results.append({'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2})
                logger.info(f"{model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")
                return y_pred
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}", exc_info=True)
                raise

        # Evaluate Neural Network on the test set
        y_pred_nn_test = evaluate_model(y_test, y_pred_nn_test, ModelType.NEURAL_NETWORK.value)

        # Evaluate XGBoost on the test set
        y_pred_xgb_test = evaluate_model(y_test, y_pred_xgb_test, ModelType.XGBOOST.value)

        # Create a DataFrame for predictions on the entire dataset
        predictions_df = pd.DataFrame({
            'Actual Effort': y,
            'Neural Network Predicted Effort': y_pred_nn_all.flatten(),  # Flatten for consistency
            'XGBoost Predicted Effort': y_pred_xgb_all.flatten()  # Flatten for consistency
        })

        # Add cost estimation
        hourly_rate = 50  # Assuming an hourly rate of $50
        predictions_df['Neural Network Predicted Cost'] = predictions_df['Neural Network Predicted Effort'] * hourly_rate
        predictions_df['XGBoost Predicted Cost'] = predictions_df['XGBoost Predicted Effort'] * hourly_rate

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Save results and predictions to an Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"model_evaluation_{timestamp}.xlsx"

        with pd.ExcelWriter(output_file) as writer:
            results_df.to_excel(writer, sheet_name='Model Evaluation', index=False)
            predictions_df.to_excel(writer, sheet_name='Predictions', index=False)

        # Validate file creation
        if os.path.exists(output_file):
            logger.info(f"Model evaluation results saved as {output_file}")
            return output_file
        else:
            logger.error(f"Failed to save model evaluation results to {output_file}")
            return None

    except Exception as e:
        logger.error(f"An error occurred in train_and_evaluate_models: {e}", exc_info=True)
        return None