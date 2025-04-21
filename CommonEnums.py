from enum import Enum

class ErrorMessages(Enum):
    MISSING_ACTUAL_EFFORT_COLUMN = "Error: 'Actual Effort' column not found in the dataset."
    MISSING_REQUIRED_COLUMNS = "Error: Required columns are missing in the dataset."
    INVALID_KLOC_CONVERSION = "Error: Invalid KLOC conversion factor."
    COCOMO_EFFORT_CALCULATION_FAILED = "Error: Failed to calculate COCOMO effort."
    FILE_SAVE_FAILED = "Error: Failed to save the dataset."
    INVALID_DATASET_PATH = "Error: Invalid dataset file path."
    SHAP_VALUE_COMPUTATION_FAILED = "Error: Failed to compute SHAP values."
    OBJECT_POINTS_CALCULATION_FAILED = "Error: Failed to calculate Object Points."
    ESTIMATED_EFFORT_CALCULATION_FAILED = "Error: Failed to calculate Estimated Effort."
    ACTUAL_EFFORT_CALCULATION_FAILED = "Error: Failed to calculate Actual Effort."
    PROJECT_GAIN_LOSS_CALCULATION_FAILED = "Error: Failed to calculate Project Gain/Loss."
    MISSING_MODEL_PREDICTIONS = "Error: Model prediction columns (e.g., 'XGBoost Effort', 'Neural Network Effort') not found in the dataset."

class ModelType(Enum):
    NEURAL_NETWORK = "Neural Network"
    XGBOOST = "XGBoost"

class LogLevel(Enum):
    INFO = "INFO"
    ERROR = "ERROR"
    DEBUG = "DEBUG"