import os
from enum import Enum

   
class Path(Enum):
    # File paths
    LOG_FILE = "logs/project.log"

    # Messages
    SUCCESS_DATASET_MESSAGE = "Dataset loaded successfully."
    FAIL_DATASET_MESSAGE = "Failed to load dataset."
    SUCCESS_MODEL_TRAINING = "Model training completed successfully."
    FAIL_MODEL_TRAINING = "Model training failed."

# Other configurations
class Config:
    
    DATASET_PATH = 'SEERA cost estimation dataset.xlsx'
    TEAM_SIZE_COL = 'Team size'  # Replace with actual column name
    DURATION_COL = 'Actual duration'  # Replace with actual column name
    DAILY_HOURS_COL = 'Daily working hours'  # Replace with actual column name
    
    # COCOMO II parameters
    COCOMO_A = 2.4  # Constant for organic projects
    COCOMO_B = 1.05  # Exponent for organic projects
    COCOMO_EAF = 1.0  # Effort Adjustment Factor (default is 1.0)

    # Conversion factors
    KLOC_CONVERSION_FACTOR = 0.2  # 1 Object Point = 0.2 KLOC
    WORKING_DAYS_PER_MONTH = 22  # 22 working days/month
    WORKING_HOURS_PER_DAY = 8  # 8 hours/day
    
    # Required columns for preprocessing
    REQUIRED_COLUMNS = ['Object points', 'Estimated effort', 'Actual effort', '% project gain (loss)']

    # Target variable for SHAP analysis
    TARGET_VARIABLE = 'Actual effort'

    # Random Forest parameters
    RANDOM_STATE = 42  # Random seed for reproducibility
    
     # Object Points calculation
    OBJECT_POINTS_SCALING_FACTOR = 1.2  # Scaling factor for Object Points

    # Effort calculation
    PART_TIME_FACTOR = 0.5  # Factor for part-time team members
    WORKING_DAYS_PER_MONTH = 22  # Number of working days per month
    
    # Neural Network parameters
    NN_EPOCHS = 200
    NN_BATCH_SIZE = 32
    NN_LEARNING_RATE = 0.001

    # XGBoost parameters
    XGB_N_ESTIMATORS = [100, 200]
    XGB_MAX_DEPTH = [3, 5]
    XGB_LEARNING_RATE = [0.01, 0.1]
    XGB_SUBSAMPLE = [0.8, 1.0]