# API Documentation: Software Effort Estimation System

## Core Modules API Reference

### 1. Preprocessing Module (`preprocessing.py`)

#### `preprocess_data(file_path: str) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, Dict]`
Preprocesses the raw dataset for model training and estimation.

**Parameters:**
- `file_path` (str): Path to the input dataset

**Returns:**
- `DataFrame`: Processed dataset
- `ndarray`: Important features
- `StandardScaler`: Fitted scaler object
- `Dict`: Label encoders for categorical variables

**Example:**
```python
df, features, scaler, encoders = preprocess_data("path/to/dataset.xlsx")
```

### 2. COCOMO Model (`cocomo_model.py`)

#### `apply_cocomo_model(df: pd.DataFrame) -> pd.DataFrame`
Applies the COCOMO model to estimate software effort.

**Parameters:**
- `df` (DataFrame): Preprocessed project data

**Returns:**
- `DataFrame`: Original dataframe with COCOMO estimates added

**Example:**
```python
df_with_cocomo = apply_cocomo_model(preprocessed_df)
```

### 3. Machine Learning Module (`machine_learning.py`)

#### `train_and_evaluate_models(df: pd.DataFrame, important_features: List) -> str`
Trains and evaluates Neural Network and XGBoost models.

**Parameters:**
- `df` (DataFrame): Processed dataset
- `important_features` (List): List of feature names

**Returns:**
- `str`: Path to the evaluation results file

**Example:**
```python
eval_file = train_and_evaluate_models(df, features)
```

### 4. Dynamic Estimation (`dynamic_estimation.py`)

#### `dynamic_cost_estimation(df: pd.DataFrame, team_size_col: str, duration_col: str, daily_hours_col: str) -> pd.DataFrame`
Calculates effort based on resource allocation.

**Parameters:**
- `df` (DataFrame): Project data
- `team_size_col` (str): Team size column name
- `duration_col` (str): Duration column name
- `daily_hours_col` (str): Daily hours column name

**Returns:**
- `DataFrame`: Updated dataframe with dynamic estimates

**Example:**
```python
df_with_dynamic = dynamic_cost_estimation(df, 'Team_Size', 'Duration', 'Daily_Hours')
```

### 5. Hybrid Estimation (`hybrid_estimation.py`)

#### `hybrid_estimation(df: pd.DataFrame, ml_predictions: Dict, cocomo_effort: np.ndarray, dynamic_effort: np.ndarray) -> Tuple[pd.DataFrame, str]`
Combines multiple estimation methods using adaptive weighting.

**Parameters:**
- `df` (DataFrame): Project data
- `ml_predictions` (Dict): ML model predictions
- `cocomo_effort` (ndarray): COCOMO estimates
- `dynamic_effort` (ndarray): Dynamic estimates

**Returns:**
- `DataFrame`: Results with all estimates
- `str`: Path to output file

**Example:**
```python
results_df, output_file = hybrid_estimation(df, ml_pred, cocomo_est, dynamic_est)
```

#### `calculate_weights(df: pd.DataFrame, ml_predictions: Dict, cocomo_effort: np.ndarray, dynamic_effort: np.ndarray) -> Dict`
Calculates adaptive weights for different methods.

**Parameters:**
- `df` (DataFrame): Project data
- `ml_predictions` (Dict): ML model predictions
- `cocomo_effort` (ndarray): COCOMO estimates
- `dynamic_effort` (ndarray): Dynamic estimates

**Returns:**
- `Dict`: Calculated weights for each method

### 6. Visualization Functions

#### `create_comparison_plots(results_df: pd.DataFrame, df_index: pd.Index) -> None`
Generates comparison plots for all estimation methods.

**Parameters:**
- `results_df` (DataFrame): Results containing all estimates
- `df_index` (Index): Project indices

**Example:**
```python
create_comparison_plots(results_df, df.index)
```

## Configuration

### Environment Variables
```python
DATASET_PATH = "path/to/dataset.xlsx"
LOG_FILE = "logs/project.log"
TEAM_SIZE_COL = "Team_Size"
DURATION_COL = "Duration"
DAILY_HOURS_COL = "Daily_Hours"
```

### Model Configuration
```python
# Neural Network
NN_LEARNING_RATE = 0.001
NN_EPOCHS = 100
NN_BATCH_SIZE = 32

# XGBoost
XGB_N_ESTIMATORS = [100, 200, 300]
XGB_MAX_DEPTH = [3, 4, 5]
XGB_LEARNING_RATE = [0.01, 0.1]
XGB_SUBSAMPLE = [0.8, 0.9, 1.0]
```

## Error Handling

### Common Exceptions
```python
class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass

class ModelError(Exception):
    """Raised when model training/prediction fails"""
    pass
```

## Usage Examples

### Complete Estimation Pipeline
```python
# 1. Preprocess data
df, features, scaler, encoders = preprocess_data(Config.DATASET_PATH)

# 2. Apply COCOMO
df = apply_cocomo_model(df)

# 3. Train ML models
eval_file = train_and_evaluate_models(df, features)

# 4. Load ML predictions
predictions_df = pd.read_excel(eval_file, sheet_name=1)
ml_predictions = {
    'Neural Network': predictions_df['Neural Network Predicted Effort'].values,
    'XGBoost': predictions_df['XGBoost Predicted Effort'].values
}

# 5. Calculate dynamic estimation
df = dynamic_cost_estimation(df, 
                           Config.TEAM_SIZE_COL,
                           Config.DURATION_COL,
                           Config.DAILY_HOURS_COL)

# 6. Apply hybrid estimation
results_df, output_file = hybrid_estimation(
    df,
    ml_predictions,
    df['COCOMO Effort (person-hours)'].values,
    df['Dynamic Effort'].values
)

# 7. Generate visualizations
create_comparison_plots(results_df, df.index)
```

## Best Practices

1. **Data Preparation**
   - Always validate input data
   - Handle missing values appropriately
   - Normalize/scale features

2. **Model Training**
   - Use cross-validation
   - Monitor for overfitting
   - Save model artifacts

3. **Error Handling**
   - Implement proper exception handling
   - Log errors and warnings
   - Validate outputs

4. **Performance**
   - Use batch processing for large datasets
   - Implement caching where appropriate
   - Monitor memory usage

## Troubleshooting

### Common Issues and Solutions

1. **Data Loading Errors**
   ```python
   try:
       df = pd.read_excel(file_path)
   except FileNotFoundError:
       logger.error(f"Dataset not found at {file_path}")
   ```

2. **Model Training Issues**
   ```python
   if df.empty or len(df) < MIN_SAMPLES:
       raise DataValidationError("Insufficient data for training")
   ```

3. **Memory Management**
   ```python
   # Clear memory after processing
   import gc
   gc.collect()
   ``` 