# Software Effort Estimation Using Hybrid Approach

## Project Overview
This project implements a comprehensive software effort estimation system that combines traditional algorithmic methods (COCOMO), machine learning approaches (Neural Networks and XGBoost), and dynamic estimation techniques into a sophisticated hybrid model.

## Features
- **Multiple Estimation Methods:**
  - COCOMO (Constructive Cost Model)
  - Neural Network-based estimation
  - XGBoost-based estimation
  - Dynamic effort estimation
  - Hybrid approach combining all methods

- **Advanced Machine Learning Implementation:**
  - Deep Neural Network with multiple layers
  - XGBoost with hyperparameter tuning
  - Feature engineering and preprocessing
  - Model evaluation and validation

- **Hybrid Estimation System:**
  - Adaptive weighting mechanism
  - Complexity-based weight adjustment
  - Performance-based method selection
  - Pattern analysis for different project types

- **Comprehensive Visualization:**
  - Individual comparison plots for each method
  - Scatter plots with perfect prediction lines
  - Performance metrics visualization
  - Trend analysis graphs

## Project Structure
```
├── main.py                 # Main execution file
├── preprocessing.py        # Data preprocessing module
├── cocomo_model.py        # COCOMO implementation
├── machine_learning.py    # ML models implementation
├── dynamic_estimation.py  # Dynamic estimation module
├── hybrid_estimation.py   # Hybrid approach implementation
├── configuration.py       # Configuration settings
├── CommonEnums.py        # Common enumerations
└── figures.py            # Visualization module
```

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python main.py
```

This will:
1. Preprocess the input data
2. Apply all estimation methods
3. Generate comparative analysis
4. Create visualization plots
5. Save results in Excel format

## Output Files
- `comparison_*.png`: Line plots comparing each method with actual effort
- `scatter_*.png`: Scatter plots showing estimation accuracy
- `hybrid_estimation_*.xlsx`: Detailed results and metrics
- `dynamic_cost_estimation_*.xlsx`: Dynamic estimation results
- `model_evaluation_*.xlsx`: ML models performance metrics

## Performance Metrics
The system evaluates estimations using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²)
- Correlation Coefficient

## Key Findings
- XGBoost shows superior performance for complex projects
- Hybrid approach provides balanced estimates
- Dynamic estimation offers practical validation
- COCOMO serves as a reliable baseline

## Future Improvements
- Integration with real-time project data
- Additional machine learning models
- Enhanced visualization options
- API development for external integration

## Dependencies
- Python 3.8+
- TensorFlow
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Author
[Your Name]

## License
[Your License Choice] 