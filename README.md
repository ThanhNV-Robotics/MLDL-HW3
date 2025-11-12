# MLDL-HW3: Car Price Prediction
## Author
Van Thanh Nguyen, ID: 20242084
## Project Overview
This project implements multiple machine learning regression models to predict used car prices based on various features such as year, kilometers driven, fuel type, transmission, and other specifications.

## Dataset
- **Training Data**: `train.csv` - Contains car features and prices for model training
- **Test Data**: `test.csv` - Contains car features for price prediction
- **Processed Data**: 
  - `train_processed.csv` - Preprocessed training data
  - `test_processed.csv` - Preprocessed test data

## Project Structure
```
MLDL-HW3/
├── README.md                          # Project documentation
├── train.csv                          # Original training dataset
├── test.csv                           # Original test dataset
├── train_processed.csv                # Preprocessed training data
├── test_processed.csv                 # Preprocessed test data
├── preprocess_train_data.ipynb        # Training data preprocessing notebook
├── preprocess_test_data.ipynb         # Test data preprocessing notebook
├── main_prediction_model.ipynb        # Main model training and comparison notebook
├── sample.csv                         # Sample submission format
└── submission_*.csv                   # Model prediction outputs
```

## Data Preprocessing

### Missing Value Handling
- **Training Data**: Removed all rows containing missing values (`\N`)
- **Test Data**: 
  - Numerical features: Imputed with column averages
  - Categorical features: Random assignment from existing values

### Feature Engineering
1. **Year → Age**: Converted car year to age (2025 - Year)
2. **String Parsing**: Extracted numerical values from:
   - `Mileage` (e.g., "19.67 kmpl" → 19.67)
   - `Engine` (e.g., "1248 CC" → 1248)
   - `Power` (e.g., "88.7 bhp" → 88.7)
3. **One-Hot Encoding**: Applied to categorical variables:
   - Fuel_Type
   - Transmission
   - Owner_Type
   - Colour

### Dropped Features
- `Name` (unique car model names, too many categories)
- `Location` (geographic information)
- `New_Price` (not available for most cars)
- `ID` (identifier, not predictive)

## Models Implemented

### 1. Decision Tree Regressor
- **Parameters**:
  - `max_depth=10`
  - `min_samples_split=20`
  - `min_samples_leaf=10`
- **Evaluation**: MAPE and 10-fold cross-validation

### 2. Random Forest Regressor
- **Parameters**:
  - `n_estimators=100`
  - `max_depth=20`
  - `min_samples_split=10`
  - `min_samples_leaf=5`
- **Evaluation**: MAPE and 10-fold cross-validation

### 3. Gradient Boosting Regressor
- **Parameters**:
  - `n_estimators=100`
  - `learning_rate=0.1`
  - `max_depth=20`
- **Evaluation**: MAPE

### 4. XGBoost Regressor (Recommended)
- **Parameters**:
  - `n_estimators=200`
  - `learning_rate=0.1`
  - `max_depth=10`
  - `eval_metric='mape'`
- **Features**: 
  - MAPE monitoring during training
  - Feature importance analysis
  - Early stopping capability

## Evaluation Metric

**MAPE (Mean Absolute Percentage Error)**:
```
MAPE = (1/n) * Σ|actual - predicted| / |actual| * 100%
```
- Lower MAPE indicates better model performance
- Interpretable as average percentage error

## Installation & Requirements

### Required Libraries
```bash
pip install numpy pandas scikit-learn xgboost matplotlib
```

### Package Versions (Recommended)
- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 2.0.0
- matplotlib >= 3.4.0

## Usage

### 1. Preprocess Data
Run the preprocessing notebooks in order:
```bash
# Open and run all cells in:
preprocess_train_data.ipynb
preprocess_test_data.ipynb
```

### 2. Train Models and Generate Predictions
```bash
# Open and run all cells in:
main_prediction_model.ipynb
```

### 3. Submission Files
The following CSV files will be generated:
- `submission_decision_tree.csv`
- `submission_random_tree_forest.csv`
- `submission_Gradient_Boosting.csv`
- `submission_xgboost.csv` (recommended)

## Results

Model comparison based on validation MAPE:
| Model | Validation MAPE |
|-------|-----------------|
| Decision Tree | TBD% |
| Random Forest | TBD% |
| Gradient Boosting | TBD% |
| XGBoost | TBD% |

*Note: Run `main_prediction_model.ipynb` to see actual results*

## Key Features

- ✅ Comprehensive data preprocessing pipeline
- ✅ Multiple model comparison framework
- ✅ MAPE-based evaluation for all models
- ✅ 10-fold cross-validation
- ✅ Feature importance analysis
- ✅ Reproducible results with fixed random seeds
- ✅ Ready-to-submit prediction files

## Future Improvements

- [ ] Hyperparameter tuning with Grid Search / Random Search
- [ ] Feature selection based on importance scores
- [ ] Ensemble methods combining multiple models
- [ ] Additional feature engineering (interaction terms)
- [ ] Outlier detection and handling

## License
This project is for educational purposes (Machine Learning and Deep Learning HW3).