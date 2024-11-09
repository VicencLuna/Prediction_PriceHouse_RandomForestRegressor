Here's a README.md file for the provided code:

# House Price Prediction using Random Forest Regressor

This project uses a Random Forest Regressor to predict house prices based on various features. It includes data preprocessing, feature engineering, model training, and evaluation.

## Features

- Data loading and preprocessing
- Feature engineering
- Model training using RandomForestRegressor
- Model evaluation and prediction
- Handling of missing values
- Feature importance analysis

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- statsmodels

## Usage

1. Ensure your input data is in the correct directory:
   ```
   C:|TEMP

   ```

2. Run the script:
   ```
   python house_price_prediction.py
   ```

3. The script will:
   - Load and preprocess the data
   - Engineer new features
   - Train a Random Forest model
   - Evaluate the model
   - Generate predictions for test data



## Input Files

- `train.csv`: Training dataset
- `test.csv`: Test dataset for predictions

## Output Files

- `temp.csv`: Preprocessed training data
- `matriz_correlacion`: Correlation matrix
- `df_new.csv`: Preprocessed test data
PREDICTION: - `predicciones_rf.csv`: Final predictions

## Code Structure

The script performs the following main steps:
1. Data loading and initial preprocessing
2. Feature engineering (including one-hot encoding)
3. Correlation analysis and feature selection
4. Model training and evaluation
5. Feature importance analysis
6. Prediction on test data

## Model Details

- Algorithm: Random Forest Regressor
- Number of estimators: 100
- Evaluation metrics: Mean Squared Error, R² Score, Root Mean Square Deviation

## Customization

To modify the model's behavior:
1. Adjust the `RandomForestRegressor` parameters in the model creation step.
2. Modify feature engineering steps in the preprocessing section.
3. Change the test size in the `train_test_split` function.

## Results

The script outputs: predicciones_xgb.csv
- Model performance metrics
- Feature importance ranking
- Predictions for test data


## License

[MIT](https://choosealicense.com/licenses/mit/)


