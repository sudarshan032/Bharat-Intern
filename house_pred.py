import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Fetch the Boston housing dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Create a DataFrame with the loaded data
columns = [f'feature_{i}' for i in range(data.shape[1])]
house_price_dataframe = pd.DataFrame(data, columns=columns)
house_price_dataframe['price'] = target

# Calculate correlation
correlation = house_price_dataframe.corr()

# Constructing a heatmap to understand the correlation
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

# Split the data into training and test sets
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the XGBoost model
model = XGBRegressor()

# Train the model with X_train
model.fit(X_train_scaled, Y_train)

# Evaluate on training data
training_data_prediction = model.predict(X_train_scaled)
score_1 = metrics.r2_score(Y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error (Training): ", score_1)
print('Mean Absolute Error (Training): ', score_2)

# Scatter plot for actual vs predicted prices on training data
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price (Training)")
plt.show()

# Evaluate on test data
test_data_prediction = model.predict(X_test_scaled)
score_1 = metrics.r2_score(Y_test, test_data_prediction)
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error (Test): ", score_1)
print('Mean Absolute Error (Test): ', score_2)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 200]
}

xgb_reg = XGBRegressor()
grid_search = GridSearchCV(xgb_reg, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, Y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the model with cross-validation
cv_scores = cross_val_score(model, X_train_scaled, Y_train, cv=5, scoring='neg_mean_absolute_error')
print("Cross-Validation Scores:", -cv_scores)
