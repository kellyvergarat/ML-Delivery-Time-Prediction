pip install -r requirements.txt

import pickle
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import lightgbm as lgb

#load the dataset
df = pd.read_csv("../data/Food_Delivery_Times.csv")

df.columns = [col.lower() for col in df.columns]
df = df.drop('order_id', axis=1)


categorical_features = ['weather','traffic_level','time_of_day','vehicle_type']
numerical_features = ['distance_km','preparation_time_min','courier_experience_yrs','delivery_time_min']

# Fill missing missing values values with mode (most frequent value)
df['weather'] = df['weather'].fillna(df['weather'].mode()[0])
df['traffic_level'] = df['traffic_level'].fillna(df['traffic_level'].mode()[0])
df['time_of_day'] = df['time_of_day'].fillna(df['time_of_day'].mode()[0])

# Fill missing missing values values with median (middle value)
df['courier_experience_yrs'] = df['courier_experience_yrs'].fillna(df['courier_experience_yrs'].median())

# ### Encode categorical features
df_encoded = pd.get_dummies(
    df,
    columns=categorical_features,
    drop_first=True
)

# Split the original dataframe 'df' into two parts: 
# 'df_full_train' (80% of the data) and 'df_test' (20% of the data)
df_full_train, df_test = train_test_split(df_encoded, test_size=0.2, random_state=1)

# Further split 'df_full_train' into 'df_train' (75% of df_full_train) 
# and 'df_val' (25% of df_full_train)
# This means 'df_train' is 60% of the original data and 'df_val' is 20% of the original data
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.delivery_time_min.values
y_val = df_val.delivery_time_min.values
y_test = df_test.delivery_time_min.values

del df_train['delivery_time_min']   
del df_val['delivery_time_min']
del df_test['delivery_time_min']

### Scale Numeric Columns
numerical_features.remove('delivery_time_min')
scaler = StandardScaler()
df_train[numerical_features] = scaler.fit_transform(df_train[numerical_features])
df_val[numerical_features] = scaler.transform(df_val[numerical_features])
df_test[numerical_features] = scaler.transform(df_test[numerical_features])

# ### Train models
def evaluate_model_performance(regressor, model_name):
    regressor.fit(df_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = regressor.predict(df_val)

    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    r2score = r2_score(y_val, y_val_pred)
    
    print(f"Mean Squared Error on validation set for {model_name} is: ", round(rmse, 2))
    print(f"R2 Score on validation set for {model_name} is: ", round(r2score, 2))
    
    return regressor

# Ridge hyperparameter tuning

# %%
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}  

grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error') 

# Fit the GridSearchCV object (it will train and evaluate multiple models)
grid_search.fit(df_train, y_train)

# Get the best model (the one with the best hyperparameter)
best_ridge_model = grid_search.best_estimator_
print("Best Ridge Model: ", grid_search.best_params_)
evaluate_model_performance(best_ridge_model, "Ridge Regression with Hyperparameter Tuning")

# ### Use the model on the test set
#Concatenate the training and validation dataframes
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = df_full_train
X_full_train.shape

# %%
y_full_train = np.concatenate([y_train, y_val]) 
y_full_train.shape

# %%
df_test.shape

# %%
y_test.shape

# %%
alpha=1
ridge = Ridge(alpha=alpha, random_state=1)
ridge.fit(X_full_train, y_full_train)
y_test_pred = ridge.predict(df_test)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2score = r2_score(y_test, y_test_pred)

print(f"Mean Squared Error on test set for Ridge Regression is: {round(rmse,2)}")
print(f"R2 Score on test set for Ridge Regression is: {round(r2score,2)}")

# %%
# Select a single row from the test set
single_row = df_test.iloc[0]

# Get the actual premium price for the selected row
actual_delivery_time = y_test[0]

# Reshape the single row to match the expected input shape for the model
single_row_reshaped = single_row.values.reshape(1, -1)

# Predict the premium price using the trained Ridge model
predicted_delivery_time = ridge.predict(single_row_reshaped)[0]

print(f"Actual Premium Price: {actual_delivery_time}")
print(f"Predicted Premium Price: {predicted_delivery_time}")

# %% [markdown]
# ### Save the model
# 

# %%
# Set the output file path
output_file = '../model/ridge_model.bin'

# Open the file in write-binary mode
with open(output_file, 'wb') as f_out:
    # Serialize and save the Ridge model and test data to the file
    pickle.dump((ridge, df_test), f_out)

print(f'Model saved to {output_file}')

# %%
#load the model
input_file = '../model/ridge_model.bin'
with open(input_file, 'rb') as f_in: 
    model, test_data = pickle.load(f_in)

# %%
model


