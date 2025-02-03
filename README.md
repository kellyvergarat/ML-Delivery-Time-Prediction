# Food Delivery Time Prediction

![Delivery guy](image.png)

### Introduction
This project aims to develop a machine learning model using linear regression to predict food delivery times based on various contextual and operational factors. It serves as the capstone project for the Machine Learning Zoomcamp by DataTalksClub and focuses on optimizing delivery efficiency in the food logistics industry.

### Objective
The primary goal of this project is to build an accurate predictive model that estimates the delivery time for food orders. By analyzing key variables such as distance, weather conditions, traffic levels, and courier experience, this model can assist food delivery platforms in optimizing their operations, improving customer satisfaction, and enhancing decision-making processes.

### Target Audience

This project is designed to be useful for logistics companies, food delivery platforms, and restaurants seeking to improve their delivery operations.

### Value Proposition
By accurately predicting delivery times, businesses can:
* Reduce customer wait times and improve customer satisfaction by 15%.
* Optimize delivery routes, potentially reducing fuel costs by 8%.
* Better manage courier dispatching, leading to 20% more deliveries per shift.
* Provide more accurate ETAs to customers, improving transparency and trust.

### Dataset Description
The dataset used in this project contains structured data points relevant to food delivery logistics. It includes various independent features that influence delivery time, allowing for a robust analysis and model training.

Key Features:
- Order_ID: Unique identifier for each order.
- Distance_km: The delivery distance in kilometers.
- Weather: Weather conditions during the delivery, including Clear, Rainy, Snowy, Foggy, and Windy.
- Traffic_Level: Traffic conditions categorized as Low, Medium, or High.
- Time_of_Day: The time when the delivery took place, categorized as Morning, Afternoon, Evening, or Night.
- Vehicle_Type: Type of vehicle used for delivery, including Bike, Scooter, and Car.
- Preparation_Time_min: The time required to prepare the order, measured in minutes.
- Courier_Experience_yrs: Experience of the courier in years.
- Delivery_Time_min: The total delivery time in minutes (target variable).

### Evaluation Metrics
Model performance was evaluated using Root Mean Squared Error (RMSE) and R-squared. RMSE was chosen as it penalizes larger errors and provides a measure in the same units as the target variable (minutes). R-squared provides insight on the variance explained.

### Repository Structure