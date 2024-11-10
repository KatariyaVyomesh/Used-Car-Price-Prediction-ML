# Used-Car-Price-Prediction-ML
A machine learning project to predict used car prices based on features like year, mileage, fuel type, and engine size. Various models, including Random Forest and Linear Regression, were implemented and evaluated. Techniques like cross-validation and hyperparameter tuning were used to optimize performance and reduce overfittin

------------------------------------------------------------------------------------------------
# Used Car Price Prediction

This project aims to predict the selling price of used cars based on various features such as year, kilometers driven, fuel type, engine size, transmission type, and more. The model is built using machine learning algorithms like Random Forest Regression, Decision Trees, and Linear Regression. Comprehensive data preprocessing, including handling missing values, encoding categorical variables, and feature scaling, is employed to enhance model performance.

------------------------------------------------------------------------------------------------
## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Machine Learning Models](#machine-learning-models)
5. [Model Evaluation](#model-evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Performance Metrics](#performance-metrics)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

------------------------------------------------------------------------------------------------
## Project Overview

The goal of this project is to develop a machine learning model capable of predicting the selling price of used cars. The prediction model is trained on a dataset containing key features that influence car prices, such as the car's **year**, **engine size**, **mileage**, and **transmission type**. Various machine learning algorithms are employed and evaluated to identify the most accurate model for predicting car prices.

Key steps involved in the project:
- **Data Collection and Preprocessing**: Clean the dataset, handle missing values, encode categorical variables, and scale numerical features.
- **Modeling**: Implement and compare multiple machine learning models, such as Random Forest, Decision Trees, Support Vector Machines, and Linear Regression.
- **Hyperparameter Tuning**: Optimize model performance using GridSearchCV for hyperparameter tuning.
- **Evaluation**: Assess the models using cross-validation and R² score to ensure robustness and generalization.

## Dataset Description

The dataset contains the following columns:

- **name**: The brand and model name of the car.
- **year**: The manufacturing year of the car.
- **km_driven**: The number of kilometers the car has been driven.
- **fuel**: The type of fuel the car uses (e.g., Petrol, Diesel, CNG).
- **seller_type**: The type of seller (e.g., Individual, Dealer).
- **transmission**: The type of transmission (e.g., Manual, Automatic).
- **owner**: The number of previous owners.
- **mileage(km/ltr/kg)**: The fuel mileage of the car (e.g., km/ltr or km/kg).
- **engine**: The engine size (in liters).
- **max_power**: The maximum power of the car (in horsepower).
- **seats**: The number of seats in the car.
- **selling_price**: The target variable – the price at which the car is being sold.

## Data Preprocessing

The following preprocessing steps were performed on the dataset:

1. **Missing Value Handling**: Imputed missing numerical values with the mean and handled missing categorical values using encoding.
2. **Feature Encoding**: Categorical variables (e.g., fuel, transmission, seller_type) were encoded using Label Encoding and Ordinal Encoding.
3. **Feature Scaling**: Applied StandardScaler to scale numerical features for consistent range and to improve model performance.
4. **Outlier Removal**: Identified and removed any outliers from the numerical columns.

## Machine Learning Models

The following machine learning models were implemented to predict car prices:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Support Vector Regressor (SVR)**
4. **Random Forest Regressor**
5. **Bagging Regressor**
6. **K-Nearest Neighbors (KNN)**
7. **Lasso Regression**

The models were evaluated based on their R² score and performance on the testing dataset.

## Model Evaluation

The models were evaluated using cross-validation with 10 folds. Hyperparameter tuning was done using **GridSearchCV** for the Random Forest Regressor to optimize key parameters like the number of trees (`n_estimators`), maximum tree depth (`max_depth`), and the minimum number of samples required to split or be in a leaf node.

### Performance Metrics

- **Training Accuracy**: 98.66%
- **Testing Accuracy**: 92.82%

The Random Forest Regressor with tuned hyperparameters showed the best performance, achieving high accuracy on both the training and testing datasets.

## Installation

### Prerequisites

To run this project, you'll need the following:

- Python 3.x
- Jupyter Notebook (optional, for running the notebook)
- The following Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
------------------------------------------------------------------------------------------------

# Performance Metrics
**Training Accuracy**: 98.66%
**Testing Accuracy**: 92.82%
The Random Forest model performed exceptionally well, achieving a high training accuracy and good generalization on the test data.

------------------------------------------------------------------------------------------------
# License
This project is licensed under the MIT License. See the LICENSE file for more information.

------------------------------------------------------------------------------------------------
# Acknowledgements
-> The dataset used in this project is sourced from CarDekho.
-> Special thanks to the Scikit-learn library for providing the machine learning tools and Pandas for data manipulation and cleaning.
->The project was built as part of a machine learning course and is intended for educational purposes.
