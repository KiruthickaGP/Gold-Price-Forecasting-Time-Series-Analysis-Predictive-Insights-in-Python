**Gold Price Time Series Analysis and Forecasting**

**Dataset**
The dataset used in this project is named "monthly_csv.csv" and includes monthly gold prices. The data is loaded into a pandas DataFrame for further analysis.


**Data Exploration and Visualization:**

Loading the dataset and exploring its structure.
Creating visualizations to understand the trends and patterns in gold prices over time.


**Statistical Analysis:**

Calculating descriptive statistics for the dataset.
Generating box plots and seasonal plots to analyze monthly and yearly trends.

**Time Series Decomposition:**

Resampling the data to analyze average gold prices on a yearly, quarterly, and decadal basis.
Calculating mean, standard deviation, and coefficient of variation over the years.

**Linear Regression Model:**

Splitting the data into training and testing sets.
Building a linear regression model based on time to predict future gold prices.
Evaluating the model's performance using Mean Absolute Percentage Error (MAPE).

**Naive Forecast Model:**

Creating a naive forecast model by using the last observed value as a prediction for future prices.
Evaluating the performance of the naive model.

**Exponential Smoothing Model:**

Implementing the Holt-Winters Exponential Smoothing model for time series forecasting.
Generating predictions and calculating MAPE for the model.

**Visualization of Results:**

Plotting actual vs. predicted values with confidence intervals for the Exponential Smoothing model.
