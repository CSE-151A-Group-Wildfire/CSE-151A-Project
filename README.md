# Milestone 4 

<details open>
<summary><h2>Introduction</h2></summary>
&nbsp;

The focus of the project was to analyze and predict weather conditions in Los Angeles County using a dataset maintained and saved by the California Weather Government Website. The dataset includes over twenty years of weather data, which is essential for predicting weather conditions on an hourly basis. By understanding trends and patterns, we can use weather variables such as temperature, humidity, and wind speed to gain insights into climate change and its local impact.
We chose this project because global warming and climate change are issues that affect all of us. Selecting a project that allows us to observe these changes in our backyard can help bring the message home. Accurate weather predictions can improve resource management, provide timely warnings for extreme weather events, and support informed decision-making processes. The broader impact of this work includes practical applications in urban planning, public safety, and environmental conservation, making it a valuable contribution to both scientific research and community welfare.



</details>

<details open>
<summary><h2>Methods</h2></summary>

## Data Exploration 
&nbsp;

We initially started our data exploration by analyzing the correlation between weather changes over the past few decades and their direct impact on wildfire patterns. We were able to get data of where and when wildfires occurred in Los Angeles County through the California Wildfires Government Website. We started to realized when we started merging the two datasets from California Weather and California Wildfires that we only had 991 rows to train our model with which isn’t enough rows. As a result, we quickly decided to remove the the wildfire dataset and only use the weather dataset from the California Weather Government Website. With our weather dataset alone, we were able to use 205642 rows for our models.  As a result, we decided to analyze and predict weather conditions in Los Angeles County instead.






&nbsp;

## Pre-Processing

The preprocessing phase focused on cleaning and preparing our data from the weather stations in Los Angeles for modeling. To ensure that our model produced the best output, we took the following steps:

## Dropping Unnecessary Columns:
Many columns were determined to be irrelevant for predicting the weather in LA County. These columns included 'Unnamed: 32', 'Unnamed: 41', 'metar', 'metar_origin', 'pressure_change_code', 'weather_cond_code', 'pressure_change_code', 'visibility', 'cloud_layer_1', 'cloud_layer_2', 'cloud_layer_3', 'wind_cardinal_direction', 'cloud_layer_1_code', 'cloud_layer_2_code', 'cloud_layer_3_code', and 'heat_index'.

## Handling Missing Values:
Missing values were common in many columns, but for our target column, 'air_temp', there were only a few missing values. We handled this by dropping those rows, and for numeric columns, we replaced missing values with the mean of their respective columns to maintain consistency.

## Data Type Conversion:
The 'air_temp' column was converted to a numeric type to ensure accurate calculations during modeling.

## Date and Time Components Extraction:
The 'date_time' column was converted to a datetime format, and additional columns for year, month, day, and hour were extracted. This allowed for a more detailed interpretation of the weather data and enabled us to model weather conditions based on hourly data.

## Calculating Hourly and Yearly Averages:
Hourly average temperatures were calculated by grouping data by year, month, day, and hour. Similar calculations were performed for wind speed and sea level pressure to obtain hourly averages. Yearly average temperatures and precipitation were also calculated to observe long-term trends.

## Normalization:
Features such as temperature, wind speed, and sea level pressure were normalized using MinMaxScaler to ensure that our features remained consistent. This step was crucial for improving the performance of our models.

## Scatter Plot:

![scatterplot](https://github.com/user-attachments/assets/8dcf46ea-5671-473e-a66e-62f071899ac0)

The scatter plot that we made involved the the yearly average temperature over a 24 years spanning 2000 to 2024. With each blue dot representing the average temperature for a specific year. The X-axis shows the years, and Y-axis represents the yearly average temperatures. The plot reveals variability in average temperatures across the years, indicating fluctuations and possible trends in temperature changes. This visualization helps identify patterns, such as increasing or decreasing temperatures, which may indicate climate change. It also highlights any outliers or unusual values that warrant further investigation, providing a foundational understanding for further analysis and modeling.


## Pair Plot:

![pairplot](https://github.com/user-attachments/assets/143dd7cc-a989-4852-8956-bfea441d2993)

The pair plot shows the relationships between various weather variables in our dataset. It helps us see patterns, correlations, and potential outliers that are important for building predictive models.

The plots for year, month, day, and hour show how data points are distributed over time. The month, day, and hour data are collected at regular intervals, while the year data is spread continuously from 2000 to 2025. 

Sea level pressure varies widely but often centers around 1010-1020 hPa. There are noticeable changes in sea level pressure across different months and hours, hinting at seasonal and daily effects. Relative humidity ranges from 0% to 100%, showing a negative correlation with air temperature and hourly average temperature. This means higher temperatures generally come with lower humidity.

Hourly average temperature strongly correlates with air temperature, as expected. The plots show that temperatures vary significantly throughout the day and year. Wind speed has a wide range but clusters at lower speeds. It changes across different times of the day and year but doesn't strongly correlate with other variables.

These insights are useful for predictive modeling. Air temperature, sea level pressure, and relative humidity have clear patterns and correlations that can improve model accuracy. Daily and seasonal temperature and humidity variations should be included in models. The negative correlation between temperature and humidity and the variability in sea level pressure are important factors for temperature prediction models.

In conclusion, the pair plot helps us understand the relationships between weather variables, highlighting the importance of temporal patterns and correlations for predictive modeling. These insights will help us build more accurate and reliable weather prediction models. This analysis underscores the value of exploratory data analysis in guiding the modeling process.


## Correlation Matrix: 

![correlation](https://github.com/user-attachments/assets/4693697d-9a74-4bfb-b8e6-7488d896ba50)

The correlation matrix heatmap shows the relationships between different weather variables in our dataset. This is important for selecting features for predictive modeling and understanding how these variables interact.

Air temperature has a strong positive correlation with itself and a moderate positive correlation with hourly average temperature and wind speed. It also has a negative correlation with relative humidity and sea level pressure. Wind speed is moderately positively correlated with air temperature and hourly average temperature, and negatively correlated with sea level pressure. Sea level pressure is negatively correlated with air temperature, wind speed, relative humidity, and hourly average temperature.

Precipitation has weak correlations with most variables but shows a slight positive correlation with wind speed and relative humidity. The hour of the day correlates positively with air temperature and hourly average temperature, showing that temperatures vary significantly within a day. Year and month have very weak correlations with most variables, indicating that short-term factors influence weather more than long-term trends. Relative humidity has a strong negative correlation with air temperature and hourly average temperature, indicating that higher temperatures are associated with lower humidity levels.

Hourly average temperature is strongly positively correlated with air temperature and moderately positively correlated with wind speed and hour. It has a negative correlation with sea level pressure and relative humidity.

These correlations are useful for predictive modeling. Variables like air temperature, wind speed, and hourly average temperature are important for predictions. The negative correlation between air temperature and relative humidity suggests that humidity should be considered when predicting temperature. Weak correlations of year and month with other variables suggest that short-term features like hour are more important for predictions.

In conclusion, the correlation matrix heatmap helps us understand the relationships between weather variables. Strong correlations between temperature-related variables highlight their connections, while weak correlations with long-term factors suggest that immediate conditions are more important for weather predictions. This understanding will help us select the right features and improve the accuracy of our predictive models.




&nbsp;
## Model 1: Neural Network 








&nbsp;

## Model 2: K Nearest Neighbor 


The second model we decided to use is the KNN, K Nearest Neighbor. This model was selected for predicting weather-related variables due to its simplicity and effectiveness in capturing patterns in data without assuming an underlying distribution. The KNN algorithm works by finding the closest training examples in the feature space and predicting the target value based on the average of these neighbors. This approach makes it particularly well-suited for problems where the relationship between features and the target variable is non-linear.

&nbsp;

We split the data into training and test sets with an 80-20 split. This step was essential for evaluating the model's performance on unseen data. We then trained the KNN model on the training data using five neighbors.
&nbsp;
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

```
&nbsp;
We made predictions on the test set and evaluated the model using Mean Squared Error (MSE), a common metric for regression models.
&nbsp;

```
y_pred = knn_model.predict(X_test)

mse_temp = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0])
mse_wind = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1])
mse_sea = mean_squared_error(y_test.iloc[:, 2], y_pred[:, 2])

print(f"Mean Squared Error for Temperature: {mse_temp}")
print(f"Mean Squared Error for Wind Speed: {mse_wind}")
print(f"Mean Squared Error for Sea Pressure: {mse_sea}")
```
&nbsp;
We visualized the actual versus predicted values to gain further insights into the model's performance. Here is an example for Temperature. 
&nbsp;
```

#Temperature
plt.subplot(1, 3, 1)
plt.plot(X_test_sorted, y_test.iloc[sorted_indices, 0], label='Actual Temperature', color='blue')
plt.plot(X_test_sorted, y_pred[sorted_indices, 0], label='Predicted Temperature', color='red', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Temperature')
plt.title('Actual vs Predicted Temperature')
plt.legend()

plt.tight_layout()
plt.show()
```




</details>

<details open>
<summary><h2>Results</h2></summary>

## Model 1 




## Model 2: KNN Final Model 
&nbsp;
Detailed Analysis
The Mean Squared Error (MSE) values for the predictions were:
&nbsp;

- Temperature: 0.005237
  &nbsp;
- Wind Speed: 0.018628
 &nbsp;
- Sea Pressure: 0.006608

![knn](https://github.com/user-attachments/assets/cafba7d4-ad9d-46ed-b5f1-87e45e6aa27a)


&nbsp;
Temperature Prediction:
&nbsp;

The low MSE of 0.005237 for temperature indicates that the KNN model's predictions were highly accurate. The visualization showed a close alignment between the actual and predicted temperature values, suggesting that the model effectively captured the temporal patterns and variations in temperature. This performance demonstrates the model's ability to handle the non-linear relationships present in the temperature data.
&nbsp;

Wind Speed Prediction:
&nbsp;

The MSE of 0.018628 for wind speed was higher than that for temperature and sea pressure. The visualization revealed that while the model could capture the general trend of wind speed, there were some noticeable deviations. These deviations indicate that the model struggled to predict wind speed as accurately as it did for temperature and sea pressure. Wind speed can be more variable and influenced by a wider range of factors, which may not have been fully captured by the features used in the model.
&nbsp;

Sea Pressure Prediction:
&nbsp;

The MSE of 0.006608 for sea pressure was also low, indicating good model performance. The close alignment between the actual and predicted values in the visualization suggests that the model was effective in capturing the trends in sea pressure data. The performance for sea pressure was similar to that for temperature, indicating that the model was consistent in predicting variables that may have more stable patterns compared to wind speed.
&nbsp;

Overall Performance:

&nbsp;
The KNN model showed varying performance across different weather-related variables. It performed best for **temperature prediction**, followed by sea pressure, and then wind speed. The low MSE values for temperature and sea pressure indicate that the model was able to accurately capture the patterns in these variables. However, the higher MSE for wind speed suggests that there is room for improvement in capturing the more complex patterns associated with wind speed.








</details>

<details open>
<summary><h2>Discussion</h2></summary>
  &nbsp;
# Discussion
  &nbsp;
## Dataset
&nbsp;

### Non-Normal Distributions
Several variables, such as `sea_level_pressure`, `relative_humidity`, `hourly_avg_temp`, and `wind_speed`, show non-normal distributions with potential outliers. Skewed distributions may affect the performance of models that assume normally distributed input features.

### Outliers
The scatter plots suggest the presence of outliers in variables such as `wind_speed` and `hourly_avg_temp`. Outliers can distort the model training process, leading to less accurate predictions.

### Variability in Data
There is high variability in `relative_humidity` and `sea_level_pressure`, which might require further normalization or transformation. We normalized these features in the later part.

### Data Preprocessing
As one of the X values used for prediction, the column `wind_speed` contains too many 0 or 0.0 values. We cannot determine whether this is due to missing data or if the data itself is actually 0. Therefore, when handling missing values, we choose to fill them with the median rather than the mean.

## Linear Regression Model

### Mean Squared Error (MSE)
The MSE of 11.77 suggests that the model has a relatively high error in its predictions. This indicates that the predictions are not close to the actual values, leading to less accurate forecasts.

### R² Score
An R² score of 0.48 implies that only 48% of the variance in the target variable (`hourly_avg_temp`) is explained by the features in the model. This is relatively low, indicating that the model is not capturing the underlying patterns effectively.

### Temporal Dependencies
Techniques like time series analysis or adding time-related features (e.g., seasonality, trends) could improve the model's accuracy.

### Data Sampling
Ensuring that the data is representative and balanced across different time periods and conditions is crucial. Stratified sampling or using cross-validation techniques like time-series split can help in better evaluating the model.

## Model 2

### Performance Issues
The plots show significant deviations between the actual and predicted values for temperature, wind speed, and sea pressure. The model's predictions do not seem to capture the variations in the actual data very well, particularly for wind speed and sea pressure, as the red dashed lines (predicted values) do not align closely with the blue lines (actual values).

### Improvements

#### Model Complexity
The neural network architecture used might not be complex enough to capture the underlying patterns in the data. Increasing the number of layers, neurons, or trying different activation functions might improve the model's performance.

#### Data Augmentation and Cleaning
Explore additional data cleaning and augmentation techniques to improve the quality of the input data.

#### Feature Engineering
Introduce new features or transform existing ones to better capture the relationships in the data.

#### Cross-Validation
Implement cross-validation to better estimate the model's performance and reduce overfitting.







</details>

<details open>
<summary><h2>Conclusion</h2></summary>
&nbsp;

In this project, we explored and analyzed a weather dataset to develop predictive models for temperature and precipitation. By examining the data, we identified patterns such as daily and seasonal temperature changes, and yearly trends in temperature and rainfall. These insights helped guide our modeling choices.
&nbsp;

Our preprocessing steps included removing unnecessary columns, filling in missing values, and extracting date and time components. This ensured the data was clean and ready for modeling.
&nbsp;

We tested three models: Linear Regression, Neural Network, and K-Nearest Neighbors (KNN). The Linear Regression model performed best with an MSE of 11.77 and an R² Score of 0.48. The Neural Network struggled with an MSE of 0.17. The KNN model had an MSE of 0.00524 for temperature, 0.0186 for wind, and 0.00661 for sea pressure, performing slightly better than the Neural Network but not as well as Linear Regression.

&nbsp;
These results suggest that the simpler Linear Regression model captured the data patterns effectively, while the more complex models may need further tuning and feature engineering.
&nbsp;

Looking back, there are several things we wish we could have done differently. Firstly, exploring additional features or transformations might have uncovered hidden patterns and improved the performance of the more complex models. For example, incorporating interaction terms or lagged variables could have helped capture temporal dependencies in the weather data. Additionally, trying out ensemble methods, such as Random Forests or Gradient Boosting, might have provided better predictions by combining the strengths of multiple models.
&nbsp;

Another area for improvement is data quality. Ensuring more accurate and consistent data collection, and addressing inconsistencies more thoroughly during preprocessing, could lead to more reliable predictions. Finally, conducting a more extensive hyperparameter tuning for the Neural Network and KNN models might have yielded better results.
&nbsp;

In future projects, we would focus on these aspects to enhance model performance. This project highlights the critical role of data exploration and preprocessing in building effective predictive models. Continuous improvements in these areas are key to achieving more accurate and reliable weather predictions.

&nbsp;

</details>

<details open>
<summary><h2>Statement of Collaboration</h2></summary>

</details>










# Milestone 3
The code starts by preparing the data for analysis. It uses a specific DataFrame, hourly_avg_temp, which includes the average hourly temperature. The 'hour' column is separated from the features for later use. The features are then scaled to a range between 0 and 1 using the MinMaxScaler from sklearn, resulting in a normalized dataset which can help compare the two different information with each other and how it relates to time. The processed features and the 'hour' column are then concatenated to form the final DataFrame, final_df.

Next, the code sets up the neural network for training. It imports necessary libraries such as pandas, numpy, and various modules from keras for building the neural network. The prepared data (processed_df) is used as the input features (X), while the original features are used as the target labels (y). The dataset is split into training and test sets, with 10% of the data reserved for testing. A sequential neural network model is built with one input layer, two hidden layers, each containing 12 neurons and using the sigmoid activation function. The output layer also uses the sigmoid activation function. The model is compiled using the SGD optimizer with a learning rate of 0.3 and the categorical crossentropy loss function, and then trained for 100 epochs with a batch size of 32 and a validation split of 10%.

After training, the model's performance is evaluated on the test set. Predictions are made using the trained model, and the true and predicted class labels are determined. A confusion matrix is computed to assess the performance of the model. This matrix is visualized using ConfusionMatrixDisplay from sklearn, displaying the results in a heatmap format. The confusion matrix helps in understanding the accuracy and error rates of the model's predictions.
In summary, the code involves a comprehensive process of data preparation, neural network training, and performance evaluation. The data is first merged and normalized, and then a neural network is built and trained to predict weather-related features. The model's accuracy is assessed using a confusion matrix, providing insights into the model's predictive capabilities and highlighting areas for potential improvement. This process demonstrates the application of machine learning techniques to analyze and predict complex weather patterns.

With our results, we noticed that our model has around an accuracy of 5.8912^-6 which means our model is underfitting. This means that we have to improve our model to have a higher accuracy to have our model to be appropriate-fitting. Another improvement we can make is to do hyperparameter tuning to determine which parameters are the best suited for our model.

## Milestone 2
We preprocessed our data to answer the question: since 2000, has the Earth gotten hotter? We'll do this by comparing every hourly collectively since 2000, to see if they steadily increase. We started by collecting a dataset called 'weather' with 43 features and over 261,000 rows. Given the abundance of data, we had to process it to our liking. First, we made a list of features we wanted to drop since they do not correlate with answering our question. These features were: ['Unnamed: 32', 'Unnamed: 41', 'metar', 'metar_origin', 'pressure_change_code', 'weather_cond_code', 'pressure_change_code', 'visibility', 'cloud_layer_1', 'cloud_layer_2', 'cloud_layer_3', 'wind_cardinal_direction', 'precip_accum_one_hour', 'cloud_layer_1_code', 'cloud_layer_2_code', 'cloud_layer_3_code', 'heat_index']. After removing the unnecessary features, we addressed the issue of missing data by dropping NA values from our main column, 'air_temp', and then filling in every numeric column with NA values using the corresponding column mean. We converted 'air_temp' to a numeric column to ensure it was in the correct format.

Next, we converted and changed the format of the 'date_time' column in 'weather.csv' to 'pd.to_datetime', allowing us to utilize the date more precisely. Using the 'date_time' column, we created four additional columns: 'year', 'month', 'day', and 'hour'. With these new columns, we calculated each hourly average temperature by grouping by 'year' and 'month', ‘day’, and ‘hour’ and calculating the mean.

In our first graph. The code uses Python's `matplotlib` and `seaborn` to create a line graph showing how three-hour accumulated precipitation changes over the years. First, it sets up the plot and sorts the data by year. It then plots the years on the x-axis and the precipitation values on the y-axis using a green line. The graph shows a sharp drop in precipitation around 2000, followed by ups and downs, and a slight increase in recent years. This helps visualize the trend of three-hour precipitation over time.

In our second graph. We also used Python's `matplotlib` and `seaborn` to create a line graph showing how air temperature changes over the years. First, it sets up the plot and sorts the data by year. It then plots the years on the x-axis and the air temperature values on the y-axis using a red line. The graph shows fluctuations in air temperature from 1998 to 2025, with notable peaks and troughs. This helps visualize the trend of air temperature over time.

For our third graph. The code uses `matplotlib` to create a line graph showing monthly average temperatures over time. It sets up a large figure, then plots the year and month on the x-axis and the average temperature on the y-axis using blue circles and lines. The graph shows regular seasonal cycles with temperatures rising and falling each year. There are clear peaks and troughs, indicating the typical seasonal changes. The labels and title help explain the graph, making it easy to see how average temperatures have varied over the years.

For our fourth graph we did a scatter plot. This code uses `matplotlib` to create a scatter plot showing yearly average temperatures over time. It sets up a figure, then plots the years on the x-axis and the average temperatures on the y-axis using blue dots. The plot is labeled to indicate that the data points represent actual temperature data. The x-axis is labeled 'Year' and the y-axis is labeled 'Yearly Average Temperature.' The graph shows individual data points for each year's average temperature, revealing trends and variations over time. The grid and legend make it easier to read and understand the plot. This visual representation helps in identifying patterns in yearly average temperatures from 2000 to 2025.

For our fifth graph we did a pairplot. The code merges weather data with monthly average temperatures, converts the 'date_time' column to extract year, month, day, and hour, and converts specified columns to numeric types. It then drops rows with missing values in these columns. Finally, it creates a pair plot using `seaborn` to show scatter plots and histograms of the relationships between different weather variables. The pair plot helps visualize how these variables, such as temperature, humidity, and wind speed, relate to each other.

For our last graph we did a correlation matrix. The code calculates the correlation matrix for various weather variables and visualizes it using a heatmap. The correlation matrix shows how strongly each pair of variables, such as temperature, humidity, and wind speed, are related. The heatmap uses colors to represent the correlation values, with annotations displaying the exact values. This helps quickly identify which variables have strong positive or negative correlations.
