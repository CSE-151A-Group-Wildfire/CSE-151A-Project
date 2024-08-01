# Milestone 4 

<details open>
<summary><h2>Introduction</h2></summary>
&nbsp;

The focus of the project was to analyze and predict weather conditions in Los Angeles County using a dataset maintained and saved by the California Weather Government Website. The dataset includes over twenty years of weather data, which is essential for predicting weather conditions on an hourly basis. Specifically, we aimed to predict three key weather variables: temperature, wind speed, sea pressure, and humidity, using hours, days, and years as features. By understanding trends and patterns in these variables, we can gain insights into climate change and its local impact.

We chose this project because global warming and climate change are issues that affect all of us. Selecting a project that allows us to observe these changes in our backyard can help bring the message home. Accurate weather predictions can improve resource management, provide timely warnings for extreme weather events, and support informed decision-making processes. The broader impact of this work includes practical applications in urban planning, public safety, and environmental conservation, making it a valuable contribution to both scientific research and community welfare.

</details>


<details open>
<summary><h2>Methods</h2></summary>

## Data Exploration 
&nbsp;
We initially started our data exploration by analyzing the correlation between weather changes over the past few decades and their direct impact on wildfire patterns. We were able to get data of where and when wildfires occurred in Los Angeles County through the California Wildfires Government Website. We started to realize when we started merging the two datasets from California Weather and California Wildfires that we only had 991 rows to train our model with, which isn’t enough rows. As a result, we quickly decided to remove the wildfire dataset and only use the weather dataset from the California Weather Government Website. With our weather dataset alone, we were able to use 205,642 rows for our models. As a result, we decided to analyze and predict weather conditions in Los Angeles County instead.

## Pre-Processing

The preprocessing phase focused on cleaning and preparing our data from the weather stations in Los Angeles for modeling. To ensure that our model produced the best output, we took the following steps:

### Dropping Unnecessary Columns:
Many columns were determined to be irrelevant for predicting the weather in LA County. These columns included 'Unnamed: 32', 'Unnamed: 41', 'metar', 'metar_origin', 'pressure_change_code', 'weather_cond_code', 'pressure_change_code', 'visibility', 'cloud_layer_1', 'cloud_layer_2', 'cloud_layer_3', 'wind_cardinal_direction', 'cloud_layer_1_code', 'cloud_layer_2_code', 'cloud_layer_3_code', and 'heat_index'.

### Handling Missing Values:
Missing values were common in many columns, but for our target column, 'air_temp', there were only a few missing values. We handled this by dropping those rows, and for numeric columns, we replaced missing values with the mean of their respective columns to maintain consistency.

### Data Type Conversion:
The 'air_temp' column was converted to a numeric type to ensure accurate calculations during modeling.

### Date and Time Components Extraction:
The 'date_time' column was converted to a datetime format, and additional columns for year, month, day, and hour were extracted. This allowed for a more detailed interpretation of the weather data and enabled us to model weather conditions based on hourly data.

### Calculating Hourly and Yearly Averages:
Hourly average temperatures were calculated by grouping data by year, month, day, and hour. Similar calculations were performed for wind speed and sea level pressure to obtain hourly averages. Yearly average temperatures and precipitation were also calculated to observe long-term trends.

### Normalization:
Features such as temperature, wind speed, and sea level pressure were normalized using `MinMaxScaler` to ensure that our features remained consistent. This step was crucial for improving the performance of our models.

### Scatter Plot:
![scatterplot](https://github.com/user-attachments/assets/8dcf46ea-5671-473e-a66e-62f071899ac0)

The scatter plot that we made involved the yearly average temperature over 24 years spanning 2000 to 2024. Each blue dot represents the average temperature for a specific year. The X-axis shows the years, and the Y-axis represents the yearly average temperatures. The plot reveals variability in average temperatures across the years, indicating fluctuations and possible trends in temperature changes. This visualization helps identify patterns, such as increasing or decreasing temperatures, which may indicate climate change. It also highlights any outliers or unusual values that warrant further investigation, providing a foundational understanding for further analysis and modeling.

### Pair Plot:
![pairplot](https://github.com/user-attachments/assets/143dd7cc-a989-4852-8956-bfea441d2993)

The pair plot shows the relationships between various weather variables in our dataset. It helps us see patterns, correlations, and potential outliers that are important for building predictive models.

The plots for year, month, day, and hour show how data points are distributed over time. The month, day, and hour data are collected at regular intervals, while the year data is spread continuously from 2000 to 2025.

Sea level pressure varies widely but often centers around 1010-1020 hPa. There are noticeable changes in sea level pressure across different months and hours, hinting at seasonal and daily effects. Relative humidity ranges from 0% to 100%, showing a negative correlation with air temperature and hourly average temperature. This means higher temperatures generally come with lower humidity.

Hourly average temperature strongly correlates with air temperature, as expected. The plots show that temperatures vary significantly throughout the day and year. Wind speed has a wide range but clusters at lower speeds. It changes across different times of the day and year but doesn't strongly correlate with other variables.

These insights are useful for predictive modeling. Air temperature, sea level pressure, and relative humidity have clear patterns and correlations that can improve model accuracy. Daily and seasonal temperature and humidity variations should be included in models. The negative correlation between temperature and humidity and the variability in sea level pressure are important factors for temperature prediction models.

In conclusion, the pair plot helps us understand the relationships between weather variables, highlighting the importance of temporal patterns and correlations for predictive modeling. These insights will help us build more accurate and reliable weather prediction models. This analysis underscores the value of exploratory data analysis in guiding the modeling process.

### Correlation Matrix:
![correlation](https://github.com/user-attachments/assets/fcc597e5-07e2-49f9-a7a1-eac611271665)

The correlation matrix heatmap shows the relationships between different weather variables in our dataset. This is important for selecting features for predictive modeling and understanding how these variables interact.

Air temperature has a strong positive correlation with itself and a moderate positive correlation with hourly average temperature and wind speed. It also has a negative correlation with relative humidity and sea level pressure. Wind speed is moderately positively correlated with air temperature and hourly average temperature, and negatively correlated with sea level pressure. Sea level pressure is negatively correlated with air temperature, wind speed, relative humidity, and hourly average temperature.

Precipitation has weak correlations with most variables but shows a slight positive correlation with wind speed and relative humidity. The hour of the day correlates positively with air temperature and hourly average temperature, showing that temperatures vary significantly within a day. Year and month have very weak correlations with most variables, indicating that short-term factors influence weather more than long-term trends. Relative humidity has a strong negative correlation with air temperature and hourly average temperature, indicating that higher temperatures are associated with lower humidity levels.

Hourly average temperature is strongly positively correlated with air temperature and moderately positively correlated with wind speed and hour. It has a negative correlation with sea level pressure and relative humidity.

These correlations are useful for predictive modeling. Variables like air temperature, wind speed, and hourly average temperature are important for predictions. The negative correlation between air temperature and relative humidity suggests that humidity should be considered when predicting temperature. Weak correlations of year and month with other variables suggest that short-term features like hour are more important for predictions.

In conclusion, the correlation matrix heatmap helps us understand the relationships between weather variables. Strong correlations between temperature-related variables highlight their connections, while weak correlations with long-term factors suggest that immediate conditions are more important for weather predictions. This understanding will help us select the right features and improve the accuracy of our predictive models.

&nbsp;

# Weather Prediction Model 1: Neural Network

The first model we employed for weather prediction is a Neural Network. Neural networks are powerful tools for capturing complex relationships in data, making them suitable for problems where the underlying structure might not be linear. Here's a breakdown of the approach we took:

## Data Preprocessing

1. **Handling missing values**:
   - **Temperature**: Filled with the mean value.
   - **Wind Speed**: Filled with the median value.
   - **Sea Pressure**: Filled with the mean value.

2. **Feature selection**:
   - We focused on the hour as the only feature to predict the weather variables (temperature, wind speed, sea pressure, and humidity).

3. **Normalization**:
   - We normalized the temperature and wind speed features using `MinMaxScaler` to ensure all features are on the same scale.

## Model Building

1. **Splitting data**:
   - We split the data into training and testing sets with an 80/20 ratio. The training set is used to train the model, and the testing set is used to evaluate its performance on unseen data.

2. **Neural Network architecture**:
   - **Input layer**: Takes the hour value as input.
   - **Hidden layers**: Two hidden layers with 12 and 8 neurons respectively, each using a ReLU activation function for non-linearity.
   - **Output layer**: Contains four neurons, one for each target variable (temperature, wind speed, sea pressure, and humidity). It also uses a ReLU activation function.

3. **Training**:
   - We trained the model using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.3 and a mean squared error (mse) loss function.
   - We trained the model for 20 epochs with a batch size of 32 and a validation split of 0.1 to monitor performance on a held-out validation set during training.

---

This structured approach allows us to leverage the power of neural networks to predict multiple weather variables effectively.

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

## Model 1: Neural Network 

### Overview
The neural network model was trained to predict three key weather variables: temperature, wind speed, and sea pressure using hours, days, and years as features. The model architecture included an input layer, two hidden layers, and an output layer, with ReLU activation functions to capture non-linear relationships.

### Model Performance
The model's performance was evaluated using Mean Squared Error (MSE) as the primary metric. Here are the key results:

- **Training and Validation Loss**: Throughout the 20 epochs of training, both the training loss and validation loss stabilized around 0.1827, indicating that the model did not overfit to the training data but also struggled to improve significantly.
- **Test Loss**: The model achieved a final test loss (MSE) of 0.1826, which is consistent with the training and validation loss, showing that the model's predictions on unseen data were in line with its performance on the training data.

### Predicted vs. Actual Values
Upon evaluating the model's predictions, the following observations were made:

- **Temperature**: The model was able to capture the general trends in temperature variations but showed some deviations from the actual values. This indicates that while the model can predict temperature to a certain extent, there is room for improvement in accuracy.
- **Wind Speed**: The predictions for wind speed had a higher degree of error compared to temperature. Wind speed is inherently more variable and influenced by multiple










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

## Introduction and Objectives

The primary goal of this project was to analyze and predict weather conditions in Los Angeles County using a comprehensive dataset from the California Weather Government Website. We focused on identifying trends and patterns in temperature, humidity, wind speed, and other weather variables to understand the local impact of climate change and improve weather prediction accuracy. This section outlines our thought process, methods, and interpretation of results, along with a critical analysis of the outcomes.

## Data Exploration

### Initial Steps

Our initial exploration aimed to correlate weather changes with wildfire patterns, leveraging data from both the California Weather and Wildfire Government Websites. However, the merged dataset contained only 991 rows, which was insufficient for robust modeling. Consequently, we decided to focus solely on the weather data, which provided a much larger dataset of 205,642 rows. This decision was crucial as it ensured we had enough data to train and test our models effectively.

### Data Visualization and Feature Analysis

To understand the relationships between variables, we created scatter plots, pair plots, and correlation matrix heatmaps. These visualizations revealed significant patterns and correlations, such as the negative relationship between temperature and humidity, and the variability in sea level pressure. These insights guided our feature selection and informed our modeling approach, emphasizing the importance of thorough data exploration in the early stages.

## Data Preprocessing

### Handling Missing Values and Normalization

Preprocessing involved dropping unnecessary columns, handling missing values by filling with means or medians, and converting data types. Extracting date and time components allowed us to calculate hourly and yearly averages. Normalizing key features using MinMaxScaler ensured consistency across the dataset, which is critical for the performance of machine learning models.

### Challenges and Solutions

One challenge was the high variability and presence of outliers in variables like wind speed and hourly average temperature. We addressed this by careful normalization and outlier detection. Another issue was the potential non-normal distribution of features, which we mitigated through feature scaling and transformation.

## Model Development

### Neural Network Model

Our first model was a neural network, chosen for its ability to capture complex, non-linear relationships. The architecture included an input layer, two hidden layers with ReLU activation, and an output layer. Despite our efforts, the neural network achieved a mean squared error (MSE) of 0.19, indicating moderate prediction accuracy.

#### Interpretation and Critique

The neural network struggled to improve beyond a certain point, with training and validation loss stabilizing early. This could be due to several factors: the model complexity might not have been sufficient, or our feature selection could have been suboptimal. Additionally, neural networks require extensive hyperparameter tuning and larger datasets to perform optimally, which might have been a limitation.

### K-Nearest Neighbors (KNN) Model

Given the moderate performance of the neural network, we implemented a K-Nearest Neighbors (KNN) model. The KNN model showed significantly better results, with low MSE values for temperature (0.00524), wind speed (0.0186), and sea pressure (0.00661). The simplicity and effectiveness of KNN in capturing non-linear patterns without assuming an underlying distribution made it well-suited for our dataset.

#### Interpretation and Critique

The KNN model's superior performance suggests it was better at capturing the temporal patterns and variations in weather data. However, it also highlighted the variability in wind speed, which remained a challenge. The KNN model's reliance on the distance metric means it can struggle with high-dimensional data, but in our case, the selected features and normalization helped mitigate this issue.

## Believability and Shortcomings

### Believability of Results

The results, particularly from the KNN model, are believable given the dataset and preprocessing steps. The low MSE values for temperature and sea pressure indicate that the model effectively captured the patterns in the data. However, the moderate MSE for wind speed suggests that this variable's inherent variability requires more sophisticated modeling techniques.

### Shortcomings

- **Data Quality**: Handling missing values and outliers was a significant challenge. Ensuring data quality through more robust preprocessing could improve model performance.
- **Feature Engineering**: More advanced feature engineering, such as interaction terms or PCA, could enhance model accuracy.
- **Model Complexity**: The neural network might have benefited from a more complex architecture or different activation functions. Extensive hyperparameter tuning could also improve performance.
- **Additional Data**: Incorporating more diverse datasets, such as satellite data or regional weather records, could provide additional context and improve predictions.

## Future Work

Future projects should focus on integrating additional datasets to enrich the analysis. Exploring more advanced machine learning models like ensemble methods (Random Forests or Gradient Boosting) or deep learning architectures could provide better results. Extensive hyperparameter tuning and cross-validation techniques are essential for optimizing model performance. Addressing data quality issues and continually updating the dataset will also be crucial for maintaining prediction accuracy.

## Conclusion

Our project underscores the value of thorough data exploration and preprocessing in building effective predictive models. The KNN model demonstrated superior performance in predicting weather variables compared to the neural network. These insights contribute to a better understanding of weather patterns in Los Angeles County and support efforts in urban planning, public safety, and environmental conservation. We are excited about the possibilities for further improving weather prediction models and look forward to refining our approach in future projects.

</details>

<details open>
<summary><h2>Conclusion</h2></summary>
&nbsp;

In this project, we aimed to analyze and predict weather conditions in Los Angeles County using a comprehensive dataset from the California Weather Government Website, which includes over twenty years of weather data. Our focus was on identifying trends and patterns in temperature, humidity, wind speed, and other weather variables to understand the local impact of climate change and improve weather prediction accuracy.

We began our data exploration by attempting to correlate weather changes with wildfire patterns. However, due to insufficient data when merging the two datasets, we decided to focus solely on the weather data, resulting in a robust dataset of 205,642 rows. Our preprocessing steps included dropping unnecessary columns, handling missing values, converting data types, and extracting date and time components. We also calculated hourly and yearly averages and normalized key features to prepare the data for modeling.

Our analysis included creating scatter plots, pair plots, and correlation matrix heatmaps to visualize relationships between variables and identify significant patterns. We initially developed a neural network model to predict temperature, wind speed, sea level pressure, and humidity based on hourly data. The model achieved a mean squared error (MSE) of 0.19, indicating moderate prediction accuracy. 

We then implemented a K-Nearest Neighbors (KNN) model, which showed promising results with low MSE values for temperature (0.00524), wind speed (0.0186), and sea pressure (0.00661). This model effectively captured the temporal patterns and variations in temperature, although it struggled more with the variability in wind speed. The KNN model's performance was superior to the neural network, highlighting its effectiveness in predicting weather variables.

Reflecting on our project, there are several things we wish we could have done differently. Firstly, integrating more diverse datasets, such as satellite data or other regional weather records, could have enriched our analysis and potentially improved our models' accuracy. We also recognize the value of incorporating more advanced feature engineering techniques, such as creating interaction terms or using principal component analysis (PCA) to reduce dimensionality and focus on the most impactful features.

Furthermore, applying more sophisticated machine learning models like ensemble methods (e.g., Random Forests or Gradient Boosting) or experimenting with deep learning architectures might have provided better results. Additionally, a more extensive hyperparameter tuning process for our neural network and KNN models could have optimized their performance further.

We also acknowledge the importance of addressing data quality issues more thoroughly, particularly handling missing values and outliers. Ensuring the data is clean and accurately reflects real-world conditions is crucial for reliable predictions. In future projects, implementing cross-validation techniques, such as time-series split, could help in better evaluating model performance and reducing overfitting.

In conclusion, our analysis underscores the value of thorough data exploration and preprocessing in building effective predictive models. By incorporating a wide range of features and focusing on temporal patterns, we were able to enhance our models' accuracy. The KNN model, in particular, demonstrated superior performance in predicting weather variables compared to the neural network. The insights gained from this project contribute to better understanding weather patterns in Los Angeles County and support efforts in urban planning, public safety, and environmental conservation. Future work could focus on further refining feature engineering, exploring more advanced modeling techniques, and continually updating the dataset to maintain prediction accuracy. This project has provided a solid foundation, and we are excited about the possibilities for further improving weather prediction models.

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
