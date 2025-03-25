
# Riverside Hourly Temperature Prediction using LSTM

This project focuses on predicting hourly temperature for Riverside Municipal Airport, CA US using a Long Short-Term Memory (LSTM) neural network. 

## Project Overview

The project leverages historical hourly temperature data from the National Centers for Environmental Information (NCEI) to build a predictive model capable of forecasting future temperatures. The dataset covers a period of 2015-2024 and was acquired from NCEI.

## Data Preprocessing

1. **Data Collection:** The hourly temperature data was downloaded for each year (2015-2024) from the NCEI website.
2. **Data Combination:** The downloaded CSV files for each year were concatenated into a single combined CSV file, `combined.csv`.
3. **Datetime Conversion:** The 'DATE' column was converted to datetime objects and formatted for easier analysis and modeling.
4. **Data Cleaning:** Missing values in the `HourlyWetBulbTemperature` column were filled with the mean temperature.
5. **Data Preparation for LSTM:** The data was converted into a format suitable for an LSTM model, where the input is a sequence of past temperatures (window_size=30), and the output is the next temperature value.

## Model Development

An LSTM model was constructed using Keras and TensorFlow. The architecture consists of an input layer, an LSTM layer with 128 units, a dense layer with 32 units and ReLU activation, and an output layer with a linear activation function.

## Model Training and Evaluation

1. **Train-Validation-Test Split:** The data was split into training, validation, and test sets (75%, 15%, 10% respectively).
2. **Model Training:** The model was trained on the training data using the Adam optimizer and Mean Squared Error loss.
3. **Model Evaluation:** The model's performance was evaluated on the validation set during training to track and prevent overfitting.
4. **Test Set Evaluation:** The model was finally evaluated on the unseen test set to assess its ability to generalize to new data.

## Results

The LSTM model demonstrated promising results in predicting hourly temperatures. The training, validation, and test predictions were compared with the actual temperature values. The model is capable of providing reasonably accurate predictions of future temperatures, contributing to a greater understanding of the temperature patterns in Riverside.

## Future Work

- **Hyperparameter Optimization:** Exploring various hyperparameter configurations to further improve model accuracy.
- **Feature Engineering:** Incorporating additional meteorological features (e.g., wind speed, humidity) to potentially improve model performance.
- **Deployment:** Deploying the trained model for real-time temperature prediction.


## Libraries used:

- Pandas
- Glob
- TensorFlow
- Keras
- Matplotlib
- Numpy
