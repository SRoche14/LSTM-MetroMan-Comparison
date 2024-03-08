import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
import netCDF4
import math
import warnings


# this function loads the data. It returns a dataframe with the widths, water surface elevation
# discharge, and slope values.
def load_data(name):
    f = netCDF4.Dataset(f'/Users/sroche/PycharmProjects/stackOverflowTesting/{name}')
    time = f.groups['Reach_Timeseries'].variables['t']
    widths = f.groups['Reach_Timeseries'].variables['W']
    discharge = f.groups['Reach_Timeseries'].variables['Q']
    slope = np.array(f.groups['Reach_Timeseries'].variables['S'])

    water_surface_elevation = np.array(f.groups['Reach_Timeseries'].variables['H'])

    time_np = np.array(time).squeeze()
    widths_np = np.array(widths)
    slope_np = np.array(slope)
    water_se_np = np.array(water_surface_elevation)
    discharge_np = np.array(discharge)

    titles = [
      "Width",
      "Surface Slope",
      "Surface Elevation",
      "Discharge"
    ]

    dataset = np.stack((widths_np, slope_np, water_se_np, discharge_np), axis=1)
    dfs = [pd.DataFrame(data=dataset[:, :, i], columns=titles) for i in range(1)]
    df = dfs[0]
    print("******", name, "******")
    return df


# this function splits the dataframe from the load_data function. This creates "windows" of data
# so that we can train the LSTMs. Note that the window_size = 2 in this case. For test cases were there are more
# than 12 observations, we set window_size = 14
def df_to_X_y(df, window_size=2):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size, :3]]
        X.append(row)
        label = df_as_np[i+window_size, 3]
        y.append(label)
    return np.array(X), np.array(y)


# this runs normalization on each features, using mean and standard deviations along each feature vector
def normalize_np(array):
    warnings.filterwarnings("error", category=RuntimeWarning)
    mean = np.mean(array, axis=1, keepdims=True)
    std = np.std(array, axis=1, keepdims=True)
    try:
        normalized_array = (array - mean) / std
    except RuntimeWarning:
        row_indices = np.where((std == 0).any(axis=1))[0]
        for x in row_indices:
            for i, element in enumerate(std[x][0]):
                if element == 0:
                    std[x][0][i] = 1
                    mean[x][0][i] = 0
        normalized_array = (array - mean) / std
    return normalized_array


# the following is used for testing the models on the test data. As you can see, the autoencoder code is
# commented out. To create the auto-lstm, uncomment the autoencoder lines, including the encoded_X1_train line
# In this particular use case, we are looking at the MiddleRiver test case, which has 12 observations.
# For rivers with 12 observations, we use an LSTM with 2 units as shown here. For longer rivers,
# uncomment the code with 128 units.
def run_model(data, name):
    X1, y1 = df_to_X_y(data)
    nrmse_arr = []
    nse_arr = []
    pred = []
    # this '5' controls how many times we test our model on the test set. We report the average
    # of the 5 results to increase robustness.
    for x in range(5):
        X1_train, y1_train = X1[:math.floor(len(X1)*.40)], y1[:math.floor(len(X1)*.40)]
        print(X1_train.shape)
        X1_test, y1_test = X1[math.floor(len(X1)*.40):], y1[math.floor(len(X1)*.40):]
        print(f'Y1 Test: {y1_test}')
        normal_X1_train = normalize_np(X1_train)

        # THIS AUTOENCODER IS FOR RIVERS WITH MORE THAN 12 OBSERVATIONS
        # autoencoder = Sequential()
        # autoencoder.add(LSTM(64, activation='relu', input_shape=(14, 3), return_sequences=True))
        # autoencoder.add(LSTM(32, activation='relu', return_sequences=False))
        # autoencoder.add(RepeatVector(14))
        # autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
        # autoencoder.add(LSTM(64, activation='relu', return_sequences=True))
        # autoencoder.add(TimeDistributed(Dense(3)))
        #
        # autoencoder.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
        # autoencoder.fit(normal_X1_train, normal_X1_train, epochs=200, verbose=0)

        # THIS AUTOENCODER IS FOR RIVERS WITH 12 OBSERVATIONS
        # autoencoder = Sequential()
        # autoencoder.add(LSTM(2, activation='relu', input_shape=(2, 3), return_sequences=True))
        # autoencoder.add(RepeatVector(2))
        # autoencoder.add(LSTM(2, activation='relu', return_sequences=True))
        # autoencoder.add(TimeDistributed(Dense(3)))
        #
        # autoencoder.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01))
        # autoencoder.fit(normal_X1_train, normal_X1_train, epochs=500, verbose=0)
        #
        # encoded_X1_train = autoencoder.predict(normal_X1_train)
        # print(encoded_X1_train.shape)

        lstm_model = Sequential()
        lstm_model.add(
            LSTM(2, activation='relu', return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotUniform(), input_shape=(2, 3)))
        lstm_model.add(Flatten())
        lstm_model.add(Dense(1, 'linear'))
        # lstm_model.add(
        #     LSTM(128, activation='relu', input_shape=(14, 3)))  # Adjust input shape based on encoded features
        # lstm_model.add(Dropout(0.25))
        # lstm_model.add(Flatten())
        # lstm_model.add(Dense(64))
        # lstm_model.add(Dropout(0.1))
        # lstm_model.add(Dense(8, 'linear'))
        # lstm_model.add(Dense(1, activation='linear'))  # Adjust output dimension and activation as needed

        lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.1), metrics=[RootMeanSquaredError()])

        # A NOTE: number of epochs varied depending on test case. We can include this information if needed.
        lstm_model.fit(normal_X1_train, y1_train, epochs=40, verbose=0)

        predictions = []

        for i in range(len(X1_test)):
            test_data_window = np.array([X1_train[-1]])
            normal_test_window = normalize_np(test_data_window)
            test_prediction = lstm_model.predict(normal_test_window)
            true_test_value = y1_test[i]
            predictions.append(test_prediction.squeeze())

            # Add true test value to training set
            X1_train = np.concatenate([X1_train, X1_test[i].reshape(1, *X1_train.shape[1:])], axis=0)
            y1_train = np.concatenate([y1_train, np.array([true_test_value])], axis=0)
            normal_X1_train = normalize_np(X1_train)

            # autoencoder.fit(normal_X1_train, normal_X1_train, epochs=120, verbose=0)

            # encoded_X1_train = autoencoder.predict(normal_X1_train)
            # print(encoded_X1_train.shape)

            lstm_model.fit(normal_X1_train, y1_train, epochs=70, verbose=0)

        # evaluate forecastes vis NRMSE
        nrmse_num = np.sqrt((np.sum(np.square(np.subtract(y1_test[:], np.array(predictions[:]).squeeze()))) / len(y1_test[:])))
        nrmse_denom = np.mean(y1_test[:])
        nrmse = nrmse_num / nrmse_denom
        print('Test Set NRMSE: %.3f' % nrmse)
        nrmse_arr.append(nrmse)

        # evaluate forecasts via NSE
        numerator = np.sum(np.square(np.subtract(y1_test[:], np.array(predictions[:]).squeeze())))
        denominator = np.sum(np.square(y1_test[:] - np.mean(y1_test[:])))
        nse = 1 - (numerator / denominator)
        print('Test Set NSE: %.3f' % nse)
        nse_arr.append(nse)

        pred.append(np.array(predictions))
        # show plot on last run through (remember we do 5 times to make the results more robust)
        if x == 4:
            averaged_array = np.mean(pred, axis=0)
            plt.plot(averaged_array, color="orange", label="Predictions")
            plt.plot(y1_test, color="blue", label="Actuals")
            plt.xlabel("Day")
            plt.ylabel("Discharge (m/s^3)")
            plt.legend()
            plt.savefig(f'{name}.png')
            plt.show()
    print(f'PREDICTIONS: {np.mean(pred, axis=0)}')
    print('Average NRMSE: %.3f' % np.mean(np.array(nrmse_arr)))
    print('Average NSE: %.3f' % np.mean(np.array(nse_arr)))


# this function is for validating our model on each validation set. it runs analogously to the run_model function
# that you just saw.
def test_hyperparameters(data):
    X1, y1 = df_to_X_y(data)
    X1_train, y1_train = X1[:math.floor(len(X1)*.20)], y1[:math.floor(len(X1)*.20)]
    print(X1_train.shape)

    X1_val, y1_val = X1[math.floor(len(X1)*.20):math.floor(len(X1)*.40)], y1[math.floor(len(X1)*.20):math.floor(len(X1)*.40)]

    normal_X1_train = normalize_np(X1_train)

    normal_X1_val = normalize_np(X1_val)

    # THIS AUTOENCODER IS FOR RIVERS WITH MORE THAN 12 OBSERVATIONS
    # autoencoder = Sequential()
    # autoencoder.add(LSTM(64, activation='relu', input_shape=(14, 3), return_sequences=True))
    # autoencoder.add(LSTM(32, activation='relu', return_sequences=False))
    # autoencoder.add(RepeatVector(14))
    # autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
    # autoencoder.add(LSTM(64, activation='relu', return_sequences=True))
    # autoencoder.add(TimeDistributed(Dense(3)))
    #
    # autoencoder.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
    # autoencoder.fit(normal_X1_train, normal_X1_train, epochs=200, verbose=0)

    # THIS AUTOENCODER IS FOR RIVERS WITH 12 OBSERVATIONS
    autoencoder = Sequential()
    autoencoder.add(LSTM(2, activation='relu', input_shape=(2, 3), return_sequences=True))
    autoencoder.add(RepeatVector(2))
    autoencoder.add(LSTM(2, activation='relu', return_sequences=True))
    autoencoder.add(TimeDistributed(Dense(3)))

    autoencoder.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.01))
    autoencoder.fit(normal_X1_train, normal_X1_train, epochs=500, verbose=0)
    #

    # Extract features using the trained autoencoder
    encoded_X1_train = autoencoder.predict(normal_X1_train)
    encoded_X1_val = autoencoder.predict(normal_X1_val)

    # Now, define and train your LSTM using the encoded features
    lstm_model = Sequential()
    lstm_model.add(InputLayer(2, 3))
    lstm_model.add(LSTM(2, activation='relu', return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    lstm_model.add(Flatten())
    lstm_model.add(Dense(1, 'linear'))
    # lstm_model.add(LSTM(128, activation='relu', input_shape=(2, 3)))  # Adjust input shape based on encoded features
    # lstm_model.add(Dropout(0.25))
    # lstm_model.add(Flatten())
    # lstm_model.add(Dense(64))
    # lstm_model.add(Dropout(0.1))
    # lstm_model.add(Dense(8, 'linear'))
    # lstm_model.add(Dense(1, activation='linear'))  # Adjust output dimension and activation as needed

    cp1 = ModelCheckpoint('model/', save_best_only=True)
    lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])

    lstm_model.fit(encoded_X1_train, y1_train, validation_data=(encoded_X1_val, y1_val), epochs=200, callbacks=[cp1], verbose=1)

    predictions = []

    for i in range(len(X1_val)):

        # Make test prediction
        test_data_window = np.array([X1_train[-1]])
        normal_test_window = normalize_np(test_data_window)
        test_prediction = lstm_model.predict(normal_test_window)
        true_test_value = y1_val[i]

        predictions.append(test_prediction.squeeze())

        # Add true test value to training set
        X1_train = np.concatenate([X1_train, X1_val[i].reshape(1, *X1_train.shape[1:])], axis=0)
        y1_train = np.concatenate([y1_train, np.array([true_test_value])], axis=0)

        normal_X1_train = normalize_np(X1_train)

        print("###Autoencoder training###")
        autoencoder.fit(normal_X1_train, normal_X1_train, epochs=100, verbose=0)
        encoded_X1_train = autoencoder.predict(normal_X1_train)

        print("###LSTM training###")
        lstm_model.fit(encoded_X1_train, y1_train, epochs=10, verbose=0)

    # evaluate forecastes vis NRMSE
    nrmse_num = np.sqrt((np.sum(np.square(np.subtract(y1_val[:], np.array(predictions[:]).squeeze()))) / len(y1_val[:])))
    nrmse_denom = np.mean(y1_val[:])
    nrmse = nrmse_num / nrmse_denom
    print('Test Set NRMSE: %.3f' % nrmse)

    # evaluate forecasts via NSE
    numerator = np.sum(np.square(np.subtract(y1_val[:], np.array(predictions[:]).squeeze())))
    denominator = np.sum(np.square(y1_val[:] - np.mean(y1_val[:])))
    nse = 1 - (numerator / denominator)
    print('Test Set NSE: %.3f' % nse)
    plt.plot(predictions, color="orange", label="Predictions")
    plt.plot(y1_val, color="blue", label="Actuals")
    plt.legend()
    plt.show()


# list of rivers with greater than 12 observations, was useful for running the code
river_list = ["Brahmaputra.nc", "IowaRiver.nc", "Jamuna.nc", "Kushiyara.nc", "MississippiIntermediate.nc",
              "MissouriDownstream.nc", "MissouriMidsection.nc", "MissouriUpstream.nc", "OhioSection1.nc",
              "OhioSection2.nc", "OhioSection3.nc", "OhioSection4.nc", "OhioSection5.nc", "OhioSection7.nc",
              "OhioSection8.nc", "Padma.nc", "SeineDownstream.nc", "SeineUpstream.nc"]

# UNCOMMENT this if you would like to use the list

# for river in river_list:
#     dataset = load_data("MiddleRiver.nc")
#     # test_hyperparameters(dataset)
#     run_model(dataset, river)
dataset = load_data("MiddleRiver.nc")
run_model(dataset, "MiddleRiver.nc")
