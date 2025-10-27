from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Reshape, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_cnn_lstm_model(input_shape=(288, 2)):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Reshape((-1, 128)))

    model.add(LSTM(units=150, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(LSTM(units=100, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model
