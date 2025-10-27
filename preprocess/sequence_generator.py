import numpy as np

def create_sequences(df, n_steps=288, future_step=12):
    car_count = df['car_count'].values
    hour = df['hour'].values

    if len(car_count) <= n_steps + future_step:
        raise ValueError(f"Dữ liệu quá ngắn, cần ít nhất {n_steps + future_step + 1} phần tử")

    X = np.zeros((len(car_count) - n_steps - future_step, n_steps, 2))
    y = np.zeros(len(car_count) - n_steps - future_step)

    for i in range(len(car_count) - n_steps - future_step):
        X[i, :, 0] = car_count[i:i + n_steps]
        X[i, :, 1] = hour[i:i + n_steps]
        y[i] = car_count[i + n_steps + future_step]

    return X, y
