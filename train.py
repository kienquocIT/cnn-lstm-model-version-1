import os

import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from preprocess.data_loader import load_data
from preprocess.data_preprocess import preprocess_data
from preprocess.sequence_generator import create_sequences
from models.cnn_lstm_model import build_cnn_lstm_model
from utils.plot_utils import plot_training_history
import boto3
# --- 1. Load data ---
data = load_data("https://cnn-lstm-s3-storage.s3.ap-southeast-2.amazonaws.com/model/train/parking_data.csv")
min_car_count, max_car_count = data['car_count'].min(), data['car_count'].max()

# --- 2. Preprocess ---
processed_data, scaler_car_count, scaler_hour = preprocess_data(data)
joblib.dump(scaler_car_count, 'scaler_car_count.pkl')
joblib.dump(scaler_hour, 'scaler_hour.pkl')
print("Scaler đã được lưu vào file.")

# --- 3. Create sequences ---
n_steps, future_step = 288, 12
X, y = create_sequences(processed_data, n_steps=n_steps, future_step=future_step)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)

# --- 4. Build & compile model ---
model = build_cnn_lstm_model(input_shape=(n_steps, 2))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# --- 5. Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('best_cnn_lstm_model.keras', monitor='val_loss', save_best_only=True)

# --- 6. Train ---
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

model.save('cnn_lstm_model.keras')
print("Mô hình đã được lưu thành công.")

# --- 7. Plot ---
# plot_training_history(history)

# --- 8. Upload lên S3 ---
def upload_to_s3(local_file, s3_path, bucket_name="cnn-lstm-s3-storage"):
    s3 = boto3.client('s3')
    try:
        print(f"Đang upload {local_file} → s3://{bucket_name}/{s3_path}")
        s3.upload_file(local_file, bucket_name, s3_path)
        print(f"Đã upload {local_file} thành công!")
    except Exception as e:
        print(f"Lỗi upload {local_file}: {e}")

# Danh sách file cần upload
files_to_upload = [
    ("cnn_lstm_model.keras", "model/save/cnn_lstm_model.keras"),
    ("best_cnn_lstm_model.keras", "model/save/best_cnn_lstm_model.keras"),
    ("scaler_car_count.pkl", "model/save/scaler_car_count.pkl"),
    ("scaler_hour.pkl", "model/save/scaler_hour.pkl"),
]

for local_file, s3_path in files_to_upload:
    if os.path.exists(local_file):
        upload_to_s3(local_file, s3_path)
    else:
        print(f"File {local_file} không tồn tại, bỏ qua.")