FROM python:3.13

WORKDIR /cnn-lstml-model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Nhận biến môi trường từ GitHub Actions
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Set environment để ứng dụng có thể dùng
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

COPY . .

EXPOSE 5000
CMD ["python", "train.py"]
