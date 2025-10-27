import boto3

bucket_name = "cnn-lstm-s3-storage"
key = "model/save/test_upload.txt"
content = b"Xin chao tu local! Day la file test upload."

s3 = boto3.client("s3")

try:
    s3.put_object(Bucket=bucket_name, Key=key, Body=content)
    print(f"✅ Upload thành công: s3://{bucket_name}/{key}")

    # Kiểm tra lại bằng cách tải xuống nội dung
    response = s3.get_object(Bucket=bucket_name, Key=key)
    data = response["Body"].read().decode("utf-8")
    print("📄 Nội dung trong S3:", data)

except s3.exceptions.ClientError as e:
    print("❌ Lỗi quyền truy cập hoặc policy:", e)
except Exception as e:
    print("⚠️ Lỗi khác:", e)
