# uploader/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY upload_script.py .

# 컨테이너가 시작되면 이 스크립트를 실행
CMD ["python", "upload_script.py"]