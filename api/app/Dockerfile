FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /api/app/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /api/app/