FROM python:3.8-slim-buster
WORKDIR /frontend
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit","run","frontend.py"]