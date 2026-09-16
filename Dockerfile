FROM python:3.14-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r  requirements.txt

COPY flight_price_model.pkl .
COPY serve_api_class.py .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn serve_api_class:app --host 0.0.0.0 --port ${PORT:-10000}"]