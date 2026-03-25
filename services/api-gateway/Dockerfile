FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia ml-service para integração standalone
COPY ../ml-service/app/ /ml-service/app/

COPY app/ .

EXPOSE 8000

CMD ["python", "main.py"]
