FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Treina o modelo na build (em prod, monte via volume ou artifact store)
RUN python trainer.py

EXPOSE 8001

CMD ["python", "inference.py"]
