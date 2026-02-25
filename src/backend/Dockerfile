FROM python:3.11-slim

WORKDIR /code

# Install build dependencies for numpy (needed for qdrant-client)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]