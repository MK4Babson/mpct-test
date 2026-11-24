# Container for the bill-checking API (works on Google Cloud Run, Render, Fly.io, etc.)
FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api ./api
COPY src ./src
COPY sql ./sql
COPY data/raw ./data/raw
COPY data/north_koel ./data/north_koel
ENV PORT=8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
