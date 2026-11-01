FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
ENV PYTHONPATH=/app/src
CMD ["sh", "-c", "uvicorn main:app --app-dir /app/src --host 0.0.0.0 --port ${PORT:-8000}"]
