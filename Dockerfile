FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Render מספק PORT כ-ENV
CMD ["bash", "-lc", "exec gunicorn main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT} --timeout 180 --workers 2"]

