FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer and skip
# reinstalling everything when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
