FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies early (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary project files
# Note: we exclude tests and other junk via .dockerignore
COPY app ./app
COPY src ./src
COPY configs ./configs
COPY artifacts/model ./artifacts/model
COPY data/split/test ./data/split/test

# Expose the API port
EXPOSE 8000

# Run the application
# Run the application with dynamic port support (Render requirements)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
