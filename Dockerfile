# ---------------------------
# Base image
# ---------------------------
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (the Week 1 app)
COPY ai-fastapi-week1/ /app/

# Expose API port
EXPOSE 8000

# Run the FastAPI app via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
