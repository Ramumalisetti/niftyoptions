# Use the official Playwright Python image (it has all OS dependencies pre-installed)
FROM mcr.microsoft.com/playwright/python:v1.43.0-jammy

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render uses PORT environment variable, defaults to 10000)
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
