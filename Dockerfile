FROM python:3.11-slim

# Install system dependencies required by Playwright/Chromium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    libgconf-2-4 \
    libnss3 \
    libxss1 \
    libasound2 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and the Chromium browser
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY . .

# Expose port (Render uses PORT environment variable, usually defaults to 10000)
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
