# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose your FastAPI port
EXPOSE 10000

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
