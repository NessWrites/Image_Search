# frontend/Dockerfile
FROM python:3.9-slim
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the necessary files directly
COPY main.py .
COPY config.py .

# Run the correct file (main.py)
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]