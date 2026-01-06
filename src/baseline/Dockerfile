FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir nibabel

# Copy baseline code
COPY model.py .
COPY utils.py .

# Set entrypoint to run the baseline model
ENTRYPOINT ["python", "model.py"]
