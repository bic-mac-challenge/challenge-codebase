FROM python:3.12-slim

RUN pip install --no-cache-dir nibabel numpy scipy tqdm

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

WORKDIR /app
