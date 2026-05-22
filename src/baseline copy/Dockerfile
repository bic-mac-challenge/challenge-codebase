FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# -----------------------------
# Install dependencies
# -----------------------------

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# -----------------------------
# Copy code
# -----------------------------

COPY outputs ./outputs
COPY predict.py . 
COPY unet.py .
COPY dataset.py .
# -----------------------------
# Run inference
# -----------------------------
ENTRYPOINT ["python", "predict.py", "--features_dir", "/data/features", "--output_ct", "/data/output/ct.nii.gz"]
