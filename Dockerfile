FROM python:3.10-slim

# System packages needed for OpenCV and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install in correct order to avoid numpy conflicts
RUN pip install --no-cache-dir "numpy==1.26.4"
RUN pip install --no-cache-dir tensorflow==2.15.0
RUN pip install --no-cache-dir "h5py==3.11.0" "pillow==10.3.0"
RUN pip install --no-cache-dir \
    "scikit-learn==1.4.2" \
    "matplotlib==3.8.4" \
    "pandas==2.2.2" \
    "tqdm==4.66.2"
RUN pip install --no-cache-dir \
    "fastapi==0.110.0" \
    "uvicorn[standard]==0.29.0" \
    "python-multipart==0.0.9" \
    "aiofiles==23.2.1"
RUN pip install --no-cache-dir "opencv-python-headless==4.9.0.80"
RUN pip install --no-cache-dir "ultralytics==8.2.0"
RUN pip install --no-cache-dir roboflow
# Pin numpy back after ultralytics/roboflow may have upgraded it
RUN pip install --no-cache-dir "numpy==1.26.4" --force-reinstall

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p models/checkpoints models/yolo logs data/raw \
    data/manual/exterior_facade \
    data/manual/office_interior \
    data/manual/warehouse \
    data/yolo_dataset

EXPOSE 8000

# Default: serve existing models
# To train: docker run -e ROBOFLOW_API_KEY=xxx buildingyolo python main.py --train-only
CMD ["python", "main.py", "--serve"]
