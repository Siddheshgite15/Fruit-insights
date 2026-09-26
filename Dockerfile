# ── Stage 1: dependency builder ──────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs needed by OpenCV headless & scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/        ./app/
COPY models/     ./models/
COPY templates/  ./templates/
COPY static/     ./static/
COPY run.py      .
COPY .env        .

# Create runtime directories
RUN mkdir -p uploads logs

EXPOSE 8000

CMD ["python", "run.py"]
