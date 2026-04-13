FROM python:3.11-slim

WORKDIR /app

# Copy requirements first so Docker cache layer is only invalidated
# when requirements.txt changes — not on every code change
COPY requirements.txt .

# Install CPU-only PyTorch first (avoids pulling 2GB+ GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code last — changes here don't invalidate pip layers
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
