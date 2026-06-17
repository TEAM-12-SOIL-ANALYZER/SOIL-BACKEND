FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Hugging Face Spaces require the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

# We use --timeout 120 because TensorFlow models can sometimes take 
# a bit longer to load into memory on the free CPU tiers.
CMD ["gunicorn", "-w", "2", "--timeout", "120", "--bind", "0.0.0.0:7860", "main:app"]
