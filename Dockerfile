FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/uploads nltk_data /root/nltk_data

# Set environment variables
ENV PORT=10000
ENV NLTK_DATA=/root/nltk_data
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

# Start app with Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --log-level info app:app"]
```

## 📁 **3. .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Flask
instance/
.webassets-cache

# Environment variables
.env
.env.local

# Uploads
uploads/
*.wav
*.webm
*.mp4

# NLTK data
nltk_data/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

## 📁 **4. .env.example** (Create this for documentation)
```
# Environment Variables Template
# Copy this to .env for local development

GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here
PORT=10000
