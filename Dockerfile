# =========================
# Dockerfile
# =========================
FROM python:3.12-slim

# Define diretório de trabalho no container
WORKDIR /app

# Instala dependências do sistema necessárias para o Librosa e afins
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Expondo a porta do container
EXPOSE 8000

# Gunicorn lendo FLASK_PORT (padrão 8000 se não setado)
CMD ["sh", "-c", "gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:${FLASK_PORT:-8000} api.wsgi:app"]
