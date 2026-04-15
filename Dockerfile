# Usa una imagen basada en python
FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    # Cartopy/mapas
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    # GRIB files
    libeccodes-dev \
    libeccodes-data \
    # Compilación
    gcc \
    g++ \
    # PostgreSQL
    libpq-dev \
    # Otros útiles
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copiar solo requirements.txt primero (mejor caching)
COPY requirements.txt .

# Instalar dependencias
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Variable de entorno para logs sin buffer
ENV PYTHONUNBUFFERED=1

# Copiar TODO el código (incluyendo comunicacion/ y forecast/)
COPY . .

# Exponer puerto
EXPOSE 5000

# Ejecutar con gunicorn (1 worker es suficiente para empezar)
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "app:app"]


