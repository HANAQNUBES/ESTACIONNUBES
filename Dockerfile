#Usa una imagen basada en python
FROM python:3.10-slim

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
COPY ./requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 5000

#CMD ["python","app.py"]
CMD ["gunicorn","--workers","4","--bind","0.0.0.0:5000","app:app"]