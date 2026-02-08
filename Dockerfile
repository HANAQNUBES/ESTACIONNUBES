# Dockerfile para HANAQ Dashboard
FROM python:3.13-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libproj-dev \
    libgdal-dev \
    libgeos-dev \
    libspatialite-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install flask cfgrib xarray matplotlib cartopy scipy netCDF4 h5netcdf requests geopandas tqdm

# Exponer el puerto del dashboard
EXPOSE 5000

# Comando para iniciar el dashboard
CMD ["python", "PRONOSTICO.py"]
