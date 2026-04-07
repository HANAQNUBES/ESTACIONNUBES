#  PROYECTO HANAQ 

> Proyecto para monitoreo y pronóstico del estado de la atmosfera mediante

[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Estado](https://img.shields.io/badge/Estado-Activo-brightgreen.svg)]()
[![Licencia](https://img.shields.io/badge/Licencia-MIT-blue.svg)](LICENSE)
---
## 📋 Tabla de Contenidos

- <a href="#seccion1">Descripción General</a>
- <a href="#seccion2">Tecnologías</a>
- <a href="#seccion3">Estructura del Proyecto</a>



---

<h2 id="seccion1">Descripción General</h2>

**¿Qué hace este proyecto?**

Este proyecto integra datos de una estación de observación de cielo, midiendo temperatura, humedad y precipitación; ademas de proporcionar una imagen del cielo la cual se utiliza para deducir la cobertura nubosa y así obtener un pronóstico.

Ademas de ello, el proyecto incluye otros modelos meteorologicos como el eta,gfs y wrf para la precipitación futura.

**¿Por qué existe?**

Para centralizar el monitoreo climático en un solo sistema, facilitando la toma de decisiones en agricultura y gestión de riesgos.

**¿Quién lo usa?**

- Meteorólogos
- Ingenieros agrícolas
- Gestores de riesgos climáticos

---


<h2 id="seccion2">🚀 Tecnologías</h2>

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.14 |
| **Procesamiento datos** | xarray, cfgrib, numpy, pandas |
| **Visualización** | matplotlib, cartopy, PIL |
| **Descarga** | requests, urllib3, tqdm |
| **Web** | Flask (para API) |
| **Paralelismo** | multiprocessing, concurrent.futures |
| **Geoespecial** | cartopy, shapefiles |
| **Sistema** | os, glob, shutil, time, datetime |

---

<h2 id="seccion3">🚀 Estructura del Proyecto</h2>

```bash
proyecto/
│
├── 📄 app.py                    # Aplicación principal / API
├── 📄 requirements.txt          # Dependencias
├── 📄 dasoboard.html            # Estructura de pag web
│
├── 📁 comunicacion/             # Maneja comunicaciones en si mismo y con rasbery
│   ├── 📄 __init__.py/          
│   └── 📄 comu.py               # Archivo principal
│
├── 📁 forecast/                 # Maneja lo relacionado con los pronosticos(mapas) 
│   ├── 📄 Down_and_Consolid.py  # Descarga y consolida archivos en netcdf           
│   └── 📄 mapeado.py            # Controla las descargas y genera los mapas y gifs
│
├── 📁 imagenes_cielo/           # Guarda imagenes de cielo de la comunicacion con la rasbey
│   └── 📄 *cielo_{YYYYMMDD_HHMMSS}.jpg
│
└── 📁 temp/                     # Datos temporales (se genera al ejecutar)
    ├── 📁 datos_gfs/            # Archivos GFS
    │   ├── 📁 Consolidados/     # NetCDF consolidados
    │   ├── 📁 imgs/             # PNGs generados
    │   ├── 📄 *.grib2           # Archivos descargados
    │   └── 📄 log_file.txt      # Registro de descargas
    │
    ├── 📁 datos_eta/            # Archivos ETA (misma estructura)
    ├── 📁 datos_wrf/            # Archivos WRF (misma estructura)
    └── 📁 shapefiles/           # Shapefiles para mapas
