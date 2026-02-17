from flask import Flask, render_template_string, request, jsonify, send_file
from datetime import datetime, timezone, timedelta
import threading
import os
import io
import subprocess
import sys

app = Flask(__name__)

# Almacenamiento de datos en tiempo real desde Raspberry Pi
sensor_data = {
    "temperatura": None,
    "humedad": None,
    "presion": None,
    "viento_velocidad": None,
    "viento_direccion": None,
    "precipitacion": None,
    "luz_uv": None,
    "calidad_aire": None,
    "ultima_actualizacion": None,
    "historial": []  # Últimas 24 horas de datos
}

# Estado del modelo ETA
eta_data = {
    "ultima_actualizacion": None,
    "rodada": None,
    "imagen_path": None,
    "procesando": False,
    "error": None
}

# Estado del modelo WRF
wrf_data = {
    "ultima_actualizacion": None,
    "rodada": None,
    "imagen_path": None,
    "procesando": False,
    "error": None
}

# Lock para acceso seguro a datos desde múltiples hilos
data_lock = threading.Lock()
eta_lock = threading.Lock()
wrf_lock = threading.Lock()
cielo_lock = threading.Lock()

# Estado de la imagen del cielo
cielo_data = {
    "imagen_path": "Cielo despejado 06.jpg",
    "ultima_actualizacion": None
}

# Carpeta para imágenes del cielo
CIELO_FOLDER = "imagenes_cielo"
os.makedirs(CIELO_FOLDER, exist_ok=True)

# ============================================================================
# CONFIGURACIÓN DEL MODELO ETA
# ============================================================================
PASTA_SAIDA = "dados_eta_8km"
SHAPEFILE_COUNTRIES = "ne_10m_admin_0_countries.shp"
SHAPEFILE_STATES = "ne_10m_admin_1_states_provinces.shp"


def instalar_dependencias_eta():
    """Instala las dependencias necesarias para el modelo ETA."""
    dependencias = ["xarray", "cfgrib", "matplotlib", "cartopy", "scipy", "numpy", "geopandas", "tqdm"]
    for dep in dependencias:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
        except Exception as e:
            print(f"⚠️ Error instalando {dep}: {e}")


def descargar_shapefiles():
    """Descarga los shapefiles necesarios si no existen."""
    if not os.path.exists(SHAPEFILE_COUNTRIES) or not os.path.exists(SHAPEFILE_STATES):
        try:
            import urllib.request
            import zipfile
            
            # Natural Earth 10m countries
            if not os.path.exists(SHAPEFILE_COUNTRIES):
                url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
                urllib.request.urlretrieve(url, "countries.zip")
                with zipfile.ZipFile("countries.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove("countries.zip")
                print("✓ Shapefile de países descargado")
            
            # Natural Earth 10m states
            if not os.path.exists(SHAPEFILE_STATES):
                url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
                urllib.request.urlretrieve(url, "states.zip")
                with zipfile.ZipFile("states.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove("states.zip")
                print("✓ Shapefile de estados descargado")
                
        except Exception as e:
            print(f"⚠️ Error descargando shapefiles: {e}")


def descargar_datos_eta():
    """Descarga los datos GRIB2 del modelo ETA del CPTEC/INPE."""
    import urllib.request
    from tqdm import tqdm
    import time
    
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    
    # Detecta rodada más reciente (00 o 12 UTC)
    atraso_publicacion = timedelta(hours=4)
    agora = datetime.now(timezone.utc) - atraso_publicacion
    hora_rodada = "00"
    data_base = agora.replace(hour=int(hora_rodada), minute=0, second=0, microsecond=0)
    
    print(f"📅 Rodada detectada: {data_base.strftime('%Y-%m-%d %H:%M')} UTC")
    
    # Genera lista de archivos GRIB2
    arquivos = []
    for h in range(5):
        data_prev = data_base + timedelta(hours=h)
        nome = f"Eta_ams_08km_{data_base.strftime('%Y%m%d%H')}_{data_prev.strftime('%Y%m%d%H')}.grib2"
        url = f"https://dataserver.cptec.inpe.br/dataserver_modelos/eta/ams_08km/brutos/{data_base.strftime('%Y/%m/%d/%H')}/{nome}"
        arquivos.append((nome, url))
    
    # Descarga los archivos GRIB2
    for nome, url in tqdm(arquivos, desc="📥 Descargando archivos ETA"):
        caminho = os.path.join(PASTA_SAIDA, nome)
        if os.path.exists(caminho):
            continue
        
        sucesso = False
        for tentativa in range(3):
            try:
                urllib.request.urlretrieve(url, caminho)
                sucesso = True
                break
            except Exception as e:
                print(f"⚠️ Intento {tentativa+1}/3 falló para {nome}: {e}")
                time.sleep(5)
        
        if not sucesso:
            print(f"❌ Error al descargar {nome}")
    
    return data_base


def procesar_precipitacion_eta():
    """Procesa los datos ETA y calcula la precipitación acumulada."""
    import cfgrib
    import xarray as xr
    import numpy as np
    
    soma_unknown = None
    lat = lon = None
    rodada_inicial = rodada_final = None
    previsao_inicial = previsao_final = None
    
    arquivos = sorted([f for f in os.listdir(PASTA_SAIDA) if f.endswith('.grib2')])[1:]
    
    for i, arq in enumerate(arquivos):
        caminho = os.path.join(PASTA_SAIDA, arq)
        try:
            ds = cfgrib.open_dataset(
                caminho,
                filter_by_keys={'typeOfLevel': 'surface', 'stepType': 'accum'},
                decode_timedelta=True
            )
            
            if 'unknown' not in ds:
                continue
            
            var = ds['unknown']
            
            if soma_unknown is None:
              soma_unknown = var.copy(deep=True)
              # Corrección: buscar nombre correcto de variables
              if 'latitude' in ds:
                lat = ds['latitude']
              elif 'lat' in ds:
                lat = ds['lat']
              else:
                lat = None
              if 'longitude' in ds:
                lon = ds['longitude']
              elif 'lon' in ds:
                lon = ds['lon']
              else:
                lon = None
            else:
                soma_unknown = soma_unknown + var
            
            rodada = ds.time.values
            previsao = ds.valid_time.values
            
            if i == 0:
                rodada_inicial = rodada
                previsao_inicial = previsao
            if i == len(arquivos) - 1:
                rodada_final = rodada
                previsao_final = previsao
                
        except Exception as e:
            print(f"⚠️ Error procesando {arq}: {e}")
    
    # Crear Dataset manteniendo la estructura original
    ds_soma = xr.Dataset(
        {"precipitacao_total": soma_unknown},
        attrs={
            "descricao": "Precipitación acumulada modelo ETA",
            "rodada_inicial": str(rodada_inicial),
            "rodada_final": str(rodada_final),
            "previsao_inicial": str(previsao_inicial),
            "previsao_final": str(previsao_final)
        }
    )
    
    # Guardar NetCDF
    saida_path = "acum_prec.nc"
    ds_soma.to_netcdf(saida_path)
    
    return ds_soma


def generar_mapa_precipitacion(ds_soma):
    """Genera el mapa de precipitación y lo guarda como imagen."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import xarray as xr
    import pandas as pd
    
    # Cargar datos
    ds = xr.open_dataset('acum_prec.nc', decode_timedelta=True)
    # Corrección: buscar nombre correcto de variables
    if 'longitude' in ds:
      ds = ds.assign_coords(longitude=((ds['longitude'] + 180) % 360) - 180)
      lon = ds['longitude']
    elif 'lon' in ds:
      ds = ds.assign_coords(lon=((ds['lon'] + 180) % 360) - 180)
      lon = ds['lon']
    else:
      lon = None
    if 'latitude' in ds:
      lat = ds['latitude']
    elif 'lat' in ds:
      lat = ds['lat']
    else:
      lat = None
    prec = ds['precipitacao_total'][:, :]
    
    # Títulos
    rodada_ini = pd.to_datetime(ds_soma.attrs['rodada_inicial'])
    prev_ini = pd.to_datetime(ds_soma.attrs['previsao_inicial'])
    prev_fim = pd.to_datetime(ds_soma.attrs['previsao_final'])
    titulo = f"Valid: {prev_ini:%HUTC %d/%b} to {prev_fim:%HUTC %d/%b}"
    titulo_ = f"Inic: {rodada_ini:%HUTC %d/%b}"
    
    # Colores y niveles
    cores_prec = ['#ffffff','#dedede','#bfbfbf','#a1a1a1','#828282','#b7f0be','#a0dcb3',
                 '#88c8a9','#71b49e','#5aa093','#438c88','#2b787e','#146473','#1450b4','#2a61bb',
                 '#3f73c2','#5584c9','#6b96d0','#80a7d6','#96b9dd','#accae4','#c1dceb','#d7edf2',
                 '#cebce0','#c9addb','#c49ed5','#bf90d0','#ba81ca','#b472c5','#af63bf','#aa55ba',
                 '#a546b4','#a037af','#a53a34','#ab453f','#b24f49','#b85a54','#be655f','#c56f69',
                 '#cb7a74','#d1857f','#d88f89','#de9a94','#f8eea2','#eed68c','#e5bd76','#dba560',
                 '#d28c4a','#c87434','#ac632d','#9e5b29','#8f5225','#814a21','#72421d','#643919',
                 '#553115']
    levels_prec = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11.5, 13, 14.5, 16,
                  17.5, 19, 20.5, 22, 23.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50,
                  55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300,
                  325, 350, 375, 400, 500]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Títulos
    ax.text(0.0, 1.02, r'ETA/AMS 8km/0.08°', fontsize=20, fontweight='bold', color='red', transform=ax.transAxes)
    ax.set_title('Total Precipitation (mm)', fontweight='bold', fontsize=16, loc='left')
    ax.set_title(titulo, color='#488f31', fontsize=15, loc='right')
    ax.text(0.50, 1.005, titulo_, color='#1f77b4', fontsize=15, fontweight='normal',
            transform=ax.transAxes, ha='center')
    
    # Extensión Perú
    extent_peru = [-82, -68, -19, 1]
    ax.set_extent(extent_peru, crs=ccrs.PlateCarree())
    
    # Shapefiles
    st0 = None
    st1 = None
    if os.path.exists(SHAPEFILE_COUNTRIES):
        st0 = list(shpreader.Reader(SHAPEFILE_COUNTRIES).geometries())
        ax.add_geometries(st0, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.5)
    if os.path.exists(SHAPEFILE_STATES):
        st1 = list(shpreader.Reader(SHAPEFILE_STATES).geometries())
        ax.add_geometries(st1, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.3)
    
    # Plot precipitación
    plot = ax.contourf(lon, lat, prec, colors=cores_prec, levels=levels_prec, transform=ccrs.PlateCarree())
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3, color="gray")
    gl.top_labels = False
    gl.right_labels = False
    
    # Inset Lima
    ax_inset = ax.inset_axes([0.02, 0.002, 0.34, 0.34], projection=ccrs.PlateCarree())
    extent_lima = [-78.3, -75.3, -13.5, -9.8]
    ax_inset.set_extent(extent_lima, crs=ccrs.PlateCarree())
    if st0:
        ax_inset.add_geometries(st0, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.5)
    if st1:
        ax_inset.add_geometries(st1, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.3)
    ax_inset.contourf(lon, lat, prec, colors=cores_prec, levels=levels_prec, transform=ccrs.PlateCarree())
    ax_inset.set_title("Mapa de Precipitación Total - Lima", fontsize=12, fontweight='bold', color='black', pad=12)
    ax_inset.plot(-77, -12, marker='o', color='black', markersize=8, transform=ccrs.PlateCarree(), zorder=12)
    
    rect = Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                     fill=False, color="black", linewidth=1.5, linestyle='-', zorder=15)
    ax_inset.add_patch(rect)
    
    gl_inset = ax_inset.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl_inset.top_labels = False
    gl_inset.right_labels = False
    gl_inset.left_labels = True
    gl_inset.bottom_labels = True
    gl_inset.xformatter = LongitudeFormatter(degree_symbol="°")
    gl_inset.yformatter = LatitudeFormatter(degree_symbol="°")
    gl_inset.xlabel_style = {'size': 10, 'color': 'black', 'weight': 'bold', 'rotation': 0}
    gl_inset.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold', 'rotation': 90}
    
    # Máximo de Perú - convertir a numpy para cálculos
    lon_vals = lon.values if hasattr(lon, 'values') else lon
    lat_vals = lat.values if hasattr(lat, 'values') else lat
    prec_vals = prec.values if hasattr(prec, 'values') else prec
    
    # Crear grilla 2D de coordenadas (lat es el eje Y, lon es el eje X)
    lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
    
    mask_peru = (
        (lon_2d >= extent_peru[0]) & (lon_2d <= extent_peru[1]) &
        (lat_2d >= extent_peru[2]) & (lat_2d <= extent_peru[3])
    )
    prec_peru = np.where(mask_peru, prec_vals, np.nan)
    idx_max_peru = np.nanargmax(prec_peru)
    y, x = np.unravel_index(idx_max_peru, prec_vals.shape)
    max_val_peru = float(prec_vals[y, x])
    max_lon_peru = float(lon_vals[x])
    max_lat_peru = float(lat_vals[y])
    
    ax.plot(max_lon_peru, max_lat_peru, marker="*", color="black", markersize=14,
            transform=ccrs.PlateCarree(), zorder=20)
    ax.text(max_lon_peru + 0.2, max_lat_peru + 0.2, f"{max_val_peru:.1f}",
            color="black", fontsize=12, fontweight="bold",
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
    
    # Máximo de zona Lima
    mask_lima = (
        (lon_2d >= extent_lima[0]) & (lon_2d <= extent_lima[1]) &
        (lat_2d >= extent_lima[2]) & (lat_2d <= extent_lima[3])
    )
    prec_lima = np.where(mask_lima, prec_vals, np.nan)
    if not np.all(np.isnan(prec_lima)):
        idx_max_lima = np.nanargmax(prec_lima)
        y_lima, x_lima = np.unravel_index(idx_max_lima, prec_vals.shape)
        max_val_lima = float(prec_vals[y_lima, x_lima])
        max_lon_lima = float(lon_vals[x_lima])
        max_lat_lima = float(lat_vals[y_lima])
        
        ax_inset.plot(max_lon_lima, max_lat_lima, marker="*", color="black", markersize=12,
                      transform=ccrs.PlateCarree(), zorder=20)
        ax_inset.text(max_lon_lima + 0.05, max_lat_lima + 0.05, f"{max_val_lima:.1f}",
                      color="black", fontsize=10, fontweight="bold",
                      transform=ccrs.PlateCarree(),
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
    
    # Colorbar
    cbar = fig.colorbar(plot, ax=ax, orientation='vertical', fraction=0.035, pad=0.02)
    cbar.set_label("Precipitación (mm)", fontsize=14)
    
    # Copyright
    rect_x, rect_y, rect_width, rect_height = 0.89, 0.01, 0.1, 0.035
    rectangle = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                  transform=ax.transAxes, color='yellow', zorder=10, linewidth=1)
    ax.add_patch(rectangle)
    ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, "©Bach.Porras",
            transform=ax.transAxes, fontsize=10, fontweight='normal',
            color='blue', ha='center', va='center', zorder=10)
    
    # Guardar imagen
    img_path = "mapa_precipitacion_eta.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return img_path


def actualizar_modelo_eta():
    """Función principal para actualizar el modelo ETA en segundo plano."""
    global eta_data
    
    with eta_lock:
        if eta_data["procesando"]:
            return
        eta_data["procesando"] = True
        eta_data["error"] = None
    
    try:
        print("🌧️ Iniciando actualización del modelo ETA...")
        
        # Descargar shapefiles si no existen
        descargar_shapefiles()
        
        # Descargar datos ETA
        rodada = descargar_datos_eta()
        
        # Procesar precipitación
        ds_soma = procesar_precipitacion_eta()
        
        # Generar mapa
        img_path = generar_mapa_precipitacion(ds_soma)
        
        with eta_lock:
            eta_data["ultima_actualizacion"] = datetime.now().isoformat()
            eta_data["rodada"] = rodada.isoformat() if rodada else None
            eta_data["imagen_path"] = img_path
            eta_data["procesando"] = False
        
        print("✅ Modelo ETA actualizado correctamente")
        
    except Exception as e:
        with eta_lock:
            eta_data["error"] = str(e)
            eta_data["procesando"] = False
        print(f"❌ Error actualizando modelo ETA: {e}")


# ============================================================================
# MODELO WRF - CPTEC/INPE 7km
# ============================================================================

def descargar_datos_wrf():
    """Descarga los datos GRIB2 del modelo WRF del CPTEC/INPE."""
    import requests
    
    agora = datetime.now(timezone.utc)
    nombre_final = "wrf_cptec_6dias.grib2"
    
    # Lista de tentativas, prioridad del más reciente para el más antiguo
    tentativas = []
    
    if agora.hour >= 6:
        tentativas.append(("00", agora))
    
    ontem = agora - timedelta(days=1)
    tentativas.append(("12", ontem))
    tentativas.append(("00", ontem))
    
    for rodada_hora, rodada_data in tentativas:
        rodada_datetime = datetime(
            rodada_data.year, rodada_data.month, rodada_data.day, int(rodada_hora),
            tzinfo=timezone.utc
        )
        
        # Define validade con base en la hora de la rodada
        if rodada_hora == "00":
            validade = rodada_datetime + timedelta(hours=180)
        else:  # "12"
            validade = rodada_datetime + timedelta(hours=168)
        
        url = (
            f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/"
            f"{rodada_data.year}/{rodada_data.strftime('%m')}/{rodada_data.strftime('%d')}/{rodada_hora}/"
            f"WRF_cpt_07KM_{rodada_datetime.strftime('%Y%m%d%H')}_{validade.strftime('%Y%m%d%H')}.grib2"
        )
        
        print(f"📥 Intentando descargar WRF: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            
            if response.status_code == 200:
                if os.path.exists(nombre_final):
                    os.remove(nombre_final)
                
                with open(nombre_final, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✅ WRF descargado: {nombre_final}")
                return rodada_datetime
        except Exception as e:
            print(f"⚠️ Error descargando WRF: {e}")
            continue
    
    raise Exception("No se pudo descargar ninguna rodada WRF disponible")


def procesar_wrf():
    """Procesa los datos WRF y genera el mapa de precipitación."""
    import xarray as xr
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import matplotlib.patches as patches
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Rectangle
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import geopandas as gpd
    from shapely.geometry import Point
    
    # Cargar datos WRF
    ds = xr.open_dataset('wrf_cptec_6dias.grib2', 
                         filter_by_keys={'typeOfLevel': 'surface'}, 
                         decode_timedelta=True)
    prec = ds['tp'][:, :]
    lat = ds['latitude']
    lon = ds['longitude']
    
    # Obtener step para título
    step_val = ds.step.values
    step_horas = int(step_val / 3.6e12)
    titulo_1 = f"F{step_horas:03d}"
    
    # Colores y niveles
    cores_prec = ['#ffffff','#dedede','#bfbfbf','#a1a1a1','#828282','#b7f0be','#a0dcb3',
                 '#88c8a9','#71b49e','#5aa093','#438c88','#2b787e','#146473','#1450b4','#2a61bb',
                 '#3f73c2','#5584c9','#6b96d0','#80a7d6','#96b9dd','#accae4','#c1dceb','#d7edf2',
                 '#cebce0','#c9addb','#c49ed5','#bf90d0','#ba81ca','#b472c5','#af63bf','#aa55ba',
                 '#a546b4','#a037af','#a53a34','#ab453f','#b24f49','#b85a54','#be655f','#c56f69',
                 '#cb7a74','#d1857f','#d88f89','#de9a94','#f8eea2','#eed68c','#e5bd76','#dba560',
                 '#d28c4a','#c87434','#ac632d','#9e5b29','#8f5225','#814a21','#72421d','#643919',
                 '#553115']
    levels_prec = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11.5, 13, 14.5, 16,
                  17.5, 19, 20.5, 22, 23.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50,
                  55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300,
                  325, 350, 375, 400, 500]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Títulos
    ax.text(0.0, 1.025, r'WRF - CPTEC/INPE 7km/0.07°', fontsize=20, fontweight='bold', 
            color='red', transform=ax.transAxes)
    ax.set_title('Precipitación Total (mm)', fontweight='bold', fontsize=16, loc='left')
    ax.set_title(ds.time.dt.strftime('Inic %HUTC %d/%b  ').item() + 
                 ds.valid_time.dt.strftime('Valid %HUTC %d/%b').item(),
                 color='#488f31', fontsize=15, loc='right')
    ax.text(0.60, 1.005, titulo_1, color='#1f77b4', fontsize=15, fontweight='bold', 
            transform=ax.transAxes)
    
    # Extensión Perú
    extent = [-82, -68, -19, 1]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Shapefiles
    st0 = None
    st1 = None
    if os.path.exists(SHAPEFILE_COUNTRIES):
        countries_shp = shpreader.Reader(SHAPEFILE_COUNTRIES)
        for country in countries_shp.records():
            if country.attributes['ADMIN'] == 'Peru':
                ax.add_geometries([country.geometry], ccrs.PlateCarree(), 
                                 facecolor='none', edgecolor='black', linewidth=1.2)
    
    if os.path.exists(SHAPEFILE_STATES):
        states_shp = shpreader.Reader(SHAPEFILE_STATES)
        for state in states_shp.records():
            if state.attributes['admin'] == 'Peru':
                ax.add_geometries([state.geometry], ccrs.PlateCarree(), 
                                 facecolor='none', edgecolor='black', linewidth=0.8)
    
    # Gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', 
                      linewidth=0.25, xlocs=np.arange(-90, -60, 3), ylocs=np.arange(-20, 5, 3), 
                      draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Preparar lon/lat como meshgrid
    lon_vals = prec.longitude.values
    lat_vals = prec.latitude.values
    lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)
    lon_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
    prec_vals = prec.values
    
    # Plot precipitación
    plot = ax.contourf(lon_2d, lat_2d, prec_vals, colors=cores_prec, levels=levels_prec, 
                       transform=ccrs.PlateCarree())
    
    # Máximo de Perú
    if os.path.exists(SHAPEFILE_COUNTRIES):
        gdf_peru = gpd.read_file(SHAPEFILE_COUNTRIES)
        peru_shape = gdf_peru[gdf_peru['ADMIN'] == 'Peru'].geometry.iloc[0]
        
        points_all = [Point(lo, la) for lo, la in zip(lon_2d.flatten(), lat_2d.flatten())]
        mask_peru_shape = np.array([point.within(peru_shape) for point in points_all])
        mask_peru_shape = mask_peru_shape.reshape(prec_vals.shape)
        
        prec_peru = np.where(mask_peru_shape, prec_vals, np.nan)
        
        if np.any(~np.isnan(prec_peru)):
            idx_max_peru = np.nanargmax(prec_peru)
            y_max_peru, x_max_peru = np.unravel_index(idx_max_peru, prec_vals.shape)
            max_val_peru = float(prec_vals[y_max_peru, x_max_peru])
            
            ax.plot(lon_2d[y_max_peru, x_max_peru], lat_2d[y_max_peru, x_max_peru],
                    marker="*", color="black", markersize=14,
                    transform=ccrs.PlateCarree(), zorder=20)
            ax.text(lon_2d[y_max_peru, x_max_peru] + 0.2, lat_2d[y_max_peru, x_max_peru] + 0.2,
                    f"{max_val_peru:.1f}", color="black", fontsize=12, fontweight="bold",
                    transform=ccrs.PlateCarree(),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
            
            # Texto máxima precipitación
            ax.text(0.99, 0.01, f'Máx Precipitación Perú: {max_val_peru:.1f} mm',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
                    path_effects=[path_effects.withStroke(linewidth=1.2, foreground="black")])
    
    # Inset Lima
    ax_inset = ax.inset_axes([0.02, 0.002, 0.34, 0.34], projection=ccrs.PlateCarree())
    extent_lima = [-78.3, -75.3, -13.5, -9.8]
    ax_inset.set_extent(extent_lima, crs=ccrs.PlateCarree())
    
    if os.path.exists(SHAPEFILE_COUNTRIES):
        countries_shp_inset = shpreader.Reader(SHAPEFILE_COUNTRIES)
        for country in countries_shp_inset.records():
            if country.attributes['ADMIN'] == 'Peru':
                ax_inset.add_geometries([country.geometry], ccrs.PlateCarree(),
                                       facecolor='none', edgecolor='black', linewidth=0.6)
    
    if os.path.exists(SHAPEFILE_STATES):
        states_shp_inset = shpreader.Reader(SHAPEFILE_STATES)
        for state in states_shp_inset.records():
            if state.attributes['admin'] == 'Peru':
                ax_inset.add_geometries([state.geometry], ccrs.PlateCarree(),
                                       facecolor='none', edgecolor='black', linewidth=0.4)
    
    ax_inset.contourf(lon_2d, lat_2d, prec_vals, colors=cores_prec, levels=levels_prec,
                      transform=ccrs.PlateCarree())
    
    ax_inset.set_title("Mapa de Precipitación Total - Lima", fontsize=12, fontweight='bold',
                      color='black', pad=12)
    
    # Gridlines del inset
    gl_inset = ax_inset.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                                   alpha=0.7, linestyle='--')
    gl_inset.top_labels = False
    gl_inset.right_labels = False
    gl_inset.xformatter = LongitudeFormatter(degree_symbol="°")
    gl_inset.yformatter = LatitudeFormatter(degree_symbol="°")
    gl_inset.xlabel_style = {'size': 10, 'color': 'black', 'weight': 'bold', 'rotation': 0}
    gl_inset.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold', 'rotation': 90}
    
    # Máximo zona Lima
    if os.path.exists(SHAPEFILE_STATES):
        gdf = gpd.read_file(SHAPEFILE_STATES)
        zonas = gdf[gdf['name'].isin(['Callao', 'Lima', 'Lima Province'])]
        
        if len(zonas) > 0:
            points = [Point(lo, la) for lo, la in zip(lon_2d.flatten(), lat_2d.flatten())]
            mask_combined = np.array([any(point.within(poly) for poly in zonas.geometry) for point in points])
            mask_combined = mask_combined.reshape(prec_vals.shape)
            
            prec_zona = np.where(mask_combined, prec_vals, np.nan)
            if np.any(~np.isnan(prec_zona)):
                idx_max_zona = np.nanargmax(prec_zona)
                y_max_zona, x_max_zona = np.unravel_index(idx_max_zona, prec_vals.shape)
                max_val_zona = float(prec_vals[y_max_zona, x_max_zona])
                
                ax_inset.plot(lon_2d[y_max_zona, x_max_zona], lat_2d[y_max_zona, x_max_zona],
                             marker="*", color="black", markersize=12,
                             transform=ccrs.PlateCarree(), zorder=20)
                ax_inset.text(lon_2d[y_max_zona, x_max_zona] + 0.05, 
                             lat_2d[y_max_zona, x_max_zona] + 0.05,
                             f"{max_val_zona:.1f}", color="black", fontsize=10, fontweight="bold",
                             transform=ccrs.PlateCarree(),
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
    
    # Marcador Lima
    ax_inset.plot(-77, -12, marker='o', color='black', markersize=8,
                  transform=ccrs.PlateCarree(), zorder=12)
    
    rect_inset = Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                           fill=False, color="black", linewidth=1.5, linestyle='-', zorder=15)
    ax_inset.add_patch(rect_inset)
    
    # Colorbar
    fig_height = fig.get_size_inches()[1]
    ax_height = (ax.get_position().y1 - ax.get_position().y0) * fig_height
    fraction = ax_height / fig_height * 0.0612
    cbar = fig.colorbar(plot, ax=ax, orientation='vertical', fraction=fraction, pad=0.01)
    cbar.set_label('Precipitación (mm)', fontsize=14)
    
    selected_ticks = levels_prec[::2]
    cbar.set_ticks(selected_ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in selected_ticks])
    
    # Logo
    rect_x, rect_y, rect_width, rect_height = 0.005, 0.005, 0.065, 0.02
    rectangle = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, transform=ax.transAxes,
                                  color='yellow', zorder=10, linewidth=1)
    ax.add_patch(rectangle)
    ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, "HANAQ-UNALM", transform=ax.transAxes,
            fontsize=8, fontweight='normal', color='blue', ha='center', va='center', zorder=11)
    
    # Guardar
    img_path = "mapa_precipitacion_wrf.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    ds.close()
    return img_path


def actualizar_modelo_wrf():
    """Función principal para actualizar el modelo WRF en segundo plano."""
    global wrf_data
    
    with wrf_lock:
        if wrf_data["procesando"]:
            return
        wrf_data["procesando"] = True
        wrf_data["error"] = None
    
    try:
        print("🌀 Iniciando actualización del modelo WRF...")
        
        # Descargar shapefiles si no existen
        descargar_shapefiles()
        
        # Descargar datos WRF
        rodada = descargar_datos_wrf()
        
        # Procesar y generar mapa
        img_path = procesar_wrf()
        
        with wrf_lock:
            wrf_data["ultima_actualizacion"] = datetime.now().isoformat()
            wrf_data["rodada"] = rodada.isoformat() if rodada else None
            wrf_data["imagen_path"] = img_path
            wrf_data["procesando"] = False
        
        print("✅ Modelo WRF actualizado correctamente")
        
    except Exception as e:
        with wrf_lock:
            wrf_data["error"] = str(e)
            wrf_data["procesando"] = False
        print(f"❌ Error actualizando modelo WRF: {e}")


@app.route('/api/wrf/actualizar', methods=['POST'])
def actualizar_wrf():
    """Endpoint para iniciar la actualización del modelo WRF."""
    thread = threading.Thread(target=actualizar_modelo_wrf)
    thread.start()
    return jsonify({
        "status": "ok",
        "mensaje": "Actualización del modelo WRF iniciada"
    })


@app.route('/api/wrf/estado', methods=['GET'])
def estado_wrf():
    """Endpoint para obtener el estado del modelo WRF."""
    with wrf_lock:
        return jsonify(wrf_data)


@app.route('/api/wrf/mapa', methods=['GET'])
def obtener_mapa_wrf():
    """Endpoint para obtener la imagen del mapa de precipitación WRF."""
    with wrf_lock:
        img_path = wrf_data.get("imagen_path")
    
    if img_path and os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({"error": "Mapa no disponible. Ejecute /api/wrf/actualizar primero."}), 404


@app.route('/api/eta/actualizar', methods=['POST'])
def actualizar_eta():
    """Endpoint para iniciar la actualización del modelo ETA."""
    thread = threading.Thread(target=actualizar_modelo_eta)
    thread.start()
    return jsonify({
        "status": "ok",
        "mensaje": "Actualización del modelo ETA iniciada"
    })


@app.route('/api/eta/estado', methods=['GET'])
def estado_eta():
    """Endpoint para obtener el estado del modelo ETA."""
    with eta_lock:
        return jsonify(eta_data)


@app.route('/api/eta/mapa', methods=['GET'])
def obtener_mapa_eta():
    """Endpoint para obtener la imagen del mapa de precipitación."""
    with eta_lock:
        img_path = eta_data.get("imagen_path")
    
    if img_path and os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({"error": "Mapa no disponible. Ejecute /api/eta/actualizar primero."}), 404


# ============================================================================
# ENDPOINTS DE IMAGEN DEL CIELO
# ============================================================================

@app.route('/api/cielo/subir', methods=['POST'])
def subir_imagen_cielo():
    """Endpoint para recibir imagen del cielo desde la Raspberry Pi."""
    try:
        if 'imagen' not in request.files:
            return jsonify({"error": "No se envió ninguna imagen"}), 400
        
        archivo = request.files['imagen']
        if archivo.filename == '':
            return jsonify({"error": "Nombre de archivo vacío"}), 400
        
        # Guardar imagen original
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"cielo_{timestamp}.jpg"
        ruta_completa = os.path.join(CIELO_FOLDER, nombre_archivo)
        archivo.save(ruta_completa)
        
        with cielo_lock:
            cielo_data["imagen_path"] = ruta_completa
            cielo_data["ultima_actualizacion"] = datetime.now().isoformat()
        
        print(f"📷 Imagen del cielo recibida: {nombre_archivo}")
        
        return jsonify({
            "status": "ok",
            "mensaje": "Imagen recibida correctamente",
            "archivo": nombre_archivo,
            "timestamp": cielo_data["ultima_actualizacion"]
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cielo/imagen', methods=['GET'])
def obtener_imagen_cielo():
    """Obtiene la imagen completa del cielo."""
    with cielo_lock:
        img_path = cielo_data.get("imagen_path")
    
    if img_path and os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return jsonify({"error": "Imagen no disponible"}), 404


@app.route('/api/cielo/cuadrante/<direccion>', methods=['GET'])
def obtener_cuadrante_cielo(direccion):
    """Obtiene un cuadrante específico de la imagen del cielo (norte, sur, este, oeste)."""
    from PIL import Image
    
    with cielo_lock:
        img_path = cielo_data.get("imagen_path")
    
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "Imagen no disponible"}), 404
    
    try:
        img = Image.open(img_path)
        width, height = img.size
        
        # Dividir en 4 cuadrantes
        # Norte = arriba centro, Sur = abajo centro, Este = derecha centro, Oeste = izquierda centro
        half_w = width // 2
        half_h = height // 2
        
        cuadrantes = {
            "norte": (half_w // 2, 0, half_w + half_w // 2, half_h),           # Arriba centro
            "sur": (half_w // 2, half_h, half_w + half_w // 2, height),        # Abajo centro
            "este": (half_w, half_h // 2, width, half_h + half_h // 2),        # Derecha centro
            "oeste": (0, half_h // 2, half_w, half_h + half_h // 2),           # Izquierda centro
            "noreste": (half_w, 0, width, half_h),                              # Arriba derecha
            "noroeste": (0, 0, half_w, half_h),                                 # Arriba izquierda
            "sureste": (half_w, half_h, width, height),                         # Abajo derecha
            "suroeste": (0, half_h, half_w, height)                             # Abajo izquierda
        }
        
        direccion = direccion.lower()
        if direccion not in cuadrantes:
            return jsonify({"error": f"Dirección inválida. Use: {list(cuadrantes.keys())}"}), 400
        
        # Recortar cuadrante
        cuadrante = img.crop(cuadrantes[direccion])
        
        # Guardar en buffer
        buffer = io.BytesIO()
        cuadrante.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/cielo/estado', methods=['GET'])
def estado_cielo():
    """Obtiene el estado de la imagen del cielo."""
    with cielo_lock:
        return jsonify(cielo_data)


@app.route('/api/sensores', methods=['POST'])
def recibir_datos_sensores():
    """
    Endpoint para recibir datos desde la Raspberry Pi.
    
    Ejemplo de JSON esperado:
    {
        "temperatura": 22.5,
        "humedad": 65,
        "presion": 1013.25,
        "viento_velocidad": 12,
        "viento_direccion": 180,
        "precipitacion": 0,
        "luz_uv": 3,
        "calidad_aire": 42
    }
    """
    try:
        datos = request.get_json()
        
        with data_lock:
            sensor_data["temperatura"] = datos.get("temperatura")
            sensor_data["humedad"] = datos.get("humedad")
            sensor_data["presion"] = datos.get("presion")
            sensor_data["viento_velocidad"] = datos.get("viento_velocidad")
            sensor_data["viento_direccion"] = datos.get("viento_direccion")
            sensor_data["precipitacion"] = datos.get("precipitacion")
            sensor_data["luz_uv"] = datos.get("luz_uv")
            sensor_data["calidad_aire"] = datos.get("calidad_aire")
            sensor_data["ultima_actualizacion"] = datetime.now().isoformat()
            
            # Guardar en historial (máximo 288 registros = 24 horas con datos cada 5 min)
            registro = {
                **datos,
                "timestamp": sensor_data["ultima_actualizacion"]
            }
            sensor_data["historial"].append(registro)
            if len(sensor_data["historial"]) > 288:
                sensor_data["historial"].pop(0)
        
        return jsonify({
            "status": "ok",
            "mensaje": "Datos recibidos correctamente",
            "timestamp": sensor_data["ultima_actualizacion"]
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "mensaje": str(e)
        }), 400


@app.route('/api/sensores', methods=['GET'])
def obtener_datos_sensores():
    """Endpoint para obtener los últimos datos de sensores."""
    with data_lock:
        return jsonify(sensor_data)


@app.route('/api/historial', methods=['GET'])
def obtener_historial():
    """Endpoint para obtener el historial de datos."""
    with data_lock:
        return jsonify(sensor_data["historial"])


@app.route('/api/estado', methods=['GET'])
def estado_conexion():
    """Verifica el estado de la conexión con la Raspberry Pi."""
    with data_lock:
        ultima = sensor_data["ultima_actualizacion"]
        if ultima:
            ultimo_tiempo = datetime.fromisoformat(ultima)
            diferencia = (datetime.now() - ultimo_tiempo).total_seconds()
            conectado = diferencia < 60  # Considera desconectado si no hay datos en 60 seg
        else:
            conectado = False
            diferencia = None
    
    return jsonify({
        "raspberry_conectada": conectado,
        "ultima_actualizacion": ultima,
        "segundos_desde_ultimo_dato": diferencia
    })

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HANAQ - UNALM Dashboard</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/lucide/0.263.1/lucide.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Outfit:wght@300;400;500;600&display=swap');
    
    * {
      font-family: 'Outfit', sans-serif;
    }
    
    .title-font {
      font-family: 'Orbitron', monospace;
    }
    
    .glass-card {
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .glass-card-dark {
      background: rgba(0, 0, 0, 0.2);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .glow {
      box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
    }
    
    .animate-fade-in {
      animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-slide-in {
      animation: slideIn 0.8s ease-out;
    }
    
    @keyframes slideIn {
      from { opacity: 0; transform: translateX(-30px); }
      to { opacity: 1; transform: translateX(0); }
    }
    
    .webcam-container {
      position: relative;
      overflow: hidden;
    }
    
    .webcam-container::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.3) 100%);
      pointer-events: none;
      z-index: 1;
    }
    
    .scan-line {
      position: absolute;
      width: 100%;
      height: 2px;
      background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.8), transparent);
      animation: scan 3s linear infinite;
      z-index: 2;
    }
    
    @keyframes scan {
      0% { top: 0; }
      100% { top: 100%; }
    }
    
    .weather-icon {
      filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.6));
    }
    
    .metric-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 12px 40px rgba(59, 130, 246, 0.3);
      transition: all 0.3s ease;
    }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useState, useEffect } = React;

    // Iconos lucide simplificados como componentes
    const MapPin = () => <span>📍</span>;
    const Camera = () => <span>📷</span>;
    const Clock = () => <span>⏰</span>;
    const Calendar = () => <span>📅</span>;
    const Droplets = () => <span>💧</span>;
    const Wind = () => <span>💨</span>;
    const Eye = () => <span>👁️</span>;
    const CloudRain = () => <span>🌧️</span>;
    const AlertTriangle = () => <span>⚠️</span>;
    const Wifi = () => <span>📡</span>;
    const WifiOff = () => <span>📵</span>;
    const Thermometer = () => <span>🌡️</span>;
    const Gauge = () => <span>🎛️</span>;
    const Sun = () => <span>☀️</span>;
    const Leaf = () => <span>🍃</span>;

    const WeatherDashboard = () => {
      const [sensorData, setSensorData] = useState(null);
      const [raspberryStatus, setRaspberryStatus] = useState({ conectada: false });
      const [historial, setHistorial] = useState([]);
      const [hourlyForecast, setHourlyForecast] = useState([]);
      const [weeklyForecast, setWeeklyForecast] = useState([]);
      const [alerts, setAlerts] = useState([]);
      const [loading, setLoading] = useState(true);
      const [selectedWebcam] = useState(0);
      const [etaStatus, setEtaStatus] = useState({ procesando: false, imagen_path: null });
      const [etaImageUrl, setEtaImageUrl] = useState(null);
      const [wrfStatus, setWrfStatus] = useState({ procesando: false, imagen_path: null });
      const [wrfImageUrl, setWrfImageUrl] = useState(null);
      const [activeTab, setActiveTab] = useState('sensores'); // 'sensores', 'eta' o 'wrf'

      // Webcam del Campus UNALM - La Molina
      const webcams = [
        {
          name: "Campus UNALM",
          url: "https://images.webcams.travel/webcam/1366626800.jpg",
          location: "La Molina, Lima"
        }
      ];

      useEffect(() => {
        fetchSensorData();
        fetchWeatherForecast();
        fetchEtaStatus();
        fetchWrfStatus();
        // Actualizar datos de sensores cada 5 segundos
        const sensorInterval = setInterval(fetchSensorData, 5000);
        // Actualizar pronóstico cada 5 minutos
        const forecastInterval = setInterval(fetchWeatherForecast, 300000);
        // Actualizar estado ETA cada 10 segundos
        const etaInterval = setInterval(fetchEtaStatus, 10000);
        // Actualizar estado WRF cada 10 segundos
        const wrfInterval = setInterval(fetchWrfStatus, 10000);
        return () => {
          clearInterval(sensorInterval);
          clearInterval(forecastInterval);
          clearInterval(etaInterval);
          clearInterval(wrfInterval);
        };
      }, []);

      // Obtener estado del modelo ETA
      const fetchEtaStatus = async () => {
        try {
          const res = await fetch('/api/eta/estado');
          const data = await res.json();
          setEtaStatus(data);
          if (data.imagen_path) {
            setEtaImageUrl(`/api/eta/mapa?t=${Date.now()}`);
          }
        } catch (error) {
          console.error('Error fetching ETA status:', error);
        }
      };

      // Actualizar modelo ETA
      const actualizarModeloEta = async () => {
        try {
          await fetch('/api/eta/actualizar', { method: 'POST' });
          setEtaStatus({ ...etaStatus, procesando: true });
        } catch (error) {
          console.error('Error updating ETA model:', error);
        }
      };

      // Obtener estado del modelo WRF
      const fetchWrfStatus = async () => {
        try {
          const res = await fetch('/api/wrf/estado');
          const data = await res.json();
          setWrfStatus(data);
          if (data.imagen_path) {
            setWrfImageUrl(`/api/wrf/mapa?t=${Date.now()}`);
          }
        } catch (error) {
          console.error('Error fetching WRF status:', error);
        }
      };

      // Actualizar modelo WRF
      const actualizarModeloWrf = async () => {
        try {
          await fetch('/api/wrf/actualizar', { method: 'POST' });
          setWrfStatus({ ...wrfStatus, procesando: true });
        } catch (error) {
          console.error('Error updating WRF model:', error);
        }
      };

      // Obtener datos de sensores de la Raspberry Pi
      const fetchSensorData = async () => {
        try {
          // Obtener estado de conexión
          const statusRes = await fetch('/api/estado');
          const status = await statusRes.json();
          setRaspberryStatus({
            conectada: status.raspberry_conectada,
            ultimaActualizacion: status.ultima_actualizacion,
            segundosDesdeUltimo: status.segundos_desde_ultimo_dato
          });

          // Obtener datos de sensores
          const sensorRes = await fetch('/api/sensores');
          const data = await sensorRes.json();
          
          if (data.temperatura !== null) {
            setSensorData({
              temp: Math.round(data.temperatura),
              humidity: data.humedad,
              pressure: data.presion,
              windSpeed: data.viento_velocidad,
              windDirection: data.viento_direccion,
              precipitation: data.precipitacion,
              uvIndex: data.luz_uv,
              airQuality: data.calidad_aire,
              time: data.ultima_actualizacion ? new Date(data.ultima_actualizacion) : new Date()
            });
            setHistorial(data.historial || []);
          }
          
          // Generar alertas basadas en datos de sensores
          const newAlerts = [];
          if (data.viento_velocidad > 40) {
            newAlerts.push({
              type: 'warning',
              title: 'Vientos Fuertes',
              message: `Vientos de ${data.viento_velocidad} km/h detectados por sensores.`
            });
          }
          if (data.temperatura > 35) {
            newAlerts.push({
              type: 'warning',
              title: 'Temperatura Alta',
              message: `Temperatura de ${data.temperatura}°C. Manténgase hidratado.`
            });
          }
          if (data.luz_uv > 8) {
            newAlerts.push({
              type: 'warning',
              title: 'Índice UV Alto',
              message: `Índice UV de ${data.luz_uv}. Use protección solar.`
            });
          }
          if (data.calidad_aire > 100) {
            newAlerts.push({
              type: 'warning',
              title: 'Calidad de Aire',
              message: `AQI de ${data.calidad_aire}. Considere limitar actividades al aire libre.`
            });
          }
          setAlerts(newAlerts);
          
          setLoading(false);
        } catch (error) {
          console.error('Error fetching sensor data:', error);
          setLoading(false);
        }
      };

      // Obtener pronóstico de API externa (Open-Meteo)
      const fetchWeatherForecast = async () => {
        try {
          const lat = -12.0828;
          const lon = -76.9472;

          const response = await fetch(
            `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m,precipitation_probability,weathercode,windspeed_10m&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max&timezone=America/Lima`
          );

          const data = await response.json();

          const hourly = [];
          for (let i = 0; i < 24; i++) {
            hourly.push({
              time: new Date(data.hourly.time[i]),
              temp: Math.round(data.hourly.temperature_2m[i]),
              precipitation: data.hourly.precipitation_probability[i],
              weatherCode: data.hourly.weathercode[i],
              windSpeed: Math.round(data.hourly.windspeed_10m[i])
            });
          }
          setHourlyForecast(hourly);

          const weekly = [];
          for (let i = 0; i < 7; i++) {
            weekly.push({
              date: new Date(data.daily.time[i]),
              maxTemp: Math.round(data.daily.temperature_2m_max[i]),
              minTemp: Math.round(data.daily.temperature_2m_min[i]),
              precipitation: data.daily.precipitation_probability_max[i],
              weatherCode: data.daily.weathercode[i],
              windSpeed: Math.round(data.daily.windspeed_10m_max[i])
            });
          }
          setWeeklyForecast(weekly);
        } catch (error) {
          console.error('Error fetching forecast:', error);
        }
      };

      const getWeatherEmoji = (code) => {
        if (code === 0) return '☀️';
        if (code === 1) return '🌤️';
        if (code === 2) return '⛅';
        if (code === 3) return '☁️';
        if (code >= 45 && code <= 48) return '🌫️';
        if (code >= 51 && code <= 67) return '🌧️';
        if (code >= 71 && code <= 77) return '🌨️';
        if (code >= 80 && code <= 82) return '⛈️';
        if (code >= 85 && code <= 86) return '🌨️';
        if (code >= 95 && code <= 99) return '⛈️';
        return '☁️';
      };

      const getWeatherDescription = (code) => {
        if (code === 0) return 'Despejado';
        if (code <= 3) return 'Parcialmente nublado';
        if (code <= 48) return 'Nublado';
        if (code <= 67) return 'Lluvia';
        if (code >= 80) return 'Chubascos';
        return 'Variado';
      };

      const getWindDirection = (degrees) => {
        const directions = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'];
        return directions[Math.round(degrees / 45) % 8];
      };

      const formatTime = (date) => {
        return date.toLocaleTimeString('es-PE', { hour: '2-digit', minute: '2-digit' });
      };

      const formatDate = (date) => {
        return date.toLocaleDateString('es-PE', { weekday: 'short', day: 'numeric', month: 'short' });
      };

      if (loading) {
        return (
          <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
            <div className="text-white text-2xl font-light animate-pulse">Cargando datos meteorológicos...</div>
          </div>
        );
      }

      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6 text-white">
          {/* Header */}
          <div className="max-w-7xl mx-auto mb-8 animate-fade-in">
            <div className="flex items-center justify-between mb-2">
              <h1 className="text-5xl font-black title-font bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                HANAQ - UNALM
              </h1>
              <div className="flex items-center gap-4">
                {/* Estado Raspberry Pi */}
                <div className={`flex items-center gap-2 glass-card px-4 py-2 rounded-full ${raspberryStatus.conectada ? 'border-green-500' : 'border-red-500'} border`}>
                  {raspberryStatus.conectada ? <Wifi /> : <WifiOff />}
                  <span className={`font-medium ${raspberryStatus.conectada ? 'text-green-400' : 'text-red-400'}`}>
                    {raspberryStatus.conectada ? 'Raspberry Pi Conectada' : 'Raspberry Pi Desconectada'}
                  </span>
                </div>
                <div className="flex items-center gap-2 glass-card px-4 py-2 rounded-full">
                  <MapPin />
                  <span className="font-medium">Campus UNALM, La Molina</span>
                </div>
              </div>
            </div>
            <p className="text-blue-200 text-sm font-light">
              {sensorData ? `Datos de sensores: ${sensorData.time?.toLocaleString('es-PE')}` : 'Esperando datos de Raspberry Pi...'}
            </p>
          </div>

          {/* Alertas */}
          {alerts.length > 0 && (
            <div className="max-w-7xl mx-auto mb-6 animate-slide-in">
              {alerts.map((alert, idx) => (
                <div key={idx} className="glass-card rounded-2xl p-4 mb-3 border-l-4 border-yellow-400">
                  <div className="flex items-start gap-3">
                    <AlertTriangle />
                    <div>
                      <h3 className="font-semibold text-lg text-yellow-400">{alert.title}</h3>
                      <p className="text-gray-200">{alert.message}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Panel principal izquierdo */}
            <div className="lg:col-span-2 space-y-6">
              
              {/* Pestañas de navegación */}
              <div className="flex gap-2 mb-4 flex-wrap">
                <button 
                  onClick={() => setActiveTab('sensores')}
                  className={`px-6 py-3 rounded-xl font-semibold transition-all ${
                    activeTab === 'sensores' 
                      ? 'bg-blue-600 text-white glow' 
                      : 'glass-card-dark text-gray-300 hover:bg-white/10'
                  }`}
                >
                  📡 Sensores en Vivo
                </button>
                <button 
                  onClick={() => setActiveTab('eta')}
                  className={`px-6 py-3 rounded-xl font-semibold transition-all ${
                    activeTab === 'eta' 
                      ? 'bg-green-600 text-white glow' 
                      : 'glass-card-dark text-gray-300 hover:bg-white/10'
                  }`}
                >
                  🌧️ Modelo ETA 8km
                </button>
                <button 
                  onClick={() => setActiveTab('wrf')}
                  className={`px-6 py-3 rounded-xl font-semibold transition-all ${
                    activeTab === 'wrf' 
                      ? 'bg-purple-600 text-white glow' 
                      : 'glass-card-dark text-gray-300 hover:bg-white/10'
                  }`}
                >
                  🌀 Modelo WRF 7km
                </button>
              </div>

              {/* Contenido de Sensores */}
              {activeTab === 'sensores' && (
                <>
                  {/* Vista del Cielo 360° con zonas horizontales */}
                  <div className="glass-card rounded-3xl p-6 animate-fade-in">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <Camera />
                        <h2 className="text-2xl font-bold title-font">CIELO 360° EN TIEMPO REAL</h2>
                      </div>
                      <div className="flex items-center gap-2 glass-card-dark px-3 py-1 rounded-full">
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-xs font-medium">EN VIVO</span>
                      </div>
                    </div>
                    
                    {/* Imagen panorámica 360° con 4 zonas horizontales */}
                    <div className="relative rounded-2xl overflow-hidden mb-4">
                      <img 
                        src={`/api/cielo/imagen?t=${Date.now()}`}
                        alt="Vista del cielo 360°"
                        className="w-full h-64 object-cover"
                        onError={(e) => {
                          e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="800" height="200"%3E%3Crect fill="%23334155" width="800" height="200"/%3E%3Ctext fill="%23cbd5e1" font-family="Arial" font-size="20" x="50%25" y="50%25" text-anchor="middle" dominant-baseline="middle"%3ECámara no disponible%3C/text%3E%3C/svg%3E';
                        }}
                      />
                      
                      {/* Líneas divisorias verticales (3 líneas = 4 zonas) */}
                      <div className="absolute top-0 left-[12.5%] w-0.5 h-full bg-yellow-400/70"></div>
                      <div className="absolute top-0 left-[37.5%] w-0.5 h-full bg-yellow-400/70"></div>
                      <div className="absolute top-0 left-[62.5%] w-0.5 h-full bg-yellow-400/70"></div>
                      <div className="absolute top-0 left-[87.5%] w-0.5 h-full bg-yellow-400/70"></div>
                      
                      {/* Etiquetas en los puntos cardinales exactos */}
                      {/* OESTE(0%) → NORTE(25%) → ESTE(50%) → SUR(75%) → OESTE(100%) */}
                      <div className="absolute top-2 left-[25%] transform -translate-x-1/2 bg-blue-600/80 px-3 py-1 rounded-lg border border-white/50">
                        <span className="text-white font-bold text-sm">⬆️ NORTE</span>
                      </div>
                      <div className="absolute top-2 left-[50%] transform -translate-x-1/2 bg-green-600/80 px-3 py-1 rounded-lg border border-white/50">
                        <span className="text-white font-bold text-sm">☀️ ESTE</span>
                      </div>
                      <div className="absolute top-2 left-[75%] transform -translate-x-1/2 bg-red-600/80 px-3 py-1 rounded-lg border border-white/50">
                        <span className="text-white font-bold text-sm">⬇️ SUR</span>
                      </div>
                      
                      {/* Indicadores de extremos OESTE */}
                      <div className="absolute top-2 left-2 bg-purple-600/80 px-3 py-1 rounded-lg border border-white/50">
                        <span className="text-white font-bold text-sm">⬅️ OESTE</span>
                      </div>
                      <div className="absolute top-2 right-2 bg-purple-600/80 px-3 py-1 rounded-lg border border-white/50">
                        <span className="text-white font-bold text-sm">OESTE ➡️</span>
                      </div>
                      
                      {/* Brújula en centro inferior */}
                      <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 bg-black/60 p-2 rounded-full border border-yellow-400">
                        <span className="text-2xl">🧭</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="text-sm font-medium text-blue-200">
                        📍 Estación HANAQ - UNALM, La Molina
                      </div>
                      <div className="text-xs text-gray-400">
                        Imagen panorámica 360° • Actualización automática
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* Contenido del Modelo ETA */}
              {activeTab === 'eta' && (
                <div className="glass-card rounded-3xl p-6 animate-fade-in">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <CloudRain />
                      <h2 className="text-2xl font-bold title-font">MODELO ETA - PRECIPITACIÓN</h2>
                    </div>
                    <button 
                      onClick={actualizarModeloEta}
                      disabled={etaStatus.procesando}
                      className={`px-4 py-2 rounded-xl font-semibold transition-all ${
                        etaStatus.procesando 
                          ? 'bg-gray-600 cursor-not-allowed' 
                          : 'bg-green-600 hover:bg-green-700'
                      }`}
                    >
                      {etaStatus.procesando ? '⏳ Procesando...' : '🔄 Actualizar Modelo'}
                    </button>
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-sm text-blue-200 mb-2">
                      Datos del CPTEC/INPE - Modelo ETA 8km para Perú
                    </p>
                    {etaStatus.ultima_actualizacion && (
                      <p className="text-xs text-gray-400">
                        Última actualización: {new Date(etaStatus.ultima_actualizacion).toLocaleString('es-PE')}
                      </p>
                    )}
                    {etaStatus.error && (
                      <p className="text-xs text-red-400 mt-2">
                        Error: {etaStatus.error}
                      </p>
                    )}
                  </div>
                  
                  <div className="rounded-2xl overflow-hidden bg-slate-800">
                    {etaImageUrl ? (
                      <img 
                        src={etaImageUrl}
                        alt="Mapa de precipitación ETA"
                        className="w-full h-auto"
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    ) : (
                      <div className="h-96 flex items-center justify-center text-gray-400">
                        <div className="text-center">
                          <CloudRain />
                          <p className="mt-4">Haga clic en "Actualizar Modelo" para generar el mapa</p>
                          <p className="text-xs mt-2 text-gray-500">Primera ejecución puede tardar varios minutos</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Contenido del Modelo WRF */}
              {activeTab === 'wrf' && (
                <div className="glass-card rounded-3xl p-6 animate-fade-in">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span>🌀</span>
                      <h2 className="text-2xl font-bold title-font">MODELO WRF - PRECIPITACIÓN</h2>
                    </div>
                    <button 
                      onClick={actualizarModeloWrf}
                      disabled={wrfStatus.procesando}
                      className={`px-4 py-2 rounded-xl font-semibold transition-all ${
                        wrfStatus.procesando 
                          ? 'bg-gray-600 cursor-not-allowed' 
                          : 'bg-purple-600 hover:bg-purple-700'
                      }`}
                    >
                      {wrfStatus.procesando ? '⏳ Procesando...' : '🔄 Actualizar Modelo'}
                    </button>
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-sm text-purple-200 mb-2">
                      Datos del CPTEC/INPE - Modelo WRF 7km para Perú (6 días de pronóstico)
                    </p>
                    {wrfStatus.ultima_actualizacion && (
                      <p className="text-xs text-gray-400">
                        Última actualización: {new Date(wrfStatus.ultima_actualizacion).toLocaleString('es-PE')}
                      </p>
                    )}
                    {wrfStatus.error && (
                      <p className="text-xs text-red-400 mt-2">
                        Error: {wrfStatus.error}
                      </p>
                    )}
                  </div>
                  
                  <div className="rounded-2xl overflow-hidden bg-slate-800">
                    {wrfImageUrl ? (
                      <img 
                        src={wrfImageUrl}
                        alt="Mapa de precipitación WRF"
                        className="w-full h-auto"
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    ) : (
                      <div className="h-96 flex items-center justify-center text-gray-400">
                        <div className="text-center">
                          <span className="text-6xl">🌀</span>
                          <p className="mt-4">Haga clic en "Actualizar Modelo" para generar el mapa</p>
                          <p className="text-xs mt-2 text-gray-500">Primera ejecución puede tardar varios minutos (descarga ~150MB)</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Pronóstico por Horas */}
              <div className="glass-card rounded-3xl p-6 animate-fade-in">
                <div className="flex items-center gap-3 mb-4">
                  <Clock />
                  <h2 className="text-2xl font-bold title-font">PRÓXIMAS 24 HORAS</h2>
                </div>
                
                <div className="overflow-x-auto">
                  <div className="flex gap-4 pb-2">
                    {hourlyForecast.slice(0, 12).map((hour, idx) => (
                      <div key={idx} className="glass-card-dark rounded-2xl p-4 min-w-[100px] text-center metric-card">
                        <div className="text-sm font-medium text-blue-300 mb-2">
                          {formatTime(hour.time)}
                        </div>
                        <div className="text-5xl mb-2">
                          {getWeatherEmoji(hour.weatherCode)}
                        </div>
                        <div className="text-2xl font-bold mb-1">{hour.temp}°</div>
                        <div className="flex items-center justify-center gap-1 text-xs text-blue-300">
                          <Droplets />
                          <span>{hour.precipitation}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Pronóstico Semanal */}
              <div className="glass-card rounded-3xl p-6 animate-fade-in">
                <div className="flex items-center gap-3 mb-4">
                  <Calendar />
                  <h2 className="text-2xl font-bold title-font">PRÓXIMOS 7 DÍAS</h2>
                </div>
                
                <div className="space-y-3">
                  {weeklyForecast.map((day, idx) => (
                    <div key={idx} className="glass-card-dark rounded-xl p-4 flex items-center justify-between hover:bg-white/10 transition-all">
                      <div className="flex items-center gap-4 flex-1">
                        <div className="w-16 font-semibold text-blue-300">
                          {formatDate(day.date)}
                        </div>
                        <div className="text-5xl">
                          {getWeatherEmoji(day.weatherCode)}
                        </div>
                        <div className="flex-1 text-sm text-gray-300">
                          {getWeatherDescription(day.weatherCode)}
                        </div>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                          <Droplets />
                          <span className="text-sm">{day.precipitation}%</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Wind />
                          <span className="text-sm">{day.windSpeed} km/h</span>
                        </div>
                        <div className="text-right min-w-[80px]">
                          <span className="text-xl font-bold text-orange-400">{day.maxTemp}°</span>
                          <span className="text-gray-400 mx-1">/</span>
                          <span className="text-lg text-blue-300">{day.minTemp}°</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Panel derecho */}
            <div className="space-y-6">
              {/* Temperatura actual de sensores */}
              <div className="glass-card rounded-3xl p-8 animate-fade-in glow">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold title-font text-cyan-400">SENSORES EN VIVO</h2>
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${raspberryStatus.conectada ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                    <Thermometer />
                  </div>
                </div>
                
                <div className="text-center mb-6">
                  <div className="text-8xl font-black title-font bg-gradient-to-b from-white to-blue-200 bg-clip-text text-transparent mb-2">
                    {sensorData?.temp ?? '--'}°
                  </div>
                  <div className="text-xl text-blue-200 font-light mb-1">
                    Temperatura Real
                  </div>
                  <div className="text-sm text-gray-400">
                    Datos en tiempo real desde Raspberry Pi
                  </div>
                </div>
              </div>

              {/* Métricas de sensores */}
              <div className="grid grid-cols-2 gap-4">
                <div className="glass-card-dark rounded-2xl p-5 metric-card">
                  <div className="flex items-center gap-2 mb-3 text-blue-400">
                    <Droplets />
                    <span className="text-xs font-semibold uppercase tracking-wide">Humedad</span>
                  </div>
                  <div className="text-3xl font-bold">{sensorData?.humidity ?? '--'}%</div>
                </div>

                <div className="glass-card-dark rounded-2xl p-5 metric-card">
                  <div className="flex items-center gap-2 mb-3 text-cyan-400">
                    <Wind />
                    <span className="text-xs font-semibold uppercase tracking-wide">Viento</span>
                  </div>
                  <div className="text-3xl font-bold">{sensorData?.windSpeed ?? '--'}</div>
                  <div className="text-xs text-gray-400">km/h {sensorData?.windDirection ? getWindDirection(sensorData.windDirection) : ''}</div>
                </div>

                <div className="glass-card-dark rounded-2xl p-5 metric-card">
                  <div className="flex items-center gap-2 mb-3 text-purple-400">
                    <Gauge />
                    <span className="text-xs font-semibold uppercase tracking-wide">Presión</span>
                  </div>
                  <div className="text-3xl font-bold">{sensorData?.pressure ?? '--'}</div>
                  <div className="text-xs text-gray-400">hPa</div>
                </div>

                <div className="glass-card-dark rounded-2xl p-5 metric-card">
                  <div className="flex items-center gap-2 mb-3 text-orange-400">
                    <CloudRain />
                    <span className="text-xs font-semibold uppercase tracking-wide">Precipitación</span>
                  </div>
                  <div className="text-3xl font-bold">{sensorData?.precipitation ?? '--'}</div>
                  <div className="text-xs text-gray-400">mm</div>
                </div>

                <div className="glass-card-dark rounded-2xl p-5 metric-card">
                  <div className="flex items-center gap-2 mb-3 text-yellow-400">
                    <Sun />
                    <span className="text-xs font-semibold uppercase tracking-wide">Índice UV</span>
                  </div>
                  <div className="text-3xl font-bold">{sensorData?.uvIndex ?? '--'}</div>
                </div>

                <div className="glass-card-dark rounded-2xl p-5 metric-card">
                  <div className="flex items-center gap-2 mb-3 text-green-400">
                    <Leaf />
                    <span className="text-xs font-semibold uppercase tracking-wide">Calidad Aire</span>
                  </div>
                  <div className="text-3xl font-bold">{sensorData?.airQuality ?? '--'}</div>
                  <div className="text-xs text-gray-400">AQI</div>
                </div>
              </div>

              {/* Info adicional */}
              <div className="glass-card rounded-2xl p-6">
                <h3 className="font-bold text-lg mb-4 title-font text-cyan-400">INFORMACIÓN</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Sensores:</span>
                    <span className={`font-medium ${raspberryStatus.conectada ? 'text-green-400' : 'text-red-400'}`}>
                      {raspberryStatus.conectada ? 'Raspberry Pi' : 'Desconectado'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Pronóstico:</span>
                    <span className="font-medium">Open-Meteo API</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Actualización sensores:</span>
                    <span className="font-medium">Cada 5 segundos</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Registros históricos:</span>
                    <span className="font-medium">{historial.length}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    };

    ReactDOM.render(<WeatherDashboard />, document.getElementById('root'));
  </script>
</body>
</html>'''


@app.route('/')
def pronostico():
    """Ruta principal que muestra el dashboard de pronóstico del tiempo."""
    return render_template_string(HTML_TEMPLATE)


if __name__ == '__main__':
    print("=" * 60)
    print("🌤️  HANAQ - UNALM Dashboard")
    print("=" * 60)
    print("📍 Servidor iniciado en: http://127.0.0.1:5000")
    print("")
    print("📡 Endpoints API para Raspberry Pi:")
    print("   POST /api/sensores  - Enviar datos de sensores")
    print("   GET  /api/sensores  - Obtener últimos datos")
    print("   GET  /api/historial - Obtener historial de datos")
    print("   GET  /api/estado    - Verificar conexión")
    print("")
    print("📋 Ejemplo de envío desde Raspberry Pi:")
    print('''   curl -X POST http://<IP_SERVIDOR>:5000/api/sensores \\
        -H "Content-Type: application/json" \\
        -d '{"temperatura": 22.5, "humedad": 65, "presion": 1013.25, 
             "viento_velocidad": 12, "viento_direccion": 180,
             "precipitacion": 0, "luz_uv": 3, "calidad_aire": 42}'
''')
    print("=" * 60)
    print("Presiona Ctrl+C para detener el servidor")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)


# ============================================================================
# EJEMPLO DE CÓDIGO PARA RASPBERRY PI (guardar como raspberry_sensores.py)
# ============================================================================
"""
Ejemplo de script para enviar datos desde Raspberry Pi:

import requests
import time
import random  # Reemplazar con lecturas reales de sensores

SERVIDOR_URL = "http://<IP_DEL_SERVIDOR>:5000/api/sensores"

# Aquí importarías las librerías de tus sensores, por ejemplo:
# import Adafruit_DHT  # Para sensor DHT11/DHT22
# import board
# import adafruit_bmp280  # Para sensor BMP280
# import adafruit_ahtx0  # Para sensor AHT20

def leer_sensores():
    '''
    Lee los valores de los sensores conectados a la Raspberry Pi.
    Reemplazar con las lecturas reales de tus sensores.
    '''
    # Ejemplo con valores simulados - REEMPLAZAR con lecturas reales
    datos = {
        "temperatura": round(random.uniform(18, 28), 1),
        "humedad": round(random.uniform(50, 80), 1),
        "presion": round(random.uniform(1010, 1020), 2),
        "viento_velocidad": round(random.uniform(0, 25), 1),
        "viento_direccion": random.randint(0, 360),
        "precipitacion": round(random.uniform(0, 5), 2),
        "luz_uv": round(random.uniform(0, 11), 1),
        "calidad_aire": random.randint(20, 80)
    }
    return datos

def enviar_datos():
    '''Envía los datos de sensores al servidor Flask.'''
    try:
        datos = leer_sensores()
        response = requests.post(
            SERVIDOR_URL,
            json=datos,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            print(f"✓ Datos enviados: Temp={datos['temperatura']}°C, Hum={datos['humedad']}%")
        else:
            print(f"✗ Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Error de conexión: {e}")

if __name__ == "__main__":
    print("🍓 Raspberry Pi - Envío de datos de sensores")
    print(f"📡 Servidor: {SERVIDOR_URL}")
    while True:
        enviar_datos()
        time.sleep(5)  # Enviar cada 5 segundos
"""
