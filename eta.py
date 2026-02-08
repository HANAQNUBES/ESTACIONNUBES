import os
import subprocess
import sys  # Adicionado o import do sys

def setup_environment():
    # Baixar o arquivo ZIP do Dropbox
    download_url = "https://www.dropbox.com/scl/fo/5mbf2xq1tkbv3uqaahypf/AJg0sUw1ufTBQFSx-OuQTxw?rlkey=6c9py2vogdcha59r7dc8r2hes&st=wu8ihemy&dl=1"
    zip_filename = "dropbox_folder.zip"

    # Usar wget para baixar o arquivo
    os.system(f"wget -O {zip_filename} \"{download_url}\"")

    # Descompactar o arquivo ZIP
    os.system(f"unzip {zip_filename}")

    # Instalar as dependências necessárias
    dependencies = ["xarray", "cfgrib", "matplotlib", "cartopy", "scipy", "numpy", "ecmwf.opendata==0.3.19"]

    for dep in dependencies:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

# Chamar a função para configurar o ambiente
setup_environment()

import os
import urllib.request
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import time

# 📁 Pasta de saída
PASTA_SAIDA = "dados_eta_8km"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# 🔁 Detecta rodada mais recente (00 ou 12 UTC)
# Adiciona um atraso para garantir que a rodada esteja disponível
atraso_publicacao = timedelta(hours=4)
agora = datetime.now(timezone.utc) - atraso_publicacao

hora_rodada = "00"
data_base = agora.replace(hour=int(hora_rodada), minute=0, second=0, microsecond=0)

print(f"📅 Rodada detectada: {data_base.strftime('%Y-%m-%d %H:%M')} UTC")

# 🧾 Gera lista de arquivos GRIB2
arquivos = []
for h in range (5):
    data_prev = data_base + timedelta(hours=h)
    nome = f"Eta_ams_08km_{data_base.strftime('%Y%m%d%H')}_{data_prev.strftime('%Y%m%d%H')}.grib2"
    url = f"https://dataserver.cptec.inpe.br/dataserver_modelos/eta/ams_08km/brutos/{data_base.strftime('%Y/%m/%d/%H')}/{nome}"
    arquivos.append((nome, url))

# ⬇️ Baixa os arquivos GRIB2 com até 3 tentativas
for nome, url in tqdm(arquivos, desc="📥 Baixando arquivos"):
    caminho = os.path.join(PASTA_SAIDA, nome)
    if os.path.exists(caminho):
        print(f"✔️ Já existe: {nome}")
        continue

    sucesso = False
    for tentativa in range(3):
        try:
            urllib.request.urlretrieve(url, caminho)
            sucesso = True
            break
        except Exception as e:
            print(f"⚠️ Tentativa {tentativa+1}/3 falhou para {nome}: {e}")
            time.sleep(5)  # Espera 5 segundos antes da próxima tentativa

    if not sucesso:
        print(f"❌ Falha ao baixar {nome} após 3 tentativas.")


import os
import cfgrib
import xarray as xr

PASTA_SAIDA = "dados_eta_8km"

soma_unknown = None
lat = lon = None

# Inicializa variáveis de tempo
rodada_inicial = rodada_final = None
previsao_inicial = previsao_final = None

# Ordena arquivos e ignora o primeiro
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
            print(f"⚠️ Variável 'unknown' não encontrada em {arq}")
            continue

        var = ds['unknown']

        if soma_unknown is None:
            soma_unknown = var.copy(deep=True)
        else:
            soma_unknown += var

        if lat is None or lon is None:
            lat = ds.latitude
            lon = ds.longitude

        rodada = ds.time.values
        previsao = ds.valid_time.values

        # Salva tempo da primeira e última previsão
        if i == 0:
            rodada_inicial = rodada
            previsao_inicial = previsao
        if i == len(arquivos) - 1:
            rodada_final = rodada
            previsao_final = previsao

    except Exception as e:
        print(f"⚠️ Erro ao processar {arq}: {e}")

# Cria novo Dataset
ds_soma = xr.Dataset(
    {
        "precipitacao_total": soma_unknown
    },
    coords={
        "latitude": lat,
        "longitude": lon
    },
    attrs={
        "descricao": "Soma da variável 'unknown' dos arquivos ETA (possivelmente precipitação acumulada)",
        "rodada_inicial": str(rodada_inicial),
        "rodada_final": str(rodada_final),
        "previsao_inicial": str(previsao_inicial),
        "previsao_final": str(previsao_final)
    }
)

# Salva NetCDF
saida_path = "acum_prec.nc"
ds_soma.to_netcdf(saida_path)
print(f"\n✅ Arquivo salvo em: {saida_path}")


ds=xr.open_dataset('acum_prec.nc', decode_timedelta=True)
ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
prec = ds['precipitacao_total'][:,:]
lat = ds['latitude']
lon = ds['longitude']
#ds


import pandas as pd

# Converte as strings dos atributos em datetime (caso estejam como string)
rodada_ini = pd.to_datetime(ds_soma.attrs['rodada_inicial'])
prev_ini = pd.to_datetime(ds_soma.attrs['previsao_inicial'])
prev_fim = pd.to_datetime(ds_soma.attrs['previsao_final'])

# Formata a string no estilo desejado
titulo = (
#    f"Inic: {rodada_ini:%HUTC %d/%b}    "
    f"Valid: {prev_ini:%HUTC %d/%b} to {prev_fim:%HUTC %d/%b}"
)
titulo_ = (f"Inic: {rodada_ini:%HUTC %d/%b}    ")

print(titulo)
print(titulo_)


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from scipy.ndimage import maximum_filter, minimum_filter

def plot_maxmin_points(lon, lat, data, extrema, nsize, color='k', transform=None, ax=None):
    if extrema == 'max':
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif extrema == 'min':
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError("O parâmetro 'extrema' deve ser 'max' ou 'min'")

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        ax.annotate(
            f'{int(data[mxy[i], mxx[i]])}',  # Apenas o valor
            xy=(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]]),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            color=color, size=12, weight='normal', fontfamily='sans-serif',
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="none", facecolor="white", alpha=0.6),
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],
            clip_on=True, annotation_clip=True,
            transform=ccrs.PlateCarree()
        )

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
              

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from matplotlib.patches import Rectangle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ------------------------
# VARIABLES DE EJEMPLO (Sustituir con tus datos reales)
# ------------------------
# prec -> matriz 2D de precipitación
# lon, lat -> grillas de coordenadas
# cores_prec -> lista de colores
# levels_prec -> niveles para contourf
# titulo, titulo_ -> textos para títulos
# ------------------------
# Reemplazar con tus datos reales
# lon, lat = np.meshgrid(prec.longitude, prec.latitude)
# prec = ...
# cores_prec = ...
# levels_prec = ...
# titulo = "Ejemplo"
# titulo_ = "2025-09-27 00UTC"

# ------------------------
# FIGURA PRINCIPAL
# ------------------------
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

# Shapefiles de países y departamentos
st0 = list(shpreader.Reader('ne_10m_admin_0_countries.shp').geometries())
ax.add_geometries(st0, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.5)
st1 = list(shpreader.Reader('ne_10m_admin_1_states_provinces.shp').geometries())
ax.add_geometries(st1, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.3)

# Plot de precipitación principal
plot = ax.contourf(lon, lat, prec, colors=cores_prec, levels=levels_prec, transform=ccrs.PlateCarree())

# Gridlines
gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3, color="gray")
gl.top_labels = False
gl.right_labels = False

# ------------------------
# INSET: ZOOM EN LIMA
# ------------------------
ax_inset = ax.inset_axes([0.02, 0.002, 0.34, 0.34], projection=ccrs.PlateCarree())
extent_lima = [-78.3, -75.3, -13.5, -9.8]
ax_inset.set_extent(extent_lima, crs=ccrs.PlateCarree())
ax_inset.add_geometries(st0, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.5)
ax_inset.add_geometries(st1, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.3)
plot_zoom = ax_inset.contourf(lon, lat, prec, colors=cores_prec, levels=levels_prec, transform=ccrs.PlateCarree())

# Título del inset
ax_inset.set_title("Mapa de Precipitación Total - Lima", fontsize=12, fontweight='bold', color='black', pad=12)

# Marcador Lima Metropolitana
ax_inset.plot(-77, -12, marker='o', color='black', markersize=8, transform=ccrs.PlateCarree(), zorder=12)

# Recuadro inset
rect = Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                 fill=False, color="black", linewidth=1.5, linestyle='-',
                 zorder=15)
ax_inset.add_patch(rect)

# Gridlines profesionales inset
gl_inset = ax_inset.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl_inset.top_labels = False
gl_inset.right_labels = False
gl_inset.left_labels = True
gl_inset.bottom_labels = True
gl_inset.xformatter = LongitudeFormatter(degree_symbol="°")
gl_inset.yformatter = LatitudeFormatter(degree_symbol="°")
gl_inset.xlabel_style = {'size': 10, 'color': 'black', 'weight': 'bold', 'rotation': 0}
gl_inset.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold', 'rotation': 90}

# ------------------------
# MÁXIMO DE PERÚ
# ------------------------
mask_peru = (
    (lon >= extent_peru[0]) & (lon <= extent_peru[1]) &
    (lat >= extent_peru[2]) & (lat <= extent_peru[3])
)
prec_peru = np.where(mask_peru, prec, np.nan)
idx_max_peru = np.nanargmax(prec_peru)
y, x = np.unravel_index(idx_max_peru, prec.shape)
max_val_peru = prec[y, x]

ax.plot(lon[y, x], lat[y, x], marker="*", color="black", markersize=14,
        transform=ccrs.PlateCarree(), zorder=20)
ax.text(lon[y, x] + 0.2, lat[y, x] + 0.2, f"{max_val_peru:.1f}",
        color="black", fontsize=12, fontweight="bold",
        transform=ccrs.PlateCarree(),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

# ------------------------
# MÁXIMO DE ZONA COMBINADA: Callao + Lima Province + Lima Metropolitana
# ------------------------
gdf = gpd.read_file('ne_10m_admin_1_states_provinces.shp')
zonas = gdf[gdf['name'].isin(['Callao', 'Lima', 'Lima Province'])]

# Crear máscara combinada
points = [Point(lon_, lat_) for lon_, lat_ in zip(lon.flatten(), lat.flatten())]
mask_combined = np.array([any(point.within(poly) for poly in zonas.geometry) for point in points])
mask_combined = mask_combined.reshape(prec.shape)

prec_zona = np.where(mask_combined, prec, np.nan)
idx_max_zona = np.nanargmax(prec_zona)
y, x = np.unravel_index(idx_max_zona, prec.shape)
max_val_zona = prec[y, x]

ax_inset.plot(lon[y, x], lat[y, x], marker="*", color="black", markersize=12,
              transform=ccrs.PlateCarree(), zorder=20)
ax_inset.text(lon[y, x] + 0.05, lat[y, x] + 0.05, f"{max_val_zona:.1f}",
              color="black", fontsize=10, fontweight="bold",
              transform=ccrs.PlateCarree(),
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

# ------------------------
# COLORBAR
# ------------------------
cbar = fig.colorbar(plot, ax=ax, orientation='vertical', fraction=0.035, pad=0.02)
cbar.set_label("Precipitación (mm)", fontsize=14)

# ------------------------
# COPYRIGHT / MARCA DE AGUA
# ------------------------
rect_x, rect_y, rect_width, rect_height = 0.89, 0.01, 0.1, 0.035
rectangle = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                              transform=ax.transAxes, color='yellow',
                              zorder=10, linewidth=1)
ax.add_patch(rectangle)
ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, "©Bach.Porras",
        transform=ax.transAxes, fontsize=10, fontweight='normal',
        color='blue', ha='center', va='center', zorder=10)

plt.show()