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
    dependencies = ["xarray", "cfgrib", "matplotlib", "cartopy", "scipy", "numpy"]

    for dep in dependencies:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

# Chamar a função para configurar o ambiente
setup_environment()

import datetime
import requests
import os
from datetime import timedelta

def gerar_url(rodada_data, rodada_hora, tentativas_max=3):
    rodada_datetime = datetime(
        rodada_data.year, rodada_data.month, rodada_data.day, int(rodada_hora)
)


    # Define validade com base na hora da rodada
    if rodada_hora == "00":
        validade = rodada_datetime + timedelta(hours=180)
    else:  # "12"
        validade = rodada_datetime + timedelta(hours=168)

    url = (
        f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/"
        f"{rodada_data.year}/{rodada_data.strftime('%m')}/{rodada_data.strftime('%d')}/{rodada_hora}/"
        f"WRF_cpt_07KM_{rodada_datetime.strftime('%Y%m%d%H')}_{validade.strftime('%Y%m%d%H')}.grib2"
    )

    print(f"Rodada: {rodada_datetime} → Projeção até: {validade} → Nome final: WRF_cpt_07KM_{rodada_datetime.strftime('%Y%m%d%H')}_{validade.strftime('%Y%m%d%H')}.grib2")

    return url

def tentar_download(rodada_data, rodada_hora):
    url = gerar_url(rodada_data, rodada_hora)
    nome_temp = "temp_wrf_download.grib2"
    nome_final = "wrf_cptec_6dias.grib2"

    print(f"Tentando baixar: {url}")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Remove os arquivos existentes, se houver
        if os.path.exists(nome_temp):
            os.remove(nome_temp)
        if os.path.exists(nome_final):
            os.remove(nome_final)

        # Salva o novo arquivo temporário
        with open(nome_temp, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Renomeia para o nome final
        os.rename(nome_temp, nome_final)
        print(f"✅ Download bem-sucedido e salvo como: {nome_final}")
        return True
    else:
        print(f"❌ Falha: {response.status_code}")
        return False

# Hora atual UTC
from datetime import datetime, UTC
agora = datetime.now(UTC)


# Lista de tentativas, prioridade do mais recente para o mais antigo
tentativas = []

# Prioridade:
# 1. Hoje às 12 UTC (se depois de 16 UTC)
# 2. Hoje às 00 UTC (se depois de 06 UTC)
# 3. Ontem às 12 UTC
# 4. Ontem às 00 UTC
#if agora.hour >= 16:
#    tentativas.append(("12", agora))
if agora.hour >= 6:
    tentativas.append(("00", agora))

ontem = agora - timedelta(days=1)
tentativas.append(("12", ontem))
tentativas.append(("00", ontem))

# Testar as opções em ordem
sucesso = False
for rodada_hora, rodada_data in tentativas:
    sucesso = tentar_download(rodada_data, rodada_hora)
    if sucesso:
        break

if not sucesso:
    print("❌ Nenhuma rodada disponível foi encontrada.")


import xarray as xr
ds = xr.open_dataset('wrf_cptec_6dias.grib2', filter_by_keys={'typeOfLevel': 'surface'}, decode_timedelta=True)
prec = ds['tp'] [:,:]
lat = ds['latitude']
lon = ds['longitude']

# Valor original
step_val = ds.step.values  # Ex: 648000000000000

# Converte de nanosegundos para horas
step_horas = int(step_val / 3.6e12)  # 1 hora = 3.6e12 nanosegundos

# Formata o título
titulo_1 = f"F{step_horas:03d}"
print("Step (horas):", step_horas)
print("Título:", titulo_1)

def plot_maxmin_points(lon, lat, data, extrema, nsize, color='k', ax=None, transform=None):
    # Definir filtro máximo ou mínimo
    data_ext = maximum_filter(data, nsize, mode='nearest') if extrema == 'max' else minimum_filter(data, nsize, mode='nearest')

    # Encontrar pontos extremos
    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        ax.annotate(
            f'{data[mxy[i], mxx[i]]:.1f}',
            xy=(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]]),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            color=color, size=12, weight='normal', fontfamily='sans-serif',
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="none", facecolor="white", alpha=0.6),
            path_effects=[path_effects.withStroke(linewidth=0.5, foreground="black")],
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

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from scipy.ndimage import maximum_filter, minimum_filter
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.geometry import Point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Verificando se temos dados para plotagem
if prec is not None:
    # Cria figura e eixos com projeção
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()})

    # Títulos e cabeçalhos
    ax.text(0.0, 1.025, r'WRF - CPTEC/INPE 7km/0.07°', fontsize=20, fontweight='bold', color='red', transform=ax.transAxes)
    ax.set_title('Precipitación Total (mm)', fontweight='bold', fontsize=16, loc='left')
    ax.set_title(ds.time.dt.strftime('Inic %HUTC %d/%b  ').item() + ds.valid_time.dt.strftime('Valid %HUTC %d/%b').item(),
                 color='#488f31', fontsize=15, loc='right')
    ax.text(0.60, 1.005, titulo_1, color='#1f77b4', fontsize=15, fontweight='bold', transform=ax.transAxes)

    # === EXTENT PARA PERÚ ===
    extent = [-82, -68, -19, 1]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # === SHAPEFILES ===
    countries_shp = shpreader.Reader('ne_10m_admin_0_countries.shp')
    for country in countries_shp.records():
        if country.attributes['ADMIN'] == 'Peru':
            ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1.2)

    states_shp = shpreader.Reader('ne_10m_admin_1_states_provinces.shp')
    for state in states_shp.records():
        if state.attributes['admin'] == 'Peru':
            ax.add_geometries([state.geometry], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.8)

    # === GRELHAS DE COORDENADAS ===
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25,
                      xlocs=np.arange(-90, -60, 3), ylocs=np.arange(-20, 5, 3), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # === PREPARAR LON/LAT ===
    lon, lat = np.meshgrid(prec.longitude, prec.latitude)
    lon = (lon - 360) if lon.max() > 180 else lon

    # === PLOTAGEM DO CAMPO ===
    plot = ax.contourf(lon, lat, prec, colors=cores_prec, levels=levels_prec, transform=ccrs.PlateCarree())

    # === MÁXIMO DE PERÚ (SOLO DENTRO DEL TERRITORIO) ===
    # Cargar shapefile de Perú
    gdf_peru = gpd.read_file('ne_10m_admin_0_countries.shp')
    peru_shape = gdf_peru[gdf_peru['ADMIN'] == 'Peru'].geometry.iloc[0]

    # Crear máscara para puntos dentro de Perú
    points_all = [Point(lon_, lat_) for lon_, lat_ in zip(lon.flatten(), lat.flatten())]
    mask_peru_shape = np.array([point.within(peru_shape) for point in points_all])
    mask_peru_shape = mask_peru_shape.reshape(prec.shape)

    # Aplicar máscara
    prec_peru = np.where(mask_peru_shape, prec, np.nan)

    if np.any(~np.isnan(prec_peru)):
        idx_max_peru = np.nanargmax(prec_peru)
        y_max_peru, x_max_peru = np.unravel_index(idx_max_peru, prec.shape)
        max_val_peru = prec[y_max_peru, x_max_peru]

        # Marcar máximo en el mapa principal
        ax.plot(lon[y_max_peru, x_max_peru], lat[y_max_peru, x_max_peru],
                marker="*", color="black", markersize=14,
                transform=ccrs.PlateCarree(), zorder=20)
        ax.text(lon[y_max_peru, x_max_peru] + 0.2, lat[y_max_peru, x_max_peru] + 0.2,
                f"{max_val_peru:.1f}",
                color="black", fontsize=12, fontweight="bold",
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
    else:
        max_val_peru = np.nan

    # === MARCAR MÁXIMOS (OPCIONAL) ===
    # plot_maxmin_points(lon, lat, prec, 'max', 20, color='k', transform=ccrs.PlateCarree(), ax=ax)

    # === MÁXIMA PRECIPITAÇÃO NA ÁREA DEFINIDA ===
    mask_lon = (lon >= extent[0]) & (lon <= extent[1])
    mask_lat = (lat >= extent[2]) & (lat <= extent[3])
    mask = mask_lon & mask_lat

    if np.any(mask):
        max_precip = np.nanmax(np.where(mask, prec, np.nan))
        print("Máxima precipitación en la región:", max_precip)
    else:
        max_precip = np.nan
        print("⚠️ No hay datos en la región definida.")

    # === TEXTO COM MÁXIMA PRECIPITAÇÃO ===
    if not np.isnan(max_val_peru):
        ax.text(0.99, 0.01, f'Máx Precipitación Perú: {max_val_peru:.1f} mm',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'),
                path_effects=[path_effects.withStroke(linewidth=1.2, foreground="black")])

    # ========================================
    # === INSET: ZOOM EN LIMA PROVINCIA ===
    # ========================================
    ax_inset = ax.inset_axes([0.02, 0.002, 0.34, 0.34], projection=ccrs.PlateCarree())
    extent_lima = [-78.3, -75.3, -13.5, -9.8]
    ax_inset.set_extent(extent_lima, crs=ccrs.PlateCarree())

    # Shapefiles en el zoom
    countries_shp_inset = shpreader.Reader('ne_10m_admin_0_countries.shp')
    for country in countries_shp_inset.records():
        if country.attributes['ADMIN'] == 'Peru':
            ax_inset.add_geometries([country.geometry], ccrs.PlateCarree(),
                                   facecolor='none', edgecolor='black', linewidth=0.6)

    states_shp_inset = shpreader.Reader('ne_10m_admin_1_states_provinces.shp')
    for state in states_shp_inset.records():
        if state.attributes['admin'] == 'Peru':
            ax_inset.add_geometries([state.geometry], ccrs.PlateCarree(),
                                   facecolor='none', edgecolor='black', linewidth=0.4)

    # Plot de precipitación en el zoom
    plot_zoom = ax_inset.contourf(lon, lat, prec, colors=cores_prec, levels=levels_prec,
                                   transform=ccrs.PlateCarree())

    # Título del inset
    ax_inset.set_title("Mapa de Precipitación Total - Lima", fontsize=12, fontweight='bold',
                      color='black', pad=12)

    # Gridlines profesionales en el inset
    gl_inset = ax_inset.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                                   alpha=0.7, linestyle='--')
    gl_inset.top_labels = False
    gl_inset.right_labels = False
    gl_inset.left_labels = True
    gl_inset.bottom_labels = True
    gl_inset.xformatter = LongitudeFormatter(degree_symbol="°")
    gl_inset.yformatter = LatitudeFormatter(degree_symbol="°")
    gl_inset.xlabel_style = {'size': 10, 'color': 'black', 'weight': 'bold', 'rotation': 0}
    gl_inset.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold', 'rotation': 90}

    # === MÁXIMO DE ZONA LIMA (Callao + Lima Province + Lima Metropolitana) ===
    gdf = gpd.read_file('ne_10m_admin_1_states_provinces.shp')
    zonas = gdf[gdf['name'].isin(['Callao', 'Lima', 'Lima Province'])]

    # Crear máscara combinada
    points = [Point(lon_, lat_) for lon_, lat_ in zip(lon.flatten(), lat.flatten())]
    mask_combined = np.array([any(point.within(poly) for poly in zonas.geometry) for point in points])
    mask_combined = mask_combined.reshape(prec.shape)

    prec_zona = np.where(mask_combined, prec, np.nan)
    if np.any(~np.isnan(prec_zona)):
        idx_max_zona = np.nanargmax(prec_zona)
        y_max_zona, x_max_zona = np.unravel_index(idx_max_zona, prec.shape)
        max_val_zona = prec[y_max_zona, x_max_zona]

        # Marcar máximo en el zoom
        ax_inset.plot(lon[y_max_zona, x_max_zona], lat[y_max_zona, x_max_zona],
                     marker="*", color="black", markersize=12,
                     transform=ccrs.PlateCarree(), zorder=20)
        ax_inset.text(lon[y_max_zona, x_max_zona] + 0.05, lat[y_max_zona, x_max_zona] + 0.05,
                     f"{max_val_zona:.1f}",
                     color="black", fontsize=10, fontweight="bold",
                     transform=ccrs.PlateCarree(),
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    # Marcador en Lima Metropolitana
    ax_inset.plot(-77, -12, marker='o', color='black', markersize=8,
                  transform=ccrs.PlateCarree(), zorder=12)

    # Recuadro alrededor del inset
    rect_inset = Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                           fill=False, color="black", linewidth=1.5, linestyle='-', zorder=15)
    ax_inset.add_patch(rect_inset)

    # ========================================

    # === COLORBAR ===
    fig_height = fig.get_size_inches()[1]
    ax_height = (ax.get_position().y1 - ax.get_position().y0) * fig_height
    fraction = ax_height / fig_height * 0.0612
    cbar = fig.colorbar(plot, ax=ax, orientation='vertical', fraction=fraction, pad=0.01)
    cbar.set_label('Precipitación (mm)', fontsize=14)

    selected_ticks = levels_prec[::2]
    if selected_ticks[0] != levels_prec[0]:
        selected_ticks = np.insert(selected_ticks, 0, levels_prec[0])
    if selected_ticks[-1] != levels_prec[-1]:
        selected_ticks = np.append(selected_ticks, levels_prec[-1])
    cbar.set_ticks(selected_ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in selected_ticks])

    # === LOGO OU TEXTO FINAL ===
    rect_x, rect_y, rect_width, rect_height = 0.005, 0.005, 0.065, 0.02
    rectangle = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, transform=ax.transAxes,
                                  color='yellow', zorder=10, linewidth=1)
    ax.add_patch(rectangle)
    ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, "Bach.Daniel", transform=ax.transAxes,
            fontsize=8, fontweight='normal', color='blue', ha='center', va='center', zorder=11)

    # === SALVAR FIGURA ===
    plt.savefig('WRF_CPTEC_PERU_prec.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    print("❌ Ningún dato de precipitación fue procesado.")