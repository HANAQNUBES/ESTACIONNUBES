#%%
"Librerias"
from datetime import datetime, timezone, timedelta
import os
import urllib.request
import zipfile
from tqdm import tqdm
import time
import cfgrib
import xarray as xr
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from glob import glob as gb
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#%%
class MAPEADOR:
    def __init__(self,folder='temp')->None:
        self.folder=folder
        os.makedirs(self.folder,exist_ok=True)#maleta de archivos temporales al iniciar

        #region Valores default 
        self.salida_eta=f"{self.folder}/dados_eta_8km"
        self.salida_wrf=f"{self.folder}/datos_wrf"
        self.shapefiles={'countries':'ne_10m_admin_0_countries',
            'states':'ne_10m_admin_1_states_provinces'}
        #endregion

    def Mapa_de_pp(self,ds)->str:
        """Genera el mapa de precipitación y lo guarda como imagen."""
        #region Corrección: buscar nombre correcto de variables
        try:
            lon_name = [i for i in ds.coords if ('lon' in i)or('x' in i)][0]
            ds = ds.assign_coords(longitude=((ds[lon_name] + 180) % 360) - 180)
            lon = ds['longitude'] 
        except IndexError:
            print('No se encontró coordenada para la longitud !')
            lon = None
        try:
            lat_name = [i for i in ds.coords if ('lat' in i)or('y' in i)][0]
            lat = ds[lat_name] 
        except IndexError:
            print('No se encontró coordenada para la latitud !')
            lat = None

        try:prec = ds['precipitacao_total'][:, :]
        except:prec = ds['tp'][:, :]
        #endregion
        # Títulos

        rodada_ini = pd.to_datetime(ds.attrs['rodada_inicial'])
        prev_ini = pd.to_datetime(ds.attrs['previsao_inicial'])
        prev_fim = pd.to_datetime(ds.attrs['previsao_final'])

        graph_name=ds.attrs['descricao']
        
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
        ax.text(0.0, 1.02, graph_name, fontsize=20, fontweight='bold', color='red', transform=ax.transAxes)
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

        for i,j in zip(self.shapefiles,[.5,.3]):
            path=f"{self.folder}/shapefiles/{self.shapefiles[i]}.shp"
            if os.path.exists(path):
                st0 = list(shpreader.Reader(path).geometries())
                ax.add_geometries(st0, ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=j)
        
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
        plt.savefig(ds.attrs['path_file'], dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return ds.attrs['path_file']

    def _Shapefiles(self)->None:

        """Descarga los shapefiles necesarios si no existen."""
        shapefiles=self.shapefiles
        try:
            for i in shapefiles:
                path=f"{self.folder}/shapefiles/{shapefiles[i]}.shp"
                if not os.path.exists(path):
                    url = f"https://naciscdn.org/naturalearth/10m/cultural/{shapefiles[i]}.zip"
                    urllib.request.urlretrieve(url, f"{self.folder}/{i}.zip")
                    with zipfile.ZipFile(f"{self.folder}/{i}.zip", 'r') as zip_ref:
                        zip_ref.extractall(f"{self.folder}/shapefiles")
                    os.remove(f"{self.folder}/{i}.zip")
                    print(f"✓ {i} Shapefile's ready")
        except Exception as e:
            print(f"⚠️  Error al descargar shapefiles: {e}")

    def eta(self)->None:
        "metodo que verifica, baja y actualiza el eta"

        def _Descargar_eta(hours=5)->datetime:
            """Descarga los datos del modelo ETA del CPTEC/INPE en formato  GRIB2 """
            os.makedirs(self.salida_eta, exist_ok=True)#comprueba carpetas
            delay = timedelta(hours=4)# Detecta corrida más reciente (00 o 12 UTC)
            now = datetime.now(timezone.utc) - delay
            data_base = now.replace(hour=int("00"), minute=0, second=0, microsecond=0)#redondeamos hacia atras
            
            print(f"📅 Hora Redondeada: {data_base.strftime('%Y-%m-%d %H:%M')} UTC")
            
            # Genera lista de archivos GRIB2
            for h in tqdm(range(hours), desc="📥 Descargando archivos ETA"):
                data_prev = data_base + timedelta(hours=h)
                name = f"Eta_ams_08km_{data_base.strftime('%Y%m%d%H')}_{data_prev.strftime('%Y%m%d%H')}.grib2"
                url = f"https://dataserver.cptec.inpe.br/dataserver_modelos/eta/ams_08km/brutos/{data_base.strftime('%Y/%m/%d/%H')}/{name}"
                # Descarga los archivos GRIB2
                caminho = os.path.join(self.salida_eta, name)
                if os.path.exists(caminho):# si ya existe, no se hace nada
                    continue
                sucsses = False
                for intento in range(3):#intenta re-descargar 3 veces
                    try:
                        urllib.request.urlretrieve(url, caminho)
                        sucsses = True
                        break
                    except Exception as e:
                        print(f"⚠️ Intento {intento+1}/3 falló para {name}: {e}")
                        time.sleep(5)
                if not sucsses:
                    print(f"❌ Error al descargar {name}")
            return data_base  
    
        def _Procesar_pp_eta()->xr.Dataset:
            """Procesa los datos ETA y calcula la precipitación acumulada."""            
            soma_unknown = None
            lat = lon = None
            rodada_inicial = rodada_final = None
            previsao_inicial = previsao_final = None
            
            arquivos = sorted([f for f in os.listdir(self.salida_eta) if f.endswith('.grib2')])[1:]
            
            for i, arq in enumerate(arquivos):
                caminho = os.path.join(self.salida_eta, arq)
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
            ds = xr.Dataset(
                {"precipitacao_total": soma_unknown},
                attrs={
                    "descricao": "Precipitación acumulada modelo ETA",
                    "rodada_inicial": str(rodada_inicial),
                    "rodada_final": str(rodada_final),
                    "previsao_inicial": str(previsao_inicial),
                    "previsao_final": str(previsao_final),
                    "path_file": str(self.salida_eta+'/mapa_precipitacion_eta.png')
                }
            )
            # Guardar NetCDF
            saida_path = f"{self.salida_eta}/acum_prec.nc"
            ds.to_netcdf(saida_path)
            
            return ds

        self._Shapefiles()
        timer=_Descargar_eta()
        ds = _Procesar_pp_eta()
        return ds,timer
     
    def wrf(self)->None:
        "metodo que verifica,baja y actualiza el wrf"
        def _Descargar_wrf()->list:
            """Descarga los datos GRIB2 del modelo WRF del CPTEC/INPE."""
            
            agora = datetime.now(timezone.utc)
            os.makedirs(self.salida_wrf, exist_ok=True)#comprueba carpetas
            
            
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
                nombre_final = f"{self.salida_wrf}/wrf_cptec_6dias_{rodada_data.strftime('Y%m%d%H')}.grib2"
                self.wrf_file=nombre_final
                if os.path.exists(nombre_final):
                    print('Ya se tiene el Ultimo archivo disponible')
                    return nombre_final,rodada_datetime
                else:
                    try:
                        response = requests.get(url, stream=True, timeout=300)
                        
                        if response.status_code == 200:
                            for i in gb(f"{a.salida_wrf}/**.grib2"):os.remove(i)
                            
                            with open(nombre_final, "wb") as f:
                                for chunk in response.iter_content(chunk_size=64*1024):
                                    f.write(chunk)
                            
                            print(f"✅ WRF descargado: {nombre_final}")
                            return nombre_final,rodada_datetime
                    except Exception as e:
                        print(f"⚠️ Error descargando WRF: {e}")
                        continue
            
            raise Exception("No se pudo descargar ninguna rodada WRF disponible")
        name,rodada=_Descargar_wrf()
        ds = xr.open_dataset(name, 
                         filter_by_keys={'typeOfLevel': 'surface'}, 
                         decode_timedelta=True)

        ds=ds.assign_attrs(
            descricao="WRF - CPTEC/INPE 7km/0.07°", 
            rodada_inicial=rodada,
            previsao_inicial=rodada,
            previsao_final=rodada+timedelta(days=6),
            path_file=self.salida_wrf+'/mapa_precipitacion_wrf.png')
        return ds

    def refresh_eta(self)->str:
        self._Shapefiles()
        eta,time_ini=self.eta()
        return self.Mapa_de_pp(eta),time_ini
    
    def refresh_wrf(self)->str:
        self._Shapefiles()
        wrf=self.wrf()
        return self.Mapa_de_pp(wrf)
    
    def REFRESH_ALL(self)->None:
        self._Shapefiles()
        eta=self.eta()
        wrf=self.wrf()
        self.Mapa_de_pp(eta)
        self.Mapa_de_pp(wrf)
        print('Todo Está Actualizado')
#%%
if __name__=='__main__':
    a=MAPEADOR()
    a.REFRESH_ALL()
# %%
