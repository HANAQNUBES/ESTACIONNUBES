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
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image

cpu_count = os.cpu_count() or 1
try:
    from .Down_and_Consolid import GFS_down,ETA_down,WRF_down
except:
    from Down_and_Consolid import GFS_down,ETA_down,WRF_down
#%%
class MAPEADOR:
    """
    Gestor de información relacionada con mapas.

    Se encarga de:
    1. Descargar la información de los modelos de predicción
    2. Descargar los shapefiles para los mapas
    3. Armar los mapas
    4. Gestión de archivos de todo lo anterior

    Attributes:

        modelos_registrados (dict) : Relaciona el nombre(str) de cada modelo que se tiene con el método que ejecuta su actualizacion; lo que permite actualizar cualquier modelo registrado solo pasando el nombre, independientemente del proceso para actualizarlo.

        folder (str) : nombre de la carpeta donde se guardará todo lo que realice el gestor.

        salida_eta (str) : nombre de la sub-carpeta donde se almacena los archivos relacionados al modelo eta

        salida_wrf (str) : nombre de la sub-carpeta donde se almacena los archivos relacionados al modelo wrf
        
        shapefiles (dic) : relaciona y ordena los archivos y maletas de descarga para paises y provincias

    """
    def __init__(self,folder='temp')->None:

        """
        Inicializa el controlador de los mapas

        Args:

            folder (str,optional) : Nombre de la carpeta donde se almacenará todo


        """

        #folder (str): Nombre de la carpeta temporal* para todos los archivos de este gestor
        self.folder=folder
        os.makedirs(self.folder,exist_ok=True)#maleta de archivos temporales al iniciar
        self.modelos_registrados = {
            "eta": self.eta,
            "wrf": self.wrf,
            "gfs": self.gfs,
            #"icon": self.refresh_icon,
        }
        self.salidas={
            "eta": f"{self.folder}/datos_eta",
            "wrf": f"{self.folder}/datos_wrf",
            "gfs": f"{self.folder}/datos_gfs",
            #"icon": self.refresh_icon,
        }
        
        
        self.shapefiles={'countries':'ne_10m_admin_0_countries',
            'states':'ne_10m_admin_1_states_provinces'}
        #endregion

    def plot_precipitation_map(self, ds: xr.Dataset,folder :str) -> str:
        """
        Genera un mapa de precipitación y lo guarda como PNG.
        
        Args:
            ds (xarray.Dataset): Dataset con datos de precipitación
            
        Returns:
            str: Ruta al archivo PNG generado
        """
        # === Coordinate handling ===
        try:
            lon_name = next(name for name in ds.coords if any(coord in name.lower() for coord in ['lon', 'x']))
            ds = ds.assign_coords(longitude=((ds[lon_name] + 180) % 360) - 180)
            lon = ds['longitude']
        except StopIteration:
            print('Longitude coordinate not found!')
            lon = None

        try:
            lat_name = next(name for name in ds.coords if any(coord in name.lower() for coord in ['lat', 'y']))
            lat = ds[lat_name]
        except StopIteration:
            print('Latitude coordinate not found!')
            lat = None

        # === Precipitation variable ===
        POSIBLE_NAMES = ('precipitacao_total','tp','Total Precipitation','Precipitacion total')
        
        for posible_name in POSIBLE_NAMES:
            try:
                prec = ds[posible_name][:,:]
                break
            except:pass
        else:    
            raise KeyError(f"No valid precipitation variable found (tried {POSIBLE_NAMES})")


        # === Time handling (en inglés, pero fechas en español abajo) ===
        init_time = pd.to_datetime(ds.time.values)
        
        if hasattr(ds, 'time_bounds'):
            valid_start = pd.to_datetime(ds.time_bounds.values[0, 0])
            valid_end = pd.to_datetime(ds.time_bounds.values[0, 1])
        else:
            valid_start = valid_end = init_time

        # === TÍTULOS EN ESPAÑOL (lo que el usuario ve) ===
        # Formateadores para fechas en español
        meses_es = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                    'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        def format_datetime_es(dt):
            """Formatea datetime en español"""
            return f"{dt.hour:02d}UTC {dt.day:02d}{meses_es[dt.month-1]}"
        
        titulo_principal = f"Válido: {format_datetime_es(valid_start)} al {format_datetime_es(valid_end)}"
        titulo_inicio = f"Inicio: {format_datetime_es(init_time)}"
        
        # Título superior (opcional, si quieres agregar algo como "PRONÓSTICO")
        # graph_name = ds.attrs.get('descripcion', 'PRONÓSTICO DE PRECIPITACIÓN')

        # === Colormap (se mantiene igual) ===
        colors_precip = ['#ffffff','#dedede','#bfbfbf','#a1a1a1','#828282','#b7f0be','#a0dcb3',
                        '#88c8a9','#71b49e','#5aa093','#438c88','#2b787e','#146473','#1450b4',
                        '#2a61bb','#3f73c2','#5584c9','#6b96d0','#80a7d6','#96b9dd','#accae4',
                        '#c1dceb','#d7edf2','#cebce0','#c9addb','#c49ed5','#bf90d0','#ba81ca',
                        '#b472c5','#af63bf','#aa55ba','#a546b4','#a037af','#a53a34','#ab453f',
                        '#b24f49','#b85a54','#be655f','#c56f69','#cb7a74','#d1857f','#d88f89',
                        '#de9a94','#f8eea2','#eed68c','#e5bd76','#dba560','#d28c4a','#c87434',
                        '#ac632d','#9e5b29','#8f5225','#814a21','#72421d','#643919','#553115']
        
        levels_precip = [0, 0.5, 1, 1.5, 2.0, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11.5, 13, 14.5, 16,
                        17.5, 19, 20.5, 22, 23.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45,
                        47.5, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 175, 200,
                        225, 250, 275, 300, 325, 350, 375, 400, 500]
        
        levels_precip = np.arange(0,20,0.5)
        # === Create figure ===
        fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()})

        # === TÍTULOS EN ESPAÑOL ===
        # ax.text(0.0, 1.02, graph_name, fontsize=20, fontweight='bold', color='red', transform=ax.transAxes)
        ax.set_title('Precipitación Total (mm)', fontweight='bold', fontsize=16, loc='left')
        ax.set_title(titulo_principal, color='#488f31', fontsize=15, loc='right')
        ax.text(0.50, 1.005, titulo_inicio, color='#1f77b4', fontsize=15, fontweight='normal',
                transform=ax.transAxes, ha='center')

        # === Peru extent ===
        extent_peru = [-82, -68, -19, 1]
        ax.set_extent(extent_peru, crs=ccrs.PlateCarree())

        # === Shapefiles (cached) ===
        if not hasattr(self, '_cached_shapefiles'):
            self._cached_shapefiles = {}
            for name, linewidth in zip(self.shapefiles, [0.5, 0.3]):
                path = f"{self.folder}/shapefiles/{self.shapefiles[name]}.shp"
                if os.path.exists(path):
                    self._cached_shapefiles[name] = list(shpreader.Reader(path).geometries())
                else:
                    self._cached_shapefiles[name] = None
                    print(f"Shapefile no encontrado: {path}")

        # Plot main map shapefiles
        for name, linewidth in zip(self.shapefiles, [0.5, 0.3]):
            if self._cached_shapefiles[name]:
                ax.add_geometries(self._cached_shapefiles[name], ccrs.PlateCarree(),
                                edgecolor='k', facecolor='none', linewidth=linewidth)

        # === Precipitation contour ===
        plot = ax.contourf(lon, lat, prec, colors=colors_precip, levels=levels_precip,
                        transform=ccrs.PlateCarree(), antialiased=True)

        # === Gridlines (etiquetas en inglés está bien, son coordenadas) ===
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3, color="gray")
        gl.top_labels = False
        gl.right_labels = False

        # === Lima inset (TÍTULO EN ESPAÑOL) ===
        ax_inset = ax.inset_axes([0.02, 0.002, 0.34, 0.34], projection=ccrs.PlateCarree())
        extent_lima = [-78.3, -75.3, -13.5, -9.8]
        ax_inset.set_extent(extent_lima, crs=ccrs.PlateCarree())

        # Plot inset shapefiles
        for name, _ in zip(self.shapefiles, [0.5, 0.3]):
            if self._cached_shapefiles[name]:
                ax_inset.add_geometries(self._cached_shapefiles[name], ccrs.PlateCarree(),
                                    edgecolor='k', facecolor='none', linewidth=0.5)

        # Plot precipitation on inset
        ax_inset.contourf(lon, lat, prec, colors=colors_precip, levels=levels_precip,
                        transform=ccrs.PlateCarree())

        # Lima point
        ax_inset.plot(-77, -12, marker='o', color='black', markersize=8,
                    transform=ccrs.PlateCarree(), zorder=12)

        # Inset border
        rect = Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                        fill=False, color="black", linewidth=1.5, linestyle='-', zorder=15)
        ax_inset.add_patch(rect)

        # Inset gridlines
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
        
        # Título del inset en español
        ax_inset.set_title("Mapa de Precipitación - Lima", fontsize=12, fontweight='bold', 
                        color='black', pad=12)

        # === Maximum precipitation points ===
        lon_vals = lon.values if hasattr(lon, 'values') else np.asarray(lon)
        lat_vals = lat.values if hasattr(lat, 'values') else np.asarray(lat)
        prec_vals = prec.values if hasattr(prec, 'values') else np.asarray(prec)

        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)

        # Peru max
        mask_peru = ((lon_2d >= extent_peru[0]) & (lon_2d <= extent_peru[1]) &
                    (lat_2d >= extent_peru[2]) & (lat_2d <= extent_peru[3]))
        
        prec_peru = np.where(mask_peru, prec_vals, np.nan)
        max_idx_peru = np.nanargmax(prec_peru)
        y_idx, x_idx = np.unravel_index(max_idx_peru, prec_vals.shape)
        
        max_val_peru = float(prec_vals[y_idx, x_idx])
        max_lon_peru = float(lon_vals[x_idx])
        max_lat_peru = float(lat_vals[y_idx])

        ax.plot(max_lon_peru, max_lat_peru, marker="*", color="black", markersize=14,
                transform=ccrs.PlateCarree(), zorder=20)
        ax.text(max_lon_peru + 0.2, max_lat_peru + 0.2, f"{max_val_peru:.1f} mm",
                color="black", fontsize=12, fontweight="bold",
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

        # Lima max
        mask_lima = ((lon_2d >= extent_lima[0]) & (lon_2d <= extent_lima[1]) &
                    (lat_2d >= extent_lima[2]) & (lat_2d <= extent_lima[3]))
        
        prec_lima = np.where(mask_lima, prec_vals, np.nan)
        if not np.all(np.isnan(prec_lima)):
            max_idx_lima = np.nanargmax(prec_lima)
            y_lima, x_lima = np.unravel_index(max_idx_lima, prec_vals.shape)
            
            max_val_lima = float(prec_vals[y_lima, x_lima])
            max_lon_lima = float(lon_vals[x_lima])
            max_lat_lima = float(lat_vals[y_lima])

            ax_inset.plot(max_lon_lima, max_lat_lima, marker="*", color="black", markersize=12,
                        transform=ccrs.PlateCarree(), zorder=20)
            ax_inset.text(max_lon_lima + 0.05, max_lat_lima + 0.05, f"{max_val_lima:.1f} mm",
                        color="black", fontsize=10, fontweight="bold",
                        transform=ccrs.PlateCarree(),
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

        # === Colorbar (etiqueta en español) ===
        cbar = fig.colorbar(plot, ax=ax, orientation='vertical', fraction=0.035, pad=0.02)
        cbar.set_label("Precipitación (mm)", fontsize=14)

        # === Copyright (se mantiene) ===
        rect_x, rect_y, rect_width, rect_height = 0.89, 0.01, 0.1, 0.035
        rectangle = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                    transform=ax.transAxes, color='yellow', zorder=10, linewidth=1)
        ax.add_patch(rectangle)
        ax.text(rect_x + rect_width / 2, rect_y + rect_height / 2, "©Bach.Porras",
                transform=ax.transAxes, fontsize=10, fontweight='normal',
                color='blue', ha='center', va='center', zorder=10)

        # === Generate output path ===
        output_filename = f"precip_{init_time.strftime('%Y-%m-%d %H:%M')}.png"

        output_path = folder+'/imgs/'
        os.makedirs(output_path,exist_ok=True)#crea maleta img en cada maleta de datos si es que no está
        plt.savefig(output_path+output_filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()#previene que la imagen se muestre( ai wakala)

        return output_path+output_filename

    def _Shapefiles(self,rtr=10)->None:

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
            self._Shapefiles(rtr-1)

    def _make_gifs(self, fotos, filename)->None:
        """
        Combina una lista de imágenes en un GIF
        
        Args:
            fotos (list): Lista de rutas a las imágenes PNG
            filename (str): Nombre del archivo GIF de salida
        
        Returns:
            str: Ruta completa al GIF generado o None si falla
        """
        if not fotos:
            print("❌ No hay imágenes para crear GIF")
            return None
        
        # Ruta completa de salida
        if not filename.endswith('.gif'):
            filename += '.gif'
        output_path = filename
        
        try:
            # Cargar imágenes
            imagenes = []
            for foto in fotos:
                if os.path.exists(foto):
                    img = Image.open(foto)
                    imagenes.append(img)
                else:
                    print(f"⚠️ Imagen no encontrada: {foto}")
            
            if not imagenes:
                print("❌ No se pudo cargar ninguna imagen")
                return None
            
            # Crear GIF (200ms por frame, loop infinito)
            imagenes[0].save(
                output_path,
                save_all=True,
                append_images=imagenes[1:],
                duration=200,
                loop=0,
                optimize=True
            )
            
            print(f"✅ GIF creado: {output_path} ({len(imagenes)} frames)")
            return output_path
            
        except Exception as e:
            print(f"❌ Error creando GIF: {e}")
            return None

    def gfs(self):
        gfs_tool=GFS_down(self.salidas['gfs'])
        gfs_tool.download()
        ds=gfs_tool.consolid({'filter_by_keys': { 'name': ['Total Precipitation']}},'lluvia')
        
        # La primera hora (índice 0) es acumulado 0→3, se queda igual
        lluvia=np.nan_to_num(ds.tp.values.copy(),0)
        ds['tp'][1:][::2]= lluvia[1:][::2]-lluvia[::2][:-1]## Explicación (1)
        return ds

    def eta(self):
        eta_tool=ETA_down(self.salidas['eta'])
        eta_tool.download()
        ds=eta_tool.consolid({'filter_by_keys':{'stepType': 'accum', 'typeOfLevel': 'surface', 'dataType': 'fc'}},'lluvia')
        return ds.rename_vars({'unknown':'tp'})

    def wrf(self):
        eta_tool=WRF_down(self.salidas['wrf'])
        eta_tool.download()
        ds=eta_tool.consolid({'filter_by_keys':{'typeOfLevel': 'surface','name':'Total Precipitation'}},'lluvia')
        # La primera hora (índice 0) es acumulado 0→3, se queda igual
        lluvia=np.nan_to_num(ds.tp.values.copy(),0)
        ds['tp'][1:]= lluvia[1:]-lluvia[:-1]## Explicación (1)
        
        return ds[['tp']]
    
    def refresh_model(self,name,MAX_WORKERS= max(cpu_count-1,1))->tuple:
        """
        Funcion generalista para recargar cualquier mapa de cualquier modelo disponible

        Args:
            name (str) : Nombre del modelo deseaso
        
        Returns:
            tuple: (path_imagen, datetime_rodada)
                - path_imagen (str): Ruta al archivo PNG generado
                - datetime_rodada (datetime): Fecha/hora de inicio del modelo
        """
        FOLDER=self.salidas[name]
        fecha=datetime.today().strftime('%Y%m%d')
        log_file=FOLDER+'/imgs/log_file_imgs.txt'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f_read:content = f_read.read()
            ultima=content[content.find(':')+2:]
            if ultima==fecha:
                if len(gb(FOLDER+'/*'))<100:
                    dibujar=True
                else:dibujar=False
            else:dibujar=True
        else:dibujar=True
        gif_final = os.path.join(FOLDER, name)
        if dibujar:
            print('Se requiere re-dibujar... ')
            for i in gb(FOLDER+'/imgs/*'):os.remove(i)#intenta eliminar las imagenes en la respectiva carpeta, si es que existen
            self._Shapefiles()#verifica y descarga shapefiles si los necesita
            ds=self.modelos_registrados[name]()#corre el modelo,este puede descargar, consolidar o reusar un archivo segun el caso
            print('')
            # 3. Crear wrapper que llame al método de instancia
            plot_func = partial(self.plot_precipitation_map, folder=FOLDER)
            # 4. Ejecutar en paralelo
            datasets_tiempo = [ds.isel(time=i) for i in range(len(ds.time))]

            with Pool(processes=MAX_WORKERS) as pool:

                # Usar imap para ver progreso
                resultados = []
                for i, resultado in enumerate(pool.imap(plot_func, datasets_tiempo)):
                    resultados.append(resultado)
                    print(f"✅ Progreso: {i+1}/{len(datasets_tiempo)} mapas completados", end='\r')
                print() 

            fotos=gb(FOLDER+'/imgs/**.png')
            fotos.sort() 
            self._make_gifs(fotos,gif_final)     

            with open(log_file, 'w') as f_write:f_write.write(f"Ultima fecha : {fecha}")

        else:
            print('Fotos existentes y actuales, no es necesario redibujar')

        return gif_final+'.gif',datetime.today()  ### esto tiene que devolver la direccion relativa del gif y algun initial time

#%%
"test zone"
if __name__=='__main__':
    mapper=MAPEADOR('temp')#inicializamos
    gif_file,fecha=mapper.refresh_model('wrf')#tratamos de actualizar eta
    print(f"gif creado exitosamente, guardado en {gif_file}")
    print(f"fecha actualizada :{fecha}")
# %%
