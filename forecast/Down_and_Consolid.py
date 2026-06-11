#%%
#Librerias+
from datetime import*
import os                    # Interfaces del sistema operativo
from functools import partial
import requests              # Librería HTTP
import numpy as np               # Computación científica
from glob import glob as gb      # Búsqueda de archivos
import xarray as xr              # Manejo de datos multidimensionales
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import time
import concurrent.futures
import warnings
import psutil
from pathlib import Path
warnings.filterwarnings("ignore")
#3m 2.4 sec a 30Mps~
cpu_count = os.cpu_count() or 1
#%%

t_ini=datetime.now(UTC).replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0)


def build_wrf_url_and_filename(hora):
    """
    Funcion generadora de link y nombre de archivo

    Arguments
    ---
    fecha_forec (datetime.datetime):
        fecha del pronostico (valid_time)
    """
    fecha_forec=t_ini + timedelta(hours=float(hora))
    file_name=f"WRF_cpt_07KM_{t_ini.strftime('%Y%m%d%H')}_{fecha_forec.strftime('%Y%m%d%H')}.grib2"
    url = (f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/"
           f"{t_ini.year}/{t_ini.strftime('%m')}/{t_ini.strftime('%d')}/00/"
           f"{file_name}"
                )
    return url,file_name
# %%
A=build_wrf_url_and_filename(9)
# %%
class GeneralDownloader:

    def __init__(self,carpeta:str,hours=np.arange(0,14*24,3),fecha=datetime.now(UTC).replace(tzinfo=None))->None:

        self.carpeta=carpeta
        self.fecha=fecha.strftime('%Y%m%d')
        self.hours=hours
        os.makedirs(self.carpeta+'/Consolidados',exist_ok=True)

    def _crear_sesion(self, max_retries=10):
        """Configura y crea una sesión HTTP con pool de conexiones"""
        session = requests.Session()
        
        # Estrategia de reintentos
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False
        )
        
        # Adapter con pool grande
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,    # Conexiones iniciales
            pool_maxsize=100,        # Máximo de conexiones
            pool_block=True           # Bloquear si el pool está lleno
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "GFS-downloader/1.0"})
        
        return session
    
    def _build_links_and_filename(self, hora)->None:
        """ IMPLEMENTAR EN SUBCLASE"""
        raise NotImplementedError("Cada modelo debe implementar _generar_links")

    def _verify_download(self,file_path,expected_min_size=500):
        """Verifica existencia y tamaño > expected_min_size."""
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > expected_min_size:
                return True
            # Archivo demasiado pequeño: eliminar
            try:
                os.remove(file_path)
            except Exception:
                pass
        return False

    def _single_download(self, session, hour, max_retries=10):
        carpeta = self.carpeta
        url, file_name = self._build_links_and_filename(hour)
        file_path = os.path.join(carpeta, file_name)

        # Ya existe y es válido?
        if self._verify_download(file_path):
            return True

        for attempt in range(max_retries):
            
            try:
                with session.get(url, stream=True, timeout=(10, 300)) as r:
                    if r.status_code == 200:
                        tmp_path = file_path + ".part"
                        with open(tmp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 128):
                                if chunk:
                                    f.write(chunk)
                        os.replace(tmp_path, file_path)
                        return self._verify_download(file_path)
                    else:
                        # Si no es 200, espera antes de reintentar
                        wait = (2 ** attempt )/200 # backoff exponencial: 1,2,4,8,... segundos
                        time.sleep(wait)
                        continue
            except Exception as e:
                print(f"Error en {file_name}: {e}")
                wait = 2 ** attempt
                time.sleep(wait)
                continue

        # Si se acabaron los reintentos
        print(f"No se pudo descargar {file_name} después de {max_retries} intentos")
        return False

    def _open_GRIB(self,file,filtros):
        try:return xr.open_dataset(file,engine='cfgrib',backend_kwargs=filtros).drop('step')
        except:return xr.open_dataset(file,engine='cfgrib',backend_kwargs=filtros) 

    def download(self,MAX_WORKERS=60,MAX_RETRIES=20):
        """
        Descarga completa paralelizada de todas las horas solicitadas

        Parameters
        ------
        MAX_WORKERS (int): optional
            El numero de descargas simultaneas que se ejecutarán, este no es "verdadero" paralelismo, es un mismo nucleo lógico ejecutando multiples tareas a la vez,
            imagina a alguien haciendo trabajo,tesis, maestria y viendo una serie... , asigna cosas sensillas con este metodo, nada muy complejo.
            Aumentar este numero podría acelerar la descarga,aunque el internet de la oficina está topado a 30Mbs(marzo del 2026) y tambien depende de los servers de la noaa(no suele subir de 40Mbs)

        MAX_RETRIES (int): optional
            Ocacionalmente la descarga fallará la primera vez, este número indica cuantos re-intentos se harán antes de conciderar la descarga como fallida completamente.
            con esto no suele fallar, salvo que la coneccion sea muy inestable o algo asi.

        """
        print(' ')
        log_file=self.carpeta+'/log_file.txt'
        #verificacion, si no existe un archivo log o este contiene info desactualizada, se procede con la re-descarga
        if os.path.exists(log_file):
            with open(log_file, 'r') as f_read:content = f_read.read()
            ultima=content[content.find(':')+2:]
            if ultima==self.fecha:
                if len(gb(self.carpeta+'/*'))<100:
                    download=True
                else:download=False
            else:download=True
        else:download=True

        if download:
            
            #se eliminan para prevenir problemas en la descarga o en una consolidacion
            for i in gb(self.carpeta+'/**', recursive=True):
                if os.path.isfile(i):
                    os.remove(i)


            print(f"\n=== Iniciando descarga ===")
            print(f"Workers: {MAX_WORKERS}, Reintentos: {MAX_RETRIES}")
            # Lista de horas a descargar
            hours_to_download = list(self.hours)
            print(f"Total de archivos a descargar: {len(hours_to_download)}")

            
            pendientes, completadas, fallidas=hours_to_download,[],[]

            session = self._crear_sesion(MAX_RETRIES)#crea sesion

            inic = psutil.net_io_counters().bytes_recv;t0 = time.time()#medida para int speed
            print("Estado de la Descarga :")
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_url = {}
                for hour in hours_to_download:
                    # Crear función parcial con session fijo
                    descarga_parcial = partial(self._single_download, session)
                    future = executor.submit(descarga_parcial, hour)
                    future_to_url[future] = hour

                for future in concurrent.futures.as_completed(future_to_url):
                    
                    hour = future_to_url[future]
                    if hour in pendientes:
                        pendientes.remove(hour)
                    ok = future.result()
                    if ok:
                        completadas.append(hour)
                    else:
                        fallidas.append(hour)

                    fin = psutil.net_io_counters().bytes_recv;t1 = time.time()#medida para int speed
                    
                    mbps = ((fin - inic) * 8) / ((t1-t0) * 1_000_000)

                    print(f"completas: {len(completadas)}|pendientes: {len(pendientes)}|fallidos: {len(fallidas)}|Internet speed :{mbps:.2f}         ")
            print(f"\n=== RESUMEN DE DESCARGA ===")
            print(f"Descargas exitosas: {len(completadas)}/{len(completadas)+len(fallidas)}")
            if fallidas:print(f"Se requiere redescargar estas horas:{fallidas}")
            else:print(f"Descargas fallidas: {len(fallidas)}/{len(completadas)+len(fallidas)}")
            with open(log_file, 'w') as f_write:f_write.write(f"Ultima fecha : {self.fecha}")
        else:
            print('Parece que no es necesario re-descargar...')
            time.sleep(3)

    def consolid(self, filtros, name):
        final_file_name=self.carpeta+'/Consolidados/'+f'{name}.nc'
        if os.path.exists(final_file_name):
            print('\n=== Archivo consolidado existente ===')
            return xr.open_dataset(final_file_name)
        
        files=[i for i in gb(self.carpeta+'/*')[:] if not('.idx' in i) and ('.grib2' in i)]
        files.sort()
        files=files[1:]
        print(f'\n=== Se Consolidarán {len(files)} archivos === ')

        consolid = xr.open_mfdataset(
            files,
            engine='cfgrib',
            backend_kwargs=filtros,
            concat_dim='time',
            combine='nested',
            parallel=True
        )
        
        consolid['time']=consolid['valid_time']
        consolid=consolid.drop('valid_time')##
        #consolid=consolid.set_coords('time')
        """if 'tp' in consolid:
            lluvia=np.nan_to_num(consolid.tp.values,0)
            consolid['tp'][1:][::2]= lluvia[1:][::2]-lluvia[::2][:-1]## Explicación (1)"""
        consolid.to_netcdf(final_file_name)
        print(f'Archivo {name} consolidado en {final_file_name}')
        return consolid
# %%
class GFS_down(GeneralDownloader):
    def __init__(self, carpeta, hours=np.arange(0, 7 * 24, 3),extent=(-85, -23, -67, 1),
                 resolution = '25',hour_run = '00',fecha=datetime.now(UTC).replace(tzinfo=None)):
        super().__init__(carpeta, hours, fecha)
        self.resolution = resolution
        self.hour_run = hour_run
        self.extent = extent

    def _build_links_and_filename(self,hora):
            h3 = str(hora).zfill(3)
            min_lon = str(self.extent[0]); min_lat = str(self.extent[1])
            max_lon = str(self.extent[2]); max_lat = str(self.extent[3])
            if self.resolution == '25':
                url = (
                    f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p{self.resolution}.pl"
                    f"?file=gfs.t{self.hour_run}z.pgrb2.0p{self.resolution}.f{h3}"
                    f"&all_lev=on&all_var=on&subregion="
                    f"&leftlon={min_lon}&rightlon={max_lon}&toplat={max_lat}&bottomlat={min_lat}"
                    f"&dir=%2Fgfs.{self.fecha}%2F{self.hour_run}%2Fatmos"
                )
                file_name = f"gfs.t{self.hour_run}z.pgrb2.0p{self.resolution}.f{h3}"
            elif self.resolution == '50':
                url = (
                    f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p{self.resolution}.pl"
                    f"?file=gfs.t{self.hour_run}z.pgrb2full.0p{self.resolution}.f{h3}"
                    f"&all_lev=on&all_var=on&subregion="
                    f"&leftlon={min_lon}&rightlon={max_lon}&toplat={max_lat}&bottomlat={min_lat}"
                    f"&dir=%2Fgfs.{self.fecha}%2F{self.hour_run}%2Fatmos"
                )
                # Usa nombre consistente con la URL
                file_name = f"gfs.t{self.hour_run}z.pgrb2full.0p{self.resolution}.f{h3}"
            elif self.resolution == '1':
                url = (
                    f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{self.resolution}p00.pl"
                    f"?file=gfs.t{self.hour_run}z.pgrb2.{self.resolution}p00.f{h3}"
                    f"&all_lev=on&all_var=on&subregion="
                    f"&leftlon={min_lon}&rightlon={max_lon}&toplat={max_lat}&bottomlat={min_lat}"
                    f"&dir=%2Fgfs.{self.fecha}%2F{self.hour_run}%2Fatmos"
                )
                file_name = f"gfs.t{self.hour_run}z.pgrb2.{self.resolution}p00.f{h3}"
            else:
                raise ValueError(f"Resolución no válida: {self.resolution}")
            return url, file_name
    def consolid(self,filtros,name,MAX_WORKERS= max(cpu_count-1,1)):
        """
        Consolida la descarga para la zona elegida en una bonita carpeta con ese nombre, se generarán archivos idx la primera vez que se lea en algun tipo de nivel/variable y eso podria demorar un poco.
        Al ejecutarse otra vez se leerá de nuevo y no demorará tanto.
        Si el gestor concidera que es necesario redescargar, eliminará los consolidados que estén.

        Parameters
        ----------
        filtros (dict): 
            Filtra la información que deseas cargar desde los archivos GRIB.  
            Los GRIB de la NOAA contienen MUCHÍSIMA información; puedes ver todas las
            variables disponibles en ``gfs/variables.txt``.

            
        name (str): 
            nombre con el que se guardará el consolidado, no debe tener '.nc'
        
        MAX_WORKERS (int): optional 
            Indica el número máximo de hilos para el procesamiento, recordando el _MAX_WORKERS_ del metodo de descarga: este es veradero paralelismo,
            es decir, utilizará procesadores lógicos reales de tu computadora, incrementarlos aumentará bastante la velocidad de procesamiento a cambio de mayor consumo,
            por defecto deja uno libre para que la laptop no explote(se me pausaba la musica cuando no lo hacia T_T).
            Recordar que es importante ejecutar las partes del codigo que contengan este tipo de procesamiento directamente en la consola y NO en jupyter.... a no ser que estes en linux....


            Ejemplos de filtros válidos:


            >>> filtros = {
            ...     "filter_by_keys": {
            ...         "typeOfLevel": "isobaricInhPa",
            ...         "name": [
            ...             "Specific humidity",
            ...             "Relative humidity",
            ...             "Temperature",
            ...             "U component of wind",
            ...             "V component of wind",
            ...             "Geopotential height",
            ...         ]
            ...     }
            ... }

            >>> filtros = {
            ...     "filter_by_keys": {
            ...         "name": [
            ...             "Total Precipitation",
            ...             "Precipitable water",
            ...             "2 metre temperature",
            ...         ]
            ...     }
            ... }

            >>> filtros = {
            ...     "filter_by_keys": {
            ...         "name": [
            ...             "100 metre U wind component",
            ...             "100 metre V wind component",
            ...         ]
            ...     }
            ... }

            >>> filtros = {
            ...     "filter_by_keys": {
            ...         "typeOfLevel": "isobaricInhPa",
            ...         "level": [1000, 850],
            ...         "name": ["Specific humidity", "Temperature"],
            ...     }
            ... }

            >>> filtros = {
            ...     "filter_by_keys": {
            ...         "typeOfLevel": "isobaricInhPa",
            ...         "level": [1000, 850, 500, 300],
            ...         "name": [
            ...             "Geopotential height",
            ...             "Temperature",
            ...             "U component of wind",
            ...             "V component of wind",
            ...         ]
            ...     }
            ... }

        Explicaciones
        ----

        (1):
            El gfs hace algo chistoso con las variables acumulables(alguna que diga 'total' x como la pp) y es que no la acumula cada 3h como sería logico,
            la acumula cada 3 y 6 horas, de modo que el dato en la hora 0(inicializacion del modelo) no existe, la de la hora 3 es el acumulado desde 0 a 3
            el de la hora 6 es el del 0 hasta el 6 y la hora 9 se reinicia siendo el acumulado desde la hora 66 hasta la 9,
            por eso hay que restar los intercalados para tener lo que queremos,

            NOTA: en realidad hay que hacer esto con la gran mayoría las variables que se acumulan, deben decir (accum) en su descripcion en el archivo variables.txt [##gfs/variables.txt]
            pero mientras solo uses la precipitacion total estará todo bien :D.

        (2):
            El gfs y practicamente todos los modelos, en particular los globales, dan su ``valid_time`` en Tiempo universal coordinado (UTC), para compararlo con estaciones
            aquí en Perú es IMPORTANTE restar 5 horas
        
        (3):
            De momento voy a ignorar los archivos de la hora inicial, muchas variables y sobre todo las acumulativas o de promedio rompen criterio opatrones que se usan para graficar
            con esta al ser la inicial, en principio no parece afectar demaciado..
            
        """
        final_file_name=self.carpeta+'/Consolidados/'+f'{name}.nc'
        if os.path.exists(final_file_name):
            print('\n=== Archivo consolidado existente ===')
            return xr.open_dataset(final_file_name)
        
        files=[i for i in gb(self.carpeta+'/*')[:] if not('.idx' in i) and ('.f' in i)]
        files.sort()
        files=files[1:]
        print(f'\n=== Se Consolidarán {len(files)} archivos === ')
        
        consolid = xr.open_mfdataset(
            files,
            engine='cfgrib',
            backend_kwargs=filtros,
            concat_dim='time',
            combine='nested',
            parallel=True
        )
        
        consolid['time']=consolid['valid_time']
        consolid=consolid.drop('valid_time')##
        #consolid=consolid.set_coords('time')
        """if 'tp' in consolid:
            lluvia=np.nan_to_num(consolid.tp.values,0)
            consolid['tp'][1:][::2]= lluvia[1:][::2]-lluvia[::2][:-1]## Explicación (1)"""
        consolid.to_netcdf(final_file_name)
        print(f'Archivo {name} consolidado en {final_file_name}')
        return consolid

class ETA_down(GeneralDownloader):
    def __init__(self, carpeta, hours=np.arange(0, 7 * 24, 1), fecha=datetime.now(UTC).replace(tzinfo=None)-timedelta(days=1)):
        super().__init__(carpeta, hours, fecha)
    def _build_links_and_filename(self,hora):
        """
        Funcion generadora de link y nombre de archivo

        Arguments
        ---
        fecha_forec (datetime.datetime):
            fecha del pronostico (valid_time)
        """
        t_ini=datetime.strptime(self.fecha,'%Y%m%d')
        fecha_forec= t_ini + timedelta(hours=float(hora))
        file_name = f"Eta_ams_08km_{t_ini.strftime('%Y%m%d%H')}_{fecha_forec.strftime('%Y%m%d%H')}.grib2"
        url = f"https://dataserver.cptec.inpe.br/dataserver_modelos/eta/ams_08km/brutos/{t_ini.strftime('%Y/%m/%d/%H')}/{file_name}"
        return url,file_name

class WRF_down(GeneralDownloader):
    def __init__(self, carpeta, hours=np.arange(0, 7*24, 1), fecha=(datetime.now(UTC).replace(tzinfo=None)-timedelta(days=1))):
        super().__init__(carpeta, hours, fecha)
        # Nota: WRF siempre usa corrida 00 UTC (similar al patrón observado)
    
    def _build_links_and_filename(self, hora):
        """
        Genera URL y nombre de archivo para WRF del CPTEC/INPE
        
        Args:
            hora (float): Hora de pronóstico (ej: 0, 3, 6, ...)
        
        Returns:
            tuple: (url, file_name)
        """
        t_ini = datetime.strptime(self.fecha, '%Y%m%d')
        fecha_forec = t_ini + timedelta(hours=float(hora))
        
        file_name = f"WRF_cpt_07KM_{t_ini.strftime('%Y%m%d%H')}_{fecha_forec.strftime('%Y%m%d%H')}.grib2"
        
        url = (f"https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos/"
               f"{t_ini.year}/{t_ini.strftime('%m')}/{t_ini.strftime('%d')}/00/"
               f"{file_name}")
        
        return url, file_name
    
#%%

"""
fix:

crear funcion de consolidacion generalista,o,
construir unas particulares segun el modelo
*
creo que lo mejor seria tener un _open_grib() que se modifique segun el caso

la funcion generica de consolid debe buscar todos los archivos dentro de la carpeta EXEPTO:
el archivo de log, cualquier idx, descartar carpetas
*
"""