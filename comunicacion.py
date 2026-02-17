#%%
from flask import Flask, render_template_string,jsonify,request,send_file
import threading
import os
from datetime import datetime
from mapeado import MAPEADOR
from PIL import Image

class ControlModelos:

    def __init__(self,modelos=['eta','wrf']):
        """
        Args:
            mapeador: Instancia de MAPEADOR() con métodos refresh_eta(), refresh_wrf(), etc.
        """
        self.mapeador = MAPEADOR()
        MODELOS={}
        for i in modelos:
            MODELOS[i]={
            "ultima_actualizacion": None,
            "rodada": None,
            "imagen_path": None,
            "procesando": False,
            "error": None,
            "lock":threading.Lock()
        }
        self.modelos = MODELOS
    
    def init_app(self, app):
        """Registra todas las rutas en la app de Flask"""
        
        # 1. Endpoint para obtener estado de cualquier modelo
        @app.route('/api/<nombre_modelo>/estado', methods=['GET'])
        def estado_modelo(nombre_modelo):
            return self._obtener_estado(nombre_modelo)
        # 2. Endpoint para actualizar cualquier modelo
        @app.route('/api/<nombre_modelo>/actualizar', methods=['POST'])
        def actualizar_modelo(nombre_modelo):
            return self._iniciar_actualizacion(nombre_modelo)
        # 3. Endpoint para obtener mapa/imagen
        @app.route('/api/<nombre_modelo>/mapa', methods=['GET'])
        def obtener_mapa(nombre_modelo):
            return self._servir_mapa(nombre_modelo)
    
    def _obtener_estado(self, nombre_modelo):
        """Obtiene el estado actual de un modelo"""
        if nombre_modelo not in self.modelos:
            return jsonify({"error": f"Modelo '{nombre_modelo}' no encontrado"}), 404
        
        with self.modelos[nombre_modelo]['lock']:
            
            datos = self.modelos[nombre_modelo].copy()
            datos.pop("lock", None)  # Usa pop para evitar error si no existe
            return jsonify(datos)
    
    def _iniciar_actualizacion(self, nombre_modelo):
        """Inicia la actualización de un modelo en segundo plano"""
        if nombre_modelo not in self.modelos:
            return jsonify({"error": f"Modelo '{nombre_modelo}' no encontrado"}), 404
        
        # Verificar si ya se está procesando
        with self.modelos[nombre_modelo]['lock']:   
            if self.modelos[nombre_modelo]["procesando"]:
                return jsonify({"error": f"Modelo '{nombre_modelo}' ya se está actualizando"}), 409
            
            # Marcar como procesando
            self.modelos[nombre_modelo]["procesando"] = True
            self.modelos[nombre_modelo]["error"] = None
        
        # Iniciar en segundo plano
        thread = threading.Thread(
            target=self._actualizar_modelo_background,
            args=(nombre_modelo,)
        )
        thread.start()
        
        return jsonify({
            "status": "ok",
            "mensaje": f"Actualización de '{nombre_modelo}' iniciada"
        })
    
    def _actualizar_modelo_background(self, nombre_modelo):
        """Actualiza el modelo en segundo plano (en un thread separado)"""
        try:
            # Llama al mapeador correspondiente según el modelo
            try :
                path, rodada = self.mapeador.refresh_model(nombre_modelo)
            except KeyError :
                raise ValueError(f"Método 'refresh_{nombre_modelo}' no encontrado en el mapeador")
            
            # Actualizar estado
            with self.modelos[nombre_modelo]['lock']:
                self.modelos[nombre_modelo]["imagen_path"] = path
                self.modelos[nombre_modelo]["rodada"] = rodada.isoformat() if rodada else None
                self.modelos[nombre_modelo]["ultima_actualizacion"] = datetime.now().isoformat()
                self.modelos[nombre_modelo]["procesando"] = False
                self.modelos[nombre_modelo]["error"] = None
            
            print(f"✅ Modelo '{nombre_modelo}' actualizado correctamente")
            
        except Exception as e:
            # Manejar error
            with self.modelos[nombre_modelo]['lock']:
                self.modelos[nombre_modelo]["procesando"] = False
                self.modelos[nombre_modelo]["error"] = str(e)
            
            print(f"❌ Error actualizando '{nombre_modelo}': {e}")
    
    def _servir_mapa(self, nombre_modelo):
        """Sirve la imagen del mapa si existe"""
        if nombre_modelo not in self.modelos:
            return jsonify({"error": f"Modelo '{nombre_modelo}' no encontrado"}), 404
        
        with self.modelos[nombre_modelo]['lock']:
            img_path = self.modelos[nombre_modelo]["imagen_path"]
        
        if img_path and os.path.exists(img_path):
            return send_file(img_path, mimetype='image/png')
        else:
            return jsonify({
                "error": f"Mapa no disponible para '{nombre_modelo}'. Ejecute /api/{nombre_modelo}/actualizar primero."
            }), 404
#%%
class ControlRasberi:
    def __init__(self)->None:
        self.sensor = {
            "time" : None,
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
    def init_app(self,app):

        self.data_lock = threading.Lock()
        @app.route('/api/sensores', methods=['POST'])
        def recibir_datos_sensores():
            try:
                datos = request.get_json()
                print(f"📡 DATOS RECIBIDOS: {datos}")
                
                with self.data_lock:
                    for key in datos:
                        if key in self.sensor:
                            self.sensor[key] = datos[key]
                    self.sensor["ultima_actualizacion"] = datetime.now().isoformat()
                    
                    # Guardar en historial (máximo 288 registros = 24 horas con datos cada 5 min)
                    registro = {
                        **datos,
                        "timestamp": self.sensor["ultima_actualizacion"]
                    }
                    self.sensor["historial"].append(registro)
                    if len(self.sensor["historial"]) > 288:
                        self.sensor["historial"].pop(0)
                
                return jsonify({
                    "status": "ok",
                    "mensaje": "Datos recibidos correctamente",
                    "timestamp": self.sensor["ultima_actualizacion"]
                }), 200
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "mensaje": str(e)
                }), 400
        @app.route('/api/sensores', methods=['GET'])
        def obtener_datos_sensores():
            """Endpoint para obtener los últimos datos de sensores."""
            with self.data_lock:
                return jsonify(self.sensor)
        @app.route('/api/historial', methods=['GET'])
        def obtener_historial():
            """Endpoint para obtener el historial de datos."""
            with self.data_lock:
                return jsonify(self.sensor["historial"])
        @app.route('/api/estado', methods=['GET'])
        def estado_conexion():
            """Verifica el estado de la conexión con la Raspberry Pi."""
            with self.data_lock:
                ultima = self.sensor["ultima_actualizacion"]
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
        
#%%
class ControlCamara:
    def __init__(self):
        self.camara={
            'imagen_path':None,
            'ultima_actualizacion':None
        }
    def init_app(self,app,folder="imagenes_cielo"):

        self.data_lock = threading.Lock()
        os.makedirs(folder, exist_ok=True)
        self.folder=folder
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
                ruta_completa = os.path.join(self.folder, nombre_archivo)
                archivo.save(ruta_completa)
                
                with self.data_lock:
                    self.camara["imagen_path"] = ruta_completa
                    self.camara["ultima_actualizacion"] = datetime.now().isoformat()
                
                print(f"📷 Imagen del cielo recibida: {nombre_archivo}")

                url_imagen = f"/api/cielo/imagen?t={int(datetime.now().timestamp()*1000)}"
                return jsonify({
                    "status": "ok",
                    "mensaje": "Imagen recibida correctamente",
                    "archivo": nombre_archivo,
                    "timestamp": self.camara["ultima_actualizacion"],
                    "url":url_imagen
                }), 200
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route('/api/cielo/imagen', methods=['GET'])
        def obtener_imagen_cielo():
            """Obtiene la imagen completa del cielo."""
            with self.data_lock:
                img_path = self.camara.get("imagen_path")
            
            if img_path and os.path.exists(img_path):
                return send_file(img_path, mimetype='image/jpeg')
            else:
                return jsonify({"error": "Imagen no disponible"}), 404

        @app.route('/api/cielo/cuadrante/<direccion>', methods=['GET'])
        def obtener_cuadrante_cielo(direccion):
            """Obtiene un cuadrante específico de la imagen del cielo (norte, sur, este, oeste)."""
            
            with self.data_lock:
                img_path = self.camara.get("imagen_path")
            
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
            with self.data_lock:
                return jsonify(self.camara)

#%%
if __name__ =='__main__':
    a=ControlModelos()
