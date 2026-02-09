#%%
from flask import Flask, render_template_string, request, jsonify, send_file
import threading
from mapeado import MAPEADOR
from datetime import*
import os
app = Flask(__name__)
#%%
maps=MAPEADOR()
"Zona de configuracion"
# Data por defecto de la rasbery
sensor_data = {
    "temperatura": 10,
    "humedad": 40,
    "presion": None,
    "viento_velocidad": None,
    "viento_direccion": None,
    "precipitacion": None,
    "luz_uv": None,
    "calidad_aire": None,
    "ultima_actualizacion": None,
    "historial": []  # Últimas 24 horas de datos
}
# Estado del modelo WRF
wrf_data = {
    "ultima_actualizacion": None,
    "rodada": None,
    "imagen_path": None,
    "procesando": False,
    "error": None
}
#%%
"HTML->estructura principal"
HTML_TEMPLATE = open('dashboard.html', 'r', encoding='utf-8').read() 
@app.route('/')
def pronostico()->None:
    """Ruta principal que muestra el dashboard de pronóstico del tiempo."""
    return render_template_string(HTML_TEMPLATE)
#%%
"Actualizar ETA"
# Estado del modelo ETA
eta_data = {
    "ultima_actualizacion": None,
    "rodada": None,
    "imagen_path": None,
    "procesando": False,
    "error": None
}
eta_lock = threading.Lock()
@app.route('/api/eta/actualizar', methods=['POST'])
def thr_actualizar_eta()->jsonify:
    global eta_data
    """Endpoint para iniciar la actualización del modelo ETA."""
    def actualizar_eta()->None:
        path,rodada = maps.refresh_eta()
        with eta_lock:
            eta_data["ultima_actualizacion"] = datetime.now().isoformat()
            eta_data["rodada"] = rodada.isoformat() if rodada else None
            eta_data["imagen_path"] = path
            eta_data["procesando"] = False
    thread = threading.Thread(target=actualizar_eta)
    thread.start()
    return jsonify({
        "status": "ok",
        "mensaje": "Actualización del modelo ETA iniciada"})
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
#%%
"Coneccion Rasberipi"
data_lock = threading.Lock()
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

#%%
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
# %%
