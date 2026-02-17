#%%
from flask import Flask, render_template_string
from mapeado import MAPEADOR
from comunicacion import ControlModelos,ControlRasberi,ControlCamara
from datetime import*
app = Flask(__name__)
#%%
"Zona de configuracion"
maps=MAPEADOR()# inicializa clase de gestión de descargas e imágenes

control_mapas = ControlModelos()# inicializa clase para el control de la actualizacion de modelos en el flask
control_mapas.init_app(app)#settea las comunicaciones

control_rasby = ControlRasberi()
control_rasby.init_app(app)

control_cam = ControlCamara()
control_cam.init_app(app)
#%%
"HTML->estructura principal"
HTML_TEMPLATE = open('dashboard.html', 'r', encoding='utf-8').read() 
@app.route('/')
def pronostico()->None:
    """Ruta principal que muestra el dashboard de pronóstico del tiempo."""
    return render_template_string(HTML_TEMPLATE)
#%%
if __name__ == '__main__':
    print("=" * 60)
    print("🌤️  HANAQ - UNALM Dashboard")
    print("=" * 60)
    print("📍 Servidor iniciado en: http://127.0.0.1:5000")
    print("=" * 60)
    print("Presiona Ctrl+C para detener el servidor")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000)
    #app.run(debug=True)
# %%
"""
"   📡 Endpoints API para Raspberry Pi:"
"   POST /api/sensores  - Enviar datos de sensores"
"   GET  /api/sensores  - Obtener últimos datos"
"   GET  /api/historial - Obtener historial de datos"
"   GET  /api/estado    - Verificar conexión"""