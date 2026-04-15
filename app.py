#%%
from glob import glob as gb

from flask import Flask, render_template_string, Response,send_file
from comunicacion import ControlModelos,ControlRasberi,ControlCamara
from datetime import*

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Solo muestra errores, no las peticiones GET

app = Flask(__name__)
#%%
"Zona de configuracion"

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
    # Devolver el HTML tal cual sin procesar por Jinja para evitar
    # errores con los literales '{{' usados por React/JSX en la plantilla.
    return Response(HTML_TEMPLATE, mimetype='text/html')
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