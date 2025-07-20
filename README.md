# Mini_Project-Coffe_Data
 This repository is about the mini-project assigment from Machine Learning subjetc, 2025-1S


 ## 🚀 Replicación del Proyecto

### 📥 Clonar el Repositorio

Haz clic en el botón `< > Code`, copia la URL HTTPS del repositorio y ejecuta:

```bash
git clone <URL-del-repositorio>
cd <nombre-del-repo>
```

---

### 🐍 Crear y Activar Entorno Virtual

Este proyecto utiliza **Python 3.11**. Asegúrate de tenerlo instalado antes de continuar.

```bash
python3.11 -m venv .venv

# Activar entorno virtual:
# En macOS / Linux:
source .venv/bin/activate

# En Windows:
.\.venv\Scripts\activate
```

Una vez activado, instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

**NOTA:** Los únicos archivos necesarios de tener en el directorio a parte de los scripts y el .venv, son los 3 archivos XLSX de los datos fuente, los demás archivos se generan solos con la ejecución del `main.py`.

- `main.py es un script que ejecuta los scripts en este orden:
    * `CC_FT_17.py`: Preprocesa el XLSX de FT 17.
    * `CC_FT_18.py`: Preprocesa el XLSX de FT 18.
    * `CC_FT_21.py`: Preprocesa el XLSX de FT 21.
    * `process_and_models`: Procesa detalles finales de los archivos generados por los anteriores scripts, y genera las pipelines con los modelos y reportes.
