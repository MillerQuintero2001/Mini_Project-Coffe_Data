import subprocess
import sys

# Lista de tus scripts en el orden deseado
# Asegúrate de que estos scripts estén en el mismo directorio que este ejecutor_simple.py
scripts_a_ejecutar = [
    "CC_FT_17.py",
    "CC_FT_18.py",
    "CC_FT_21.py",
    "process_and_models.py"
]

print("Iniciando la ejecución secuencial de los scripts...\n")

for script in scripts_a_ejecutar:
    print(f"🚀 Ejecutando: {script}")
    try:
        # Usamos sys.executable para asegurarnos de usar el intérprete de Python correcto
        # check=True: si un script falla, detiene la ejecución y muestra un error
        subprocess.run([sys.executable, script], check=True)
        print(f"✅ {script} se ejecutó con éxito.\n")
    except subprocess.CalledProcessError:
        print(f"❌ Error: {script} falló. La ejecución se ha detenido.\n")
        sys.exit(1) # Sale del script principal con un código de error
    except FileNotFoundError:
        print(f"⚠️ Error: No se encontró el script '{script}'. Asegúrate de que el nombre y la ruta sean correctos.\n")
        sys.exit(1) # Sale del script principal con un código de error

print("🎉 Todos los scripts se ejecutaron exitosamente.")