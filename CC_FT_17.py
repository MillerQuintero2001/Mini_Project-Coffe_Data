# # Preprocesamiento del Archivo CC FT 17
# Este archivo es del formato de control de calidad del café trillado

import pandas as pd
import numpy as np
import os
import json
from unidecode import unidecode


archivos = os.listdir()
# Obtenemos el archivo que comienza con 'CC FT 17'
for archivo in archivos:
    if archivo.startswith('CC FT 17'):
        archivo_xlsx = archivo
        break
print(f"Archivo encontrado: {archivo_xlsx}")


# Abrimos el archivo XLSX
df = pd.read_excel(archivo_xlsx)
# Veamos las hojas del archivo
hojas = pd.ExcelFile(archivo_xlsx).sheet_names
print(f"Hojas disponibles: {hojas}")


# ## CONTROL CALIDAD CAFE TRILLADO J

# Seleccionamos la hoja 1
df = pd.read_excel(archivo_xlsx, sheet_name=hojas[0])


# Observando del dataframe, vemos que hay que omitir las filas del encabezado
df = pd.read_excel(archivo_xlsx, sheet_name=hojas[0], skiprows=5)


# El siguiente paso será separar el encabezado de las columnas y las 2 primeras en un dataframe, y el resto en otro dataframe, para procesar las columnas.

# Creamos una copia del DataFrame original para trabajar con él
df_copy = df.copy()
# Separamos el encabezado de las columnas y las 2 primeras filas
header = df_copy.iloc[0:2, :].copy()
# Creamos un nuevo DataFrame sin las filas del encabezado
df_cleaned = df_copy.iloc[2:, :].reset_index(drop=True).copy()

# Limpiamos de memoria la copia inicial del DataFrame
del df_copy


columns_conflict = ['%H', 'Unnamed: 5', 'MALLAS', 'Unnamed: 7', 'PUNTAJE', 'Unnamed: 11']
new_columns = ['%H', '%H_C/NC', '#MALLAS', 'MALLAS_C/NC', 'PUNTAJE_N°', 'PUNTAJE_C/NC']
# Creamos un diccionario de mapeo
column_mapping = dict(zip(columns_conflict, new_columns))
for key, value in column_mapping.items():
    print(f"Renombrando '{key}' a '{value}'")


# Renombramos las columnas del DataFrame
df_cleaned.rename(columns=column_mapping, inplace=True)

# Limpiamos los nombres de las columnas
df_cleaned.columns = df_cleaned.columns.str.strip()
# Hacemos lo mismo para los valores de las columnas 'object'
for col in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[col] = df_cleaned[col].astype(str).str.strip()



# En la cola del dataframe se aprecian un montón de valores nulos, por cuenta de una información que hay al final del .xlsx, vamos a dropear eso


# Dropeamos las filas con una cantidad de valores nulos igual a la cantidad de columnas
num_cols = len(df_cleaned.columns)
# Si la fila tiene nulos mayor o igual a num_cols, la eliminamos
df_cleaned = df_cleaned.dropna(thresh=num_cols)


# Veamos la cantidad de valores únicos por columna
for col in df_cleaned.columns:
    print(f"{col}: {df_cleaned[col].nunique()} valores únicos")



# Veamos los valores únicos de las columnas con 25 valores únicos o menos
for col in df_cleaned.columns:
    if df_cleaned[col].nunique() <= 25:
        print(f"{col}: {df_cleaned[col].unique()}")
        print('---' * 20)


# **Las columnas con el sufjio 'C/NC'** son columnas para indicar conformidad respecto a la variable familiar, y en todas estas vemos que se cumple la conformidad ya que el valor único es el 'C', por ello son descartables. Igualmente en las columnas **'VERIFICACIÓN FISICA CAFÉ TOSTADO'** y **'LIBERACIÓN DE LOTE'** todos los valores son iguales, y no tienen relación ni aparecen en los datos de los demás archivos .xlsx, por lo tanto también se descartan.
# 
# La columna **'RESPONSABLE'** a pesar de tener un valor único, si es relacionable en contexto con los demás archivos, por ello no se descarta.


# Observando esto es posible descartar las siguientes columnas:
columns_to_drop = ['%H_C/NC', 'MALLAS_C/NC', 'VERIFICACIÓN FISICA CAFÉ TOSTADO', 
                'PUNTAJE_C/NC', 'LIBERACIÓN DE LOTE']
df_cleaned = df_cleaned.drop(columns=columns_to_drop)



# Veamos cantidad de valores nulos por columna
nulos_por_columna = df_cleaned.isnull().sum()
print("Cantidad de valores nulos por columna:")
print(nulos_por_columna)


# Verificamos nulos y dtypes


# Las siguientes columnas NO son numéricas y deben ser numéricas
numeric_columns = ['%H', '#MALLAS', 'PUNTAJE_N°']
# Veamos los valores únicos de estas columnas
for col in numeric_columns:
    print(f"{col}: {df_cleaned[col].unique()}")
    print('---' * 20)

# Reemplzar comas por punto en 'PUNTAJE_N°', y casteamos a float
df_cleaned['PUNTAJE_N°'] = df_cleaned['PUNTAJE_N°'].str.replace(',', '.').astype(float)
df_cleaned['%H'] = df_cleaned['%H'].astype(float)
df_cleaned['#MALLAS'] = df_cleaned['#MALLAS'].astype(int)


# Veamos la fila donde está el valor nulo de 'PUNTAJE_N°'
df_cleaned[df_cleaned['PUNTAJE_N°'].isnull()]


# Veamos las filas donde la columna 'DENOMINACIÓN/     MARCA' es 'Don Reinaldo'
df_cleaned[df_cleaned['DENOMINACIÓN/     MARCA'] == 'Don Reinaldo']
# Vemos que **no hay más filas con la referencia de marca 'Don Reinaldo'**, por tal motivo no podemos hacer una imputación en contexto, sino que se debe hacer general.


# Imputamos el valor faltante de 'PUNTAJE_N°' con la media general
df_cleaned['PUNTAJE_N°'] = df_cleaned['PUNTAJE_N°'].fillna(df_cleaned['PUNTAJE_N°'].mean())


# ### Guardado del Dataframe

os.makedirs('CC_FT_17_cleaned', exist_ok=True)
# Guardamos el DataFrame limpio como un csv
df_cleaned.to_csv('CC_FT_17_cleaned/CC_FT_17_sheet_1.csv', index=False)
print("DataFrame limpio guardado como 'CC_FT_17_cleaned/CC_FT_17_sheet_1.csv'")




# ## Sheet2

# Seleccionamos la hoja 'Sheet2'
df = pd.read_excel(archivo_xlsx, sheet_name=hojas[1])

# Omitimos las primeras 5 filas del encabezado
df = pd.read_excel(archivo_xlsx, sheet_name=hojas[1], skiprows=5)
# Mostramos las primeras filas del DataFrame



# Creamos una copia del DataFrame original para trabajar con él
df_copy = df.copy()
# Separamos el encabezado de las columnas y las 2 primeras filas
header = df_copy.iloc[0:2, :].copy()
# Creamos un nuevo DataFrame sin las filas del encabezado
df_cleaned = df_copy.iloc[2:, :].reset_index(drop=True).copy()

# Limpiamos de memoria la copia inicial del DataFrame
del df_copy



# Es un caso similar a la anterior hoja
columns_conflict = ['%H', 'Unnamed: 5', 'MALLAS', 'Unnamed: 7', 'PUNTAJE', 'Unnamed: 11']
new_columns = ['%H', '%H_C/NC', '#MALLAS', 'MALLAS_C/NC', 'PUNTAJE_N°', 'PUNTAJE_C/NC']
# Creamos un diccionario de mapeo
column_mapping = dict(zip(columns_conflict, new_columns))
for key, value in column_mapping.items():
    print(f"Renombrando '{key}' a '{value}'")


# Renombramos las columnas del DataFrame
df_cleaned.rename(columns=column_mapping, inplace=True)

# Limpiamos los nombres de las columnas
df_cleaned.columns = df_cleaned.columns.str.strip()
# Hacemos lo mismo para los valores de las columnas 'object'
for col in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[col] = df_cleaned[col].astype(str).str.strip()


# Dropeamos las filas con una cantidad de valores nulos igual a la cantidad de columnas
num_cols = len(df_cleaned.columns)
# Si la fila tiene nulos mayor o igual a num_cols, la eliminamos
df_cleaned = df_cleaned.dropna(thresh=num_cols)


# Veamos la cantidad de valores únicos por columna
for col in df_cleaned.columns:
    print(f"{col}: {df_cleaned[col].nunique()} valores únicos")


# Veamos los valores únicos de las columnas con 27 valores únicos o menos
for col in df_cleaned.columns:
    if df_cleaned[col].nunique() <= 27:
        print(f"{col}: {df_cleaned[col].unique()}")
        print('---' * 20)


df_cleaned = df_cleaned.replace({'RESPONSABLE': {'Ac': 'AC'}})


# Contemos los 'nan' que hay en cada columna
for col in df_cleaned.columns:
    print(f"{col}: {df_cleaned[col][df_cleaned[col] == 'nan'].count()} valores nulos")



# Veamos las filas que tienen valores 'nan'
df_cleaned[df_cleaned['PUNTAJE_N°']=='nan']


# A pesar de la presencia de unos valores 'nan' en una fila, esto es en una muestra no significativa, por lo cual, nuevamente, **las columnas con el sufjio 'C/NC'** son columnas para indicar conformidad respecto a la variable familiar, y en todas estas vemos que se cumple la conformidad ya que el valor único es el 'C', por ello son descartables. Igualmente en las columnas **'VERIFICACIÓN FISICA CAFÉ TOSTADO'** y **'LIBERACIÓN DE LOTE'** todos los valores son iguales, y no tienen relación ni aparecen en los datos de los demás archivos .xlsx, por lo tanto también se descartan.
# 
# La columna **'RESPONSABLE'** a pesar de tener un valor único, si es relacionable en contexto con los demás archivos, por ello no se descarta.


# Observando esto es posible descartar las siguientes columnas:
columns_to_drop = ['%H_C/NC', 'MALLAS_C/NC', 'VERIFICACIÓN FISICA CAFÉ TOSTADO', 
                'PUNTAJE_C/NC', 'LIBERACIÓN DE LOTE']
df_cleaned = df_cleaned.drop(columns=columns_to_drop)


# Veamos de nuevo la cantidad de valores únicos por columna
for col in df_cleaned.columns:
    print(f"{col}: {df_cleaned[col].nunique()} valores únicos")

# Veamos los valores únicos de las columnas con 27 valores únicos o menos
for col in df_cleaned.columns:
    if df_cleaned[col].nunique() <= 27:
        print(f"{col}: {df_cleaned[col].unique()}")
        print('---' * 20)

# Reemplzar comas por punto en 'PUNTAJE_N°', y casteamos a float
df_cleaned['PUNTAJE_N°'] = df_cleaned['PUNTAJE_N°'].str.replace(',', '.').astype(float)
df_cleaned['%H'] = df_cleaned['%H'].str.replace(',', '.').astype(float)
df_cleaned['#MALLAS'] = df_cleaned['#MALLAS'].astype(int)



# Verificamos nulos y dtypes

# Veamos los valores únicos de las columnas con 27 valores únicos o menos
for col in df_cleaned.columns:
    if df_cleaned[col].nunique() <= 27:
        print(f"{col}: {df_cleaned[col].unique()}")
        print('---' * 20)


# Veamos la fila donde 'RESPONSABLE' es 'nan'
df_cleaned[df_cleaned['RESPONSABLE'] == 'nan']


for col in df_cleaned.columns:
    print(f"'{col}'")


# Llenamos el NaN de 'PUNTAJE_N°' con la media de los valores
# de puntaje en las filas donde la cateogría
# 'DENOMINACIÓN/     MARCA' = 'Gesha Villabernarda'
mean = df_cleaned['PUNTAJE_N°'][df_cleaned['DENOMINACIÓN/     MARCA'] == 'Gesha Villabernarda'].mean()
# Imputamos el valor faltante
df_cleaned['PUNTAJE_N°'] = df_cleaned['PUNTAJE_N°'].fillna(mean)


# Ahora llenamos el 'nan' de 'NOTAS DE CATACIÓN' con la clase mayoritaria
# de las filas donde la cateogría 'DENOMINACIÓN/     MARCA' = 'Gesha Villabernarda'
mode = df_cleaned['NOTAS DE CATACIÓN'][df_cleaned['DENOMINACIÓN/     MARCA'] == 'Gesha Villabernarda'].mode()[0]
df_cleaned.loc[df_cleaned['NOTAS DE CATACIÓN'] == 'nan', 'NOTAS DE CATACIÓN'] = mode

# Ahora con la columna 'RESPONSABLE' hacemos algo similar
mode = df_cleaned['RESPONSABLE'][df_cleaned['DENOMINACIÓN/     MARCA'] == 'Gesha Villabernarda'].mode()[0]
df_cleaned.loc[df_cleaned['RESPONSABLE'] == 'nan', 'RESPONSABLE'] = mode


# ### Guardado del Dataframe

# Guardamos el DataFrame limpio en un .csv
df_cleaned.to_csv('CC_FT_17_cleaned/CC_FT_17_sheet_2.csv', index=False)
print("DataFrame limpio guardado como 'CC_FT_17_cleaned/CC_FT_17_sheet_2.csv'")


# ## Verificación de Integridad

# Leemos los 2 archivos .csv que hemos guardado
df_sheet_1 = pd.read_csv('CC_FT_17_cleaned/CC_FT_17_sheet_1.csv')
df_sheet_2 = pd.read_csv('CC_FT_17_cleaned/CC_FT_17_sheet_2.csv')


# Verificar si TODOS los valores cumplen con el formato
cumple_formato = df_sheet_1['LOTE'].str.contains(r'^\d{2}-\d{6}$', na=False).all()
print(f"¿Todos los lotes cumplen con el formato? {cumple_formato}")

# Mostrar los valores que NO cumplen con el formato
valores_incorrectos = df_sheet_1[~df_sheet_1['LOTE'].str.contains(r'^\d{2}-\d{6}$', na=False)]
if not valores_incorrectos.empty:
    print(f"\nValores que NO cumplen con el formato (total: {len(valores_incorrectos)}):")
    print(valores_incorrectos['LOTE'].unique())
else:
    print("\nTodos los valores cumplen con el formato.")



# Para valores con múltiples guiones: mantener solo el primer guión
df_sheet_1['LOTE'] = df_sheet_1['LOTE'].str.replace(r'^(\d{2})-(.*)$', lambda m: m.group(1) + '-' + m.group(2).replace('-', ''), regex=True)


# Verificar si TODOS los valores cumplen con el formato
cumple_formato = df_sheet_1['LOTE'].str.contains(r'^\d{2}-\d{6}$', na=False).all()
print(f"¿Todos los lotes cumplen con el formato? {cumple_formato}")

# Mostrar los valores que NO cumplen con el formato
valores_incorrectos = df_sheet_1[~df_sheet_1['LOTE'].str.contains(r'^\d{2}-\d{6}$', na=False)]
if not valores_incorrectos.empty:
    print(f"\nValores que NO cumplen con el formato (total: {len(valores_incorrectos)}):")
    print(valores_incorrectos['LOTE'].unique())
else:
    print("\nTodos los valores cumplen con el formato.")


# Verificar si TODOS los valores cumplen con el formato, pero ahora en la sheet 2
cumple_formato = df_sheet_2['LOTE'].str.contains(r'^\d{2}-\d{6}$', na=False).all()
print(f"¿Todos los lotes cumplen con el formato? {cumple_formato}")

# Mostrar los valores que NO cumplen con el formato
valores_incorrectos = df_sheet_2[~df_sheet_2['LOTE'].str.contains(r'^\d{2}-\d{6}$', na=False)]
if not valores_incorrectos.empty:
    print(f"\nValores que NO cumplen con el formato (total: {len(valores_incorrectos)}):")
    print(valores_incorrectos['LOTE'].unique())
else:
    print("\nTodos los valores cumplen con el formato.")


df_sheet_1.columns == df_sheet_2.columns


df_sheet_1.shape, df_sheet_2.shape


# ## Unión

# Juntamos el dataframe hoja 2 debajo del de la hoja 1
df = pd.concat([df_sheet_1, df_sheet_2], ignore_index=True)

# Vamos a renombrar columnas según este diccionario
column_mapping = {
    'DENOMINACIÓN/     MARCA': 'MARCA',
    'NOTAS DE CATACIÓN': 'NOTAS',
    'PUNTAJE_N°': 'PUNTAJE',
    'RESPONSABLE': 'TOSTADOR'}

# Renombramos las columnas del DataFrame
df.rename(columns=column_mapping, inplace=True)


# Guardamos este nuevo DataFrame combinado
df.to_csv('CC_FT_17_cleaned.csv', index=False)
print("DataFrame combinado guardado en ./CC_FT_17_cleaned.csv")

