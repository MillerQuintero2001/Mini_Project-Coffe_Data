# # Preprocesamiento del Archivo CC FT 18
# Este archivo es del formato de control de tostión


import pandas as pd
import numpy as np
import os

archivos = os.listdir()
# Obtenemos el archivo que comienza con 'CC FT 17'
for archivo in archivos:
    if archivo.startswith('CC FT 18'):
        archivo_xlsx = archivo
        break
print(f"Archivo encontrado: {archivo_xlsx}")



# Abrimos el archivo XLSX
df = pd.read_excel(archivo_xlsx)
# Veamos las hojas del archivo
hojas = pd.ExcelFile(archivo_xlsx).sheet_names
print(f"Hojas disponibles: {hojas}")


# ## TOSTIÓN JERICÓ L (Hoja 1)
# Comenzamos preprocesando la primera de las hojas

# Seleccionamos la hoja 1, ya sabmos que siempre omitimos las 5 primeras filas
df = pd.read_excel(archivo_xlsx, sheet_name=hojas[0], skiprows=5)

# Dropeamos la columna de 'Observaciones'
df.drop(columns=['Observaciones '], inplace=True)


# Veamos la cantidad de valores únicos por columna
for col in df.columns:
    print(f'{col}: {df[col].nunique()} valores únicos')

# Veamos los valores únicos de las columnas con 15 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 15:
        print(f"Valores únicos en la columna '{col}':")
        print(df[col].unique())
        print('----'*20)


# Vamos a borrar los espacio vacíos al principio y al final de los nombres de las columnas
df.columns = df.columns.str.strip()


# Hacemos lo mismo de borrar espacios vacíos al principio y al final, pero ahora en las columnas de tipo 'object'
# Convertir a string primero y luego aplicar strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Veamos de nuevo los valores únicos de las columnas con 15 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 15:
        print(f'{col}: {df[col].unique()}')


# En la columna 'Perfil' hacemos un reemplazo
df['Perfil'] = df['Perfil'].replace({
    'Espressso': 'Espresso',
    'Filtrados': 'Filtrado',
})

df['Origen'] = df['Origen'].replace({
    'Herrra': 'Herrera',
})

# Veamos de nuevo los valores únicos de las columnas con 15 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 15:
        print(f'{col}: {df[col].unique()}')


# Veamos de nuevo la cantidad de valores únicos por columna
for col in df.columns:
    print(f'{col}: {df[col].nunique()} valores únicos')


# Imprimimos los valores únicos de las columnas con 25 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 25:
        print(f'{col}: {df[col].unique()}')


# De la columnas 'Tiempo de tueste' que contiene valores en formato MM:SS:mmSS, los convertimos a segundos
def convertir_tiempo_a_segundos(tiempo):
    if pd.isna(tiempo):
        return np.nan
    partes = tiempo.split(':')
    if len(partes) == 3:  # MM:SS:mmSS
        minutos, segundos, milisegundos = partes
        return int(minutos) * 60 + int(segundos) + int(milisegundos) / 1000
    else:
        return np.nan



# Convertimos la columna 'Tiempo de tueste' a segundos
df['Tiempo de tueste'] = df['Tiempo de tueste'].apply(convertir_tiempo_a_segundos).astype(int)



# Separamos la columna 'Temp. De inicio y final' en dos columnas
df[['Temp. Inicio', 'Temp. Final']] = df['Temp. De inicio y final'].str.split('/', expand=True)

# Eliminamos el símbolo '°' de las nuevas columnas y convertimos a numérico

# Solo si tiene símbolo '°'
df['Temp. Inicio'] = df['Temp. Inicio'].str.replace('°', '', regex=False).astype(float)
df['Temp. Final'] = df['Temp. Final'].str.replace('°', '', regex=False).astype(float)

# Eliminamos la columna original
df.drop(columns=['Temp. De inicio y final'], inplace=True)



# Guardamos el dataframe ya preprocesado en un .csv
os.makedirs('CC_FT_18_cleaned', exist_ok=True)
df.to_csv('CC_FT_18_cleaned/CC_FT_18_sheet_1.csv', index=False)
print("DataFrame guardado en 'CC_FT_18_cleaned/CC_FT_18_sheet1.csv'")



# ## TOSTIÓN JERICÓ (Hoja 2)


# Leemos la hoja 2 del archivo original
df = pd.read_excel(archivo_xlsx, sheet_name=hojas[1], skiprows=5)
# Mostramos las primeras filas del DataFrame de la hoja 2

for col in df.columns:
    print(f"'{col}'")
df.drop(columns=['Observaciones '], inplace=True)

# Veamos la cantidad de valores únicos por columna
for col in df.columns:
    print(f'{col}: {df[col].nunique()} valores únicos')

# Imprimimos los valores únicos de las columnas con 25 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 25:
        print(f'{col}: {df[col].unique()}')
        print('----'*20)


# Borramos espacios vacíos
df.columns = df.columns.str.strip()
# Convertir a string primero y luego aplicar strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()


# En columna 'Tostador' hacemos un reemplazo
df['Tostador'] = df['Tostador'].replace({'A C': 'AC'})
df['Beneficio'] = df['Beneficio'].replace({'lavado': 'Lavado'})

# De nuevo valores únicos de las columnas con 25 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 25:
        print(f'{col}: {df[col].unique()}')
        print('----'*20)

df['Tiempo de tueste'] = df['Tiempo de tueste'].str.replace('.', ':', regex=False)

# Tiempo de tueste' contiene valores en formato MM:SS, los convertimos a segundos
def convertir_tiempo_a_segundos(tiempo):
    if pd.isna(tiempo):
        return np.nan
    partes = tiempo.split(':')
    if len(partes) == 2:  # MM:SS
        minutos, segundos = partes
        return int(minutos) * 60 + int(segundos)
    else:
        return np.nan


# Convertimos la columna 'Tiempo de tueste' a segundos
df['Tiempo de tueste'] = df['Tiempo de tueste'].apply(convertir_tiempo_a_segundos)

# Separamos la columna 'Temp. De inicio y final' en dos columnas
df[['Temp. Inicio', 'Temp. Final']] = df['Temp. De inicio y final'].str.split('-', expand=True)

# Eliminamos el símbolo '°' de las nuevas columnas y convertimos a numérico

# Solo si tiene símbolo '°'
df['Temp. Inicio'] = df['Temp. Inicio'].str.replace('°', '', regex=False).astype(float)
df['Temp. Final'] = df['Temp. Final'].str.replace('°', '', regex=False).astype(float)

# Eliminamos la columna original
df.drop(columns=['Temp. De inicio y final'], inplace=True)



# ### Guardado del dataframe

# Con esto tenemos el DataFrame de la hoja 2 preprocesado
# Guardamos el dataframe ya preprocesado en un .csv
os.makedirs('CC_FT_18_cleaned', exist_ok=True)
df.to_csv('CC_FT_18_cleaned/CC_FT_18_sheet_2.csv', index=False)
print("DataFrame guardado en 'CC_FT_18_cleaned/CC_FT_18_sheet_2.csv'")


# ## Verificación de Integridad

# Leemos los 2 DataFrames guardados
df_sheet_1 = pd.read_csv('CC_FT_18_cleaned/CC_FT_18_sheet_1.csv')
df_sheet_2 = pd.read_csv('CC_FT_18_cleaned/CC_FT_18_sheet_2.csv')

df_sheet_1.columns == df_sheet_2.columns


# ## Unión

# Juntamos el dataframe hoja 2 debajo del de la hoja 1
df = pd.concat([df_sheet_1, df_sheet_2], ignore_index=True)

# Reemplazamos las columnas por su versión en mayúsculas
df.columns = df.columns.str.upper()


# Veamos los valores únicos de la columna 'BENEFICIO' Y 'PROCESO'
print(df['BENEFICIO'].unique())
print(df['PROCESO'].unique())


# Agruapamos por 'BENEFICIO' y veamos el 'PROCESO'
df.groupby('BENEFICIO')['PROCESO'].unique()
df.groupby('PROCESO')['BENEFICIO'].value_counts()
# Al inverso
df.groupby('PROCESO')['BENEFICIO'].value_counts()


# Hay una alta similitud entre los **'BENEFICIO'** y **'PROCESO'**, veamos que todos los 'Tradicional' se dividen en 'Lavado' y 'Descafeínado', pero no todos los descafeínado son lavado, la variable (columna) que es necesario dejar para permitir esta separación, es entonces la columna de 'BENEFICIO', por lo cual se dropará la de **'PROCESO'**.
df.drop(columns=['PROCESO'], inplace=True)

# Ahora chequeamos los valores únicos de la columna 'LOTE'
print(df['LOTE'].unique())

df['LOTE'].nunique()


# Reemplazamos un valor particular en la columna 'LOTE'
df['LOTE'] = df['LOTE'].replace({'1-190722': '01-190722'})
print(df['LOTE'].unique())
print(df['LOTE'].nunique())

# Guardamos este nuevo DataFrame combinado
df.to_csv('CC_FT_18_cleaned.csv', index=False)
print("DataFrame combinado guardado en ./CC_FT_18_cleaned.csv")