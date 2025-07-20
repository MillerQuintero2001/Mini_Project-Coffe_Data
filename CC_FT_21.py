# # Preprocesamiento del Archivo CC FT 21
# Este archivo es del formato de control de despachos

import pandas as pd
import numpy as np
import os
import json


archivos = os.listdir()
# Obtenemos el archivo que comienza con 'CC FT 17'
for archivo in archivos:
    if archivo.startswith('CC FT 21'):
        archivo_xlsx = archivo
        break
print(f"Archivo encontrado: {archivo_xlsx}")

# Abrimos el archivo XLSX
df = pd.read_excel(archivo_xlsx)
# Veamos las hojas del archivo
hojas = pd.ExcelFile(archivo_xlsx).sheet_names
print(f"Hojas disponibles: {hojas}")


# ## TOSTIÓN MEDELLÍN (Hoja 1)
# Comenzamos preprocesando la primera de las hojas
# Seleccionamos la hoja 1, ya sabmos que siempre omitimos las 5 primeras filas
df = pd.read_excel(archivo_xlsx, sheet_name=hojas[0], skiprows=5)


# Dropeamos la primera fila, que es un duplicado de la primera
df = df.drop(index=0)
df.reset_index(drop=True, inplace=True)


# Ahora vamos a limpiar los nombres de las columnas
df.columns = df.columns.str.strip()
# Hacemos lo mismo de borrar espacios vacíos al principio y al final, pero ahora en las columnas de tipo 'object'
# Convertir a string primero y luego aplicar strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()


# Las columnas '# PEDIDO', 'PRESENTACIÓN' Y 'CLIENTE' no son relacionables con los otros dataframes, además de que el objetivo es la predicción del puntaje de taza, para lo cuál la presentación en gramos o kilogramos del café, no es relevante.

# Dropeamos todas las columnas que son solo nulos
df = df.dropna(axis=1, how='all')

df = df.drop(columns=['# PEDIDO', 'PRESENTACIÓN', 'CLIENTE'])

# Veamos la cantidad de valores únicos por columna
valores_unicos = df.nunique()
print("Cantidad de valores únicos por columna:")
print(valores_unicos)


# Veamos los valores únicos de las columnas con 35 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 40:
        print(f"Valores únicos en la columna '{col}':")
        print(df[col].unique())
        print('----'*20)

# Corrección de errores ortográficos en los nombres
df['TIPO DE CAFÉ'] = df['TIPO DE CAFÉ'].replace({
    'Madra Laura': 'Madre Laura',
    'Madre Laura Naturlal': 'Madre Laura Natural',
    'Famila Vergara': 'Familia Vergara',
    'wush Wush Natural': 'Wush Wush Natural',
    'Tabia Natural': 'Tabi Natural'
})

# Veamos los valores únicos de las columnas con 35 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 40:
        print(f"Valores únicos en la columna '{col}':")
        print(df[col].unique())
        print('----'*20)


# Veamos la cantidad de valores únicos
valores_unicos = df.nunique()
print("Cantidad de valores únicos por columna:")
print(valores_unicos)


# ### Guardado del dataframe

# Con la hoja 1 ya procesada, guardamos como .csv
os.makedirs('CC_FT_21_cleaned', exist_ok=True)
df.to_csv('CC_FT_21_cleaned/CC_FT_21_sheet_1.csv', index=False)
print("DataFrame guardado en 'CC_FT_21_cleaned/CC_FT_21_sheet_1.csv'")

# ## TOSTIÓN JERICÓ (Hoja 2)

# Leemos ahora la hoja 2 del archivo
df= pd.read_excel(archivo_xlsx, sheet_name=hojas[1], skiprows=5)
# Mostramos las primeras filas del DataFrame de la hoja 2

# Dropeamos la primera fila, que es un duplicado de la primera
df = df.drop(index=0)
df.reset_index(drop=True, inplace=True)


# Ahora vamos a limpiar los nombres de las columnas
df.columns = df.columns.str.strip()
# Hacemos lo mismo de borrar espacios vacíos al principio y al final, pero ahora en las columnas de tipo 'object'
# Convertir a string primero y luego aplicar strip()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()


# Dropeamos todas las columnas que son solo nulos
df = df.dropna(axis=1, how='all')

df = df.drop(columns=['# PEDIDO', 'PRESENTACIÓN', 'CLIENTE'])
# Mostramos el DataFrame final


# Veamos todos los valores únicos de las columnas
for col in df.columns:
    print(f"Valores únicos en la columna '{col}':")
    print(df[col].unique())
    print('----'*20)



# ### Guardado del dataframe

# Vemos que todo está en orden, así que guardamos el DataFrame de la hoja 2
df.to_csv('CC_FT_21_cleaned/CC_FT_21_sheet_2.csv', index=False)
print("DataFrame de la hoja 2 guardado en 'CC_FT_21_cleaned/CC_FT_21_sheet_2.csv'")

# ## Verificación de Integridad

# Leemos los 2 DataFrames guardados
df_sheet_1 = pd.read_csv('CC_FT_21_cleaned/CC_FT_21_sheet_1.csv')
df_sheet_2 = pd.read_csv('CC_FT_21_cleaned/CC_FT_21_sheet_2.csv')

df_sheet_1.columns == df_sheet_2.columns

print('----'*20)


# ## Unión
# Juntamos el dataframe hoja 2 debajo del de la hoja 1
df = pd.concat([df_sheet_1, df_sheet_2], ignore_index=True)


# Veamos los cantidad de valores únicos por columna
valores_unicos = df.nunique()
print("Cantidad de valores únicos por columna:")
print(valores_unicos)


# Imprimimos los valores únicos de las columnas con 40 o menos valores únicos
for col in df.columns:
    if df[col].nunique() <= 40:
        print(f"Valores únicos en la columna '{col}':")
        print(df[col].unique())
        print('----'*20)

# Renombramos la columna 'TIPO DE CAFÉ' por 'MARCA'
df.rename(columns={'TIPO DE CAFÉ': 'MARCA',
                    'RESPONSABLE DESPACHO': 'TOSTADOR'}, inplace=True)
# Dropeamos la columna 'VERIFICA' ya que no tiene relación con ningún otro de los DataFrames
df.drop(columns=['VERIFICA'], inplace=True)

# Guardamos este nuevo DataFrame combinado
df.to_csv('CC_FT_21_cleaned.csv', index=False)
print("DataFrame combinado guardado en ./CC_FT_21_cleaned.csv")