# Anteriormente se pre-procesaron y limpiaron las diversas hojas de los 3 datasets en formato XLSX, relacionados con los formatos 17,18 y 21 que tratan sobre el control de calidad del café trillado, tostión y control de despachos. Ahora con eso limpio y unificado por hojas vamos a proceder a
# - Realizar EDA(Exploratory Data Analysis)
# - Procesar y entrenar modelos de regresión lineal



import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error ,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
import optuna



# Abrimos los 3 dataframes para tenerlos disponibles
df = pd.read_csv('CC_FT_17_cleaned.csv')
df_2 = pd.read_csv('CC_FT_18_cleaned.csv')
df_3 = pd.read_csv('CC_FT_21_cleaned.csv')


# ## EDA

# ### Formato 17: Control de Calidad del Café Trillado

# Veamos la matriz de correlación de las variables numéricas
df_numeric = df.select_dtypes(include=[np.number])
correlation_matrix = df_numeric.corr()
# Gráfica de la matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Matriz de Correlación")
plt.savefig("Matriz_Correlacion.png", dpi = 600)

# Desde un punto de vista basado en pura intuición, la **MARCA** de un café no es más que un nombre/logo, la calidad del café está asociada es a su lugar de cultivo, variedad y variables relacionadadas con el proceso desde la cosecha hasta el empaque del producto. Considerando esto y que dicha columna esta cargada de muchos valores únicos que aumentan tremendamente la dimensionalidad del problema para un modelo de regresión, se opto por descartarla en los dataframes donde está presente.
df.drop(columns=['MARCA'], inplace=True)



# Aqui nuevamente recurriendo a una idea basada en intuición, tener una variedad o planta específica de café cultivada, depende de condiciones del ambiente, por lo cual la variedad en si misma exige condiciones locales, por lo cual del ugar de origen del café no tiene mucha importancia, sino la variedad, ya que define junto con el proceso, el sabor final del café. Por ello, vamos a descartar la columna de **'ORIGEN'**.

df_2.drop(columns=['ORIGEN'], inplace=True)


# ## Merge de los dataframes de los diversos archivos
# 
# Por no notar algo útil o importante en el dataframe del formato 21 (control de despachos), no se utilizará en el procesamiento para obtener el dataset final para el modelo de ML.



unicos_df = set(df['LOTE'].unique())
unicos_df_2 = set(df_2['LOTE'].unique())

print("Valores únicos en df:")
print(f"Cantidad: {len(unicos_df)}")
print(f"Valores: {sorted(unicos_df)}")
print()

print("Valores únicos en df_2:")
print(f"Cantidad: {len(unicos_df_2)}")
print(f"Valores: {sorted(unicos_df_2)}")
print()

# 2. Valores que están en df pero NO en df_2
solo_en_df = unicos_df - unicos_df_2
print("Valores que están SOLO en df (no en df_2):")
print(f"Cantidad: {len(solo_en_df)}")
print(f"Valores: {sorted(solo_en_df)}")
print()

# 3. Valores que están en df_2 pero NO en df
solo_en_df_2 = unicos_df_2 - unicos_df
print("Valores que están SOLO en df_2 (no en df):")
print(f"Cantidad: {len(solo_en_df_2)}")
print(f"Valores: {sorted(solo_en_df_2)}")
print()

# 4. Valores que están en ambos dataframes (intersección)
en_ambos = unicos_df & unicos_df_2
print("Valores que están en AMBOS dataframes:")
print(f"Cantidad: {len(en_ambos)}")
print(f"Valores: {sorted(en_ambos)}")
print()

# 5. Resumen completo
print("=== RESUMEN COMPLETO ===")
print(f"Total únicos en df: {len(unicos_df)}")
print(f"Total únicos en df_2: {len(unicos_df_2)}")
print(f"Solo en df: {len(solo_en_df)}")
print(f"Solo en df_2: {len(solo_en_df_2)}")
print(f"En ambos: {len(en_ambos)}")
print(f"Total únicos combinados: {len(unicos_df | unicos_df_2)}")



"""De aquí vamos a dropear las siguientes columnas:
- 'FECHA': Pues está presente en ambos dataframes y puede ser conflictiva.
- 'NOTAS': son puras descripciones subjetivas del sabor del café, y además cargan 
alta complejidad categórica y dimensional.
- 'CANTIDAD': No hay información ni es intuible sobre lo que significa exactamente."""

df.drop(columns=['NOTAS', 'CANTIDAD'], inplace=True)


# Como en el anterior dataframe ya hay columna de 'FECHA', se dropea en este cuyas dimenesiones no coinciden
# para que no haya conflictos en el merge
df_2.drop(columns=['FECHA'], inplace=True)


# El dataframe base de este proceso de merge, es el `df` asociado al formato 17 del control de calidad de café trillado, pues es el que contiene la variable objetivo, como las dimensiones en filas de estos dataframes no coinciden, no se pueden combinar fácilmente, por lo cual la columna clave para lograrlo es **'LOTE'**.

# Para lograr esto, vamos a usar una estructura de datos de diccionario, donde:
# 
# - Clave: Valor del lote.
# - Valor: Otro diccionario con el valor para cada cada una de las columnas, agrupadas para ese lote.


# Agrupamos df_2 por 'LOTE' y tomamos la media de las columnas numéricas y la moda de las categóricas
df_2_grouped = df_2.groupby('LOTE')
columns = [col for col in df_2.columns if col != 'LOTE']

print(columns)



# Creamos un diccionario donde las claves son los valores únicos de 'LOTE'
dict_lote = {}
list_lotes = df_2['LOTE'].unique().tolist()
for lote in list_lotes:
    temp_dict = {}
    for col in columns:
        if df_2[col].dtype == 'object':
            # Para columnas categóricas, usamos la moda
            moda = df_2[df_2['LOTE'] == lote][col].mode()[0]
            if pd.isna(moda):
                moda = 'Desconocido'
            temp_dict[col] = moda
        else:
            # Para columnas numéricas, usamos la media
            media = df_2[df_2['LOTE'] == lote][col].mean()
            temp_dict[col] = media
    dict_lote[lote] = temp_dict





# Usamos el diccionario para crear los nuevos valores en el datafram principal (df)
for lote, values in dict_lote.items():
    for col, value in values.items():
        df.loc[df['LOTE'] == lote, col] = value


# Ya se pueden dropear las columnas de 'LOTE' y 'FECHA'
df_final = df.copy()
df_final.drop(columns=['LOTE', 'FECHA'], inplace=True)






# ## Modelos
# 
# Se probarán 2 enfoques para tratar los datos
# 1. Dropear las filas con valores nulos y entrenar modelos para ver que es lo mejor que se obtiene.
# 2. Imputar la filas con valores nulas usando KNN-Imputer.
# 
# En cualquiera de los 2 enfoques:
# * Se procesarán las variables categóricas con One-Hot encoding, y se normalizaran las numéricas.
# * Se hara búsqueda de hiperparámetros y validación cruzada.
# * Se entrenaran Linear Regression, SGD Regression y Random Forest Regressor.




# Nuevo dataframe con filas nulas dropeadas
df_model_1 = df_final.dropna()

len(df_model_1.select_dtypes(include=[np.number]).columns)


# Hagamos box plots para las columnas numéricas
numeric_columns = df_model_1.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(8, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df_model_1[col])
    plt.title(f'Boxplot de {col}')
# Aumentamos el espaciado entre los subplots con tight_layout
plt.tight_layout(pad=1.0, h_pad=3.0, w_pad=3.0)
plt.savefig("Box Plots Numeric Cols.png", dpi = 600)

# Como se pueden apreciar en los gráficos algunos outliers, esto nos permite dar cuenta que para las columnas numéricas puede ser conveniente utilizar un `Robust Scaler`.




# Dividimos los datos en variables independientes (X) y dependientes (y)
X = df_model_1.drop(columns=['PUNTAJE'])
y = df_model_1['PUNTAJE']

# Separamos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)


numeric_columns = X_train.select_dtypes(include=[np.number]).columns
categorical_columns = X_train.select_dtypes(include=['object']).columns
print("Columnas numéricas:", numeric_columns)
print("Columnas categóricas:", categorical_columns)

transformer_data = ColumnTransformer(
    transformers=[
        # Escalador robusto para manejar outliers
        ('num', RobustScaler(), numeric_columns),  
        # Codificador para variables categóricas
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output = False), categorical_columns)
    ]
)


def dataframe_evaluation(X_train, y_train, X_test, y_test, model):
    """
    Evaluates the model on both training and test datasets.

    Parameters:
    - X_train: Training features
    - y_train: Training target
    - X_test: Test features
    - y_test: Test target
    - model: Trained model

    Returns:
    - DataFrame with evaluation metrics for both train and test sets.
    """

    # Evaluamos tanto en train como test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluación del modelo en el conjunto de entrenamiento
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Evaluación del modelo en el conjunto de prueba
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # Resultados de la evaluación
    results = {
        'Train': {
            'MAE': mae_train,
            'MSE': mse_train,
            'RMSE': rmse_train,
            'R2': r2_train
        },
        'Test': {
            'MAE': mae_test,
            'MSE': mse_test,
            'RMSE': rmse_test,
            'R2': r2_test
        }
    }
    results_df = pd.DataFrame(results).T
    return results_df


# ### 1.1 Regresión Lineal


# Hacemos el modelo con Pipeline
linear_reg = Pipeline(steps=[
    # Transformación de datos
    ('transformer', transformer_data),
    # Regresor lineal
    ('regressor', LinearRegression())
])

# Entrenamos el modelo
linear_reg.fit(X_train, y_train)

report = dataframe_evaluation(X_train, y_train, X_test, y_test, linear_reg)


# ### 1.2 Regresión Lineal con Regularización Ridge


# Definimos la función objetivo para Optuna para una Ridge Regression
def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)
    solver = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'])
    ridge_model = Pipeline(steps=[
        ('transformer', transformer_data),
        ('regressor', Ridge(alpha=alpha, solver=solver, random_state=42))
    ])
    score = cross_val_score(ridge_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -score.mean()


# Instanciamos el sampler de optuna
sampler = optuna.samplers.TPESampler(seed=42)
# Creamos el estudio de Optuna
study = optuna.create_study(direction='minimize', sampler=sampler)
# Ejecutamos la optimización
study.optimize(objective, n_trials=50)
# Guardamos los mejores hiperparámetros
best_params = study.best_params
print("Mejores hiperparámetros:", best_params)



# Creamos el modelo con los mejores hiperparámetros
ridge_model = Pipeline(steps=[
    ('transformer', transformer_data),
    ('regressor', Ridge(alpha=best_params['alpha']))
])

# Entrenamos el modelo
ridge_model.fit(X_train, y_train)

# Evaluamos el modelo
report_ridge = dataframe_evaluation(X_train, y_train, X_test, y_test, ridge_model)



# ### 1.3 Random Forest

# Hacemos búsqueda de hiperparámetros con Optuna
def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    - trial: Optuna trial object

    Returns:
    - Mean Squared Error (MSE) of the model on the validation set.
    """

    # Definimos los hiperparámetros a optimizar
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=10)
    max_depth = trial.suggest_int('max_depth', 5, 50, step=3)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Creamos el modelo con los hiperparámetros sugeridos
    rf_model = Pipeline(steps=[
        ('transformer', transformer_data),
        ('regressor', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                            min_samples_split=min_samples_split, 
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=42))
    ])

    # Hacemos cross-validation para evaluar el modelo
    scores = cross_val_score(rf_model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    mse = -scores.mean()  # Convertimos a positivo ya que cross_val_score devuelve el negativo del MSE

    return mse

# Instanciamos el sampler de optuna
sampler = optuna.samplers.TPESampler(seed=42)
# Creamos el estudio de Optuna minimizando el MSE
study = optuna.create_study(direction='minimize', sampler=sampler)
# Ejecutamos la optimización
study.optimize(objective, n_trials=30)

# Guardamos los mejores hiperparámetros
best_params = study.best_params
print("Mejores hiperparámetros:", best_params)


# Creamos un nuevo modelo de Random Forest con los mejores hiperparámetros
rf_model = Pipeline(steps=[
    ('transformer', transformer_data),
    ('regressor', RandomForestRegressor(n_estimators=best_params['n_estimators'], 
                                        max_depth=best_params['max_depth'],
                                        min_samples_split=best_params['min_samples_split'],
                                        min_samples_leaf=best_params['min_samples_leaf'],
                                        random_state=42))
])
# Entrenamos el modelo
rf_model.fit(X_train, y_train)


# Veamos los resultados de la evaluación
rf_report = dataframe_evaluation(X_train, y_train, X_test, y_test, rf_model)


# #### Comparación entre modelos

# Reunimos las métricas de test para cada modelo
results = {
    'Linear Regression': report.loc['Test'],
    'Ridge Regression': report_ridge.loc['Test'],
    'Random Forest': rf_report.loc['Test']
}
# Convertimos el diccionario a DataFrame
results_df = pd.DataFrame(results).T
results_df


# Gŕaficamos en subplots cada métrica, y en cada subplot va la métrica particular para cada modelo
metrics = ['MAE', 'MSE', 'RMSE', 'R2']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
axes = axes.flatten()
for i, metric in enumerate(metrics):
    sns.barplot(x=results_df.index, hue=results_df.index, y=results_df[metric], ax=axes[i], palette='Pastel1', legend=False)
    axes[i].set_title(f'{metric} por Modelo')
    axes[i].set_ylabel(metric)
    # Desactivamos label X
    axes[i].set_xlabel('')
    # Ponemos los valores encima de las barras
    for p in axes[i].patches:
        axes[i].annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=14, color='black', rotation=0)
plt.tight_layout(pad=1.0, h_pad=3.0, w_pad=3.0)
# Guardamos la figura
plt.savefig("Metrics Regression.png", dpi = 600)


# ### Feature Importance
# Vamos a obtener las características más importantes según cada modelo

# Para la regresión lineal
linear_reg_importances = pd.DataFrame({
    'Feature': linear_reg.named_steps['transformer'].get_feature_names_out(),
    'Importance': np.abs(linear_reg.named_steps['regressor'].coef_)
}).sort_values(by='Importance', ascending=False)

# Para la Ridge Regression
ridge_importances = pd.DataFrame({
    'Feature': ridge_model.named_steps['transformer'].get_feature_names_out(),
    'Importance': np.abs(ridge_model.named_steps['regressor'].coef_)
}).sort_values(by='Importance', ascending=False)

# Para el Random Forest
rf_importances = pd.DataFrame({
    'Feature': rf_model.named_steps['transformer'].get_feature_names_out(),
    'Importance': rf_model.named_steps['regressor'].feature_importances_
}).sort_values(by='Importance', ascending=False)



# Imprimimos top 5 features según cada modelo, sin índices
print(f"5 Características más Imporantes en Regresión Lineal:\n{linear_reg_importances.head().reset_index(drop=True)}\n")
print(f"5 Características más Imporantes en Ridge Regression:\n{ridge_importances.head().reset_index(drop=True)}\n")
print(f"5 Características más Imporantes en Random Forest:\n{rf_importances.head().reset_index(drop=True)}\n")