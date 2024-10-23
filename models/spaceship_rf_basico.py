import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Configuración para evitar el FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Cargar los datos
train_df = pd.read_csv('data/train.csv')  # Ruta al archivo de entrenamiento
test_df = pd.read_csv('data/test.csv')    # Ruta al archivo de prueba
submission_df = pd.read_csv('data/sample_submission.csv')  # Ruta al archivo de envío de ejemplo

# Exploración de los datos
print(train_df.head())
print(train_df.info())

# Preprocesamiento de los datos

# Llenar valores nulos y asegurar el tipo de datos correcto
train_df = train_df.ffill().infer_objects(copy=False)
test_df = test_df.ffill().infer_objects(copy=False)

# Dividir la columna 'Cabin' en tres partes para capturar más información
train_df[['Deck', 'Num', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)

# Eliminar columnas innecesarias
train_df.drop(['Name', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test_df.drop(['Name', 'Cabin'], axis=1, inplace=True)

# Seleccionar las columnas categóricas para codificar
categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']

# Aplicar OneHotEncoder para manejar categorías no vistas
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Ajustar el encoder con los datos de entrenamiento y transformar los datos
train_encoded = encoder.fit_transform(train_df[categorical_cols])
test_encoded = encoder.transform(test_df[categorical_cols])

# Convertir los arrays codificados a DataFrames
train_encoded_df = pd.DataFrame(train_encoded, index=train_df.index, columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded, index=test_df.index, columns=encoder.get_feature_names_out(categorical_cols))

# Eliminar las columnas categóricas originales
train_df = train_df.drop(categorical_cols, axis=1)
test_df = test_df.drop(categorical_cols, axis=1)

# Unir las columnas codificadas con el resto del DataFrame
train_df = pd.concat([train_df, train_encoded_df], axis=1)
test_df = pd.concat([test_df, test_encoded_df], axis=1)

# Separar características y objetivo
X = train_df.drop('Transported', axis=1)
y = train_df['Transported']

# Convertir la variable objetivo en binaria (True/False a 1/0)
y = y.astype(int)

# Dividir el conjunto de entrenamiento para validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
rf_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de validación
y_pred = rf_model.predict(X_val)

# Evaluar el modelo
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy en validación: {accuracy:.4f}")

# Hacer predicciones para el conjunto de prueba
test_passenger_ids = test_df['PassengerId']  # Guardar PassengerId para el envío
test_df.drop(['PassengerId'], axis=1, inplace=True)

test_predictions = rf_model.predict(test_df)

# Preparar el archivo para el envío
submission_df['Transported'] = test_predictions
submission_df['Transported'] = submission_df['Transported'].astype(bool)

# Guardar el archivo de envío
submission_df.to_csv('submission.csv', index=False)

print("Archivo de envío creado correctamente.")
