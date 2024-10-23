import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
train_df = train_df.ffill().infer_objects(copy=False)
test_df = test_df.ffill().infer_objects(copy=False)

train_df[['Deck', 'Num', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)

train_df.drop(['Name', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test_df.drop(['Name', 'Cabin'], axis=1, inplace=True)

categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

train_encoded = encoder.fit_transform(train_df[categorical_cols])
test_encoded = encoder.transform(test_df[categorical_cols])

train_encoded_df = pd.DataFrame(train_encoded, index=train_df.index, columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded, index=test_df.index, columns=encoder.get_feature_names_out(categorical_cols))

train_df = train_df.drop(categorical_cols, axis=1)
test_df = test_df.drop(categorical_cols, axis=1)

train_df = pd.concat([train_df, train_encoded_df], axis=1)
test_df = pd.concat([test_df, test_encoded_df], axis=1)

X = train_df.drop('Transported', axis=1)
y = train_df['Transported'].astype(int)

# Dividir el conjunto de entrenamiento para validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar el modelo Random Forest con búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)

# Búsqueda de hiperparámetros usando GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_rf_model = grid_search.best_estimator_
print(f"Mejores hiperparámetros: {grid_search.best_params_}")

# Evaluar el modelo con validación cruzada
cv_scores = cross_val_score(best_rf_model, X, y, cv=5)
print(f"Accuracy promedio con validación cruzada: {np.mean(cv_scores):.4f}")

# Entrenar el modelo con los mejores hiperparámetros
best_rf_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de validación
y_pred = best_rf_model.predict(X_val)

# Evaluar el modelo
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy en validación: {accuracy:.4f}")

# Reporte de clasificación
print(classification_report(y_val, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión')
plt.show()

# Curva ROC
y_probs = best_rf_model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_probs)
roc_auc = roc_auc_score(y_val, y_probs)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Hacer predicciones para el conjunto de prueba
test_passenger_ids = test_df['PassengerId']
test_df.drop(['PassengerId'], axis=1, inplace=True)

test_predictions = best_rf_model.predict(test_df)

# Preparar el archivo para el envío
submission_df['Transported'] = test_predictions
submission_df['Transported'] = submission_df['Transported'].astype(bool)

# Guardar el archivo de envío
submission_df.to_csv('data/submission.csv', index=False)
print("Archivo de envío creado correctamente.")
