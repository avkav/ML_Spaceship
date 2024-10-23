import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración para evitar el FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Cargar los datos
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
submission_df = pd.read_csv('data/sample_submission.csv')

# Función para preprocesar los datos
def preprocess_data(df):
    # Separar Cabin en Deck, Num y Side
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    
    # Convertir VIP a numérico
    df['VIP'] = df['VIP'].map({'False': 0, 'True': 1})
    
    # Imputación de valores faltantes usando KNN para variables numéricas
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    knn_imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
    
    # Manejo de valores atípicos (ejemplo para Age)
    df['Age'] = np.where(df['Age'] > df['Age'].quantile(0.99), df['Age'].quantile(0.99), df['Age'])
    
    # Ingeniería de características
    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['HasCryoSleep'] = df['CryoSleep'].map({'False': 0, 'True': 1})
    
    return df

# Preprocesar los datos
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Imputar valores faltantes en variables categóricas con SimpleImputer
categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
categorical_imputer = SimpleImputer(strategy='most_frequent')
train_df[categorical_cols] = categorical_imputer.fit_transform(train_df[categorical_cols])
test_df[categorical_cols] = categorical_imputer.transform(test_df[categorical_cols])

# Verificar si todavía hay valores NaN en el conjunto de datos
print("Valores faltantes en el conjunto de entrenamiento:")
print(train_df.isnull().sum())

# Verificar si las columnas categóricas están vacías o contienen datos nulos antes del OneHotEncoder
print("\nVerificando si hay valores nulos en las columnas categóricas:")
print(train_df[categorical_cols].isnull().sum())

# Verificar si hay filas vacías en las columnas categóricas
if train_df[categorical_cols].shape[0] == 0:
    raise ValueError("Las columnas categóricas están vacías, verifica la imputación de datos o el filtrado previo.")

# One-hot encoding para variables categóricas
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Codificación con OneHotEncoder
train_encoded = encoder.fit_transform(train_df[categorical_cols])
test_encoded = encoder.transform(test_df[categorical_cols])

# Convertir a DataFrame las variables codificadas
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Combinar datos codificados con los originales, eliminando columnas innecesarias
train_df = pd.concat([train_df.drop(categorical_cols + ['Name', 'Cabin', 'PassengerId'], axis=1), train_encoded_df], axis=1)
test_df = pd.concat([test_df.drop(categorical_cols + ['Name', 'Cabin'], axis=1), test_encoded_df], axis=1)

# Separar características y objetivo
X = train_df.drop('Transported', axis=1)
y = train_df['Transported'].astype(int)

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Selección de características
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=20)
X_selected = selector.fit_transform(X_scaled, y)

# Verificar si hay valores NaN en X_selected antes de aplicar SMOTE
print("Valores faltantes en X_selected:")
print(pd.DataFrame(X_selected).isnull().sum())

# Balanceo de clases usando SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# Dividir el conjunto de entrenamiento para validación
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Configurar el modelo Random Forest con búsqueda de hiperparámetros
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomForestClassifier(random_state=42)

# Búsqueda de hiperparámetros usando RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_rf_model = random_search.best_estimator_
print(f"Mejores hiperparámetros: {random_search.best_params_}")

# Evaluar el modelo con validación cruzada
cv_scores = cross_val_score(best_rf_model, X_resampled, y_resampled, cv=5)
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

# Preparar y hacer predicciones para el conjunto de prueba
test_passenger_ids = test_df['PassengerId']
test_df = test_df.drop(['PassengerId'], axis=1)
test_scaled = scaler.transform(test_df)
test_selected = selector.transform(test_scaled)
test_predictions = best_rf_model.predict(test_selected)

# Preparar el archivo para el envío
submission_df['Transported'] = test_predictions
submission_df['Transported'] = submission_df['Transported'].astype(bool)

# Guardar el archivo de envío
submission_df.to_csv('data/submission.csv', index=False)
print("Archivo de envío creado correctamente.")
