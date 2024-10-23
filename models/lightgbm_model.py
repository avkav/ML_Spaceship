import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
submission_df = pd.read_csv('data/sample_submission.csv')

# Preprocesamiento
train_df = train_df.ffill()
test_df = test_df.ffill()

train_df[['Deck', 'Num', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)

train_df.drop(['Name', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test_df.drop(['Name', 'Cabin'], axis=1, inplace=True)

# Convertir 'Num' a tipo numérico
train_df['Num'] = pd.to_numeric(train_df['Num'], errors='coerce')
test_df['Num'] = pd.to_numeric(test_df['Num'], errors='coerce')

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

# Entrenar el modelo LightGBM
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = lgb_model.predict(X_val)

# Evaluar el modelo
print("LightGBM:")
print(classification_report(y_val, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión')
plt.show()

# Curva ROC
y_probs = lgb_model.predict_proba(X_val)[:, 1]
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

test_predictions = lgb_model.predict(test_df)

# Preparar el archivo para el envío
submission_df['Transported'] = test_predictions
submission_df['Transported'] = submission_df['Transported'].astype(bool)
submission_df.to_csv('submission_lightgbm.csv', index=False)
print("Archivo de envío creado correctamente para LightGBM.")
