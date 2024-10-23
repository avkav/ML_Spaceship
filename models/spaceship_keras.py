from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load and preprocess data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Function to preprocess data
def preprocess_data(data):
    # Handle missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['RoomService'].fillna(data['RoomService'].median(), inplace=True)
    data['FoodCourt'].fillna(data['FoodCourt'].median(), inplace=True)
    data['ShoppingMall'].fillna(data['ShoppingMall'].median(), inplace=True)
    data['Spa'].fillna(data['Spa'].median(), inplace=True)
    data['VRDeck'].fillna(data['VRDeck'].median(), inplace=True)
    
    # Convert categorical variables to numeric
    data['HomePlanet'] = pd.Categorical(data['HomePlanet']).codes
    data['CryoSleep'] = pd.Categorical(data['CryoSleep']).codes
    data['Destination'] = pd.Categorical(data['Destination']).codes
    data['VIP'] = pd.Categorical(data['VIP']).codes
    
    return data

# Preprocess train and test data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Prepare features and target
X = train_data.drop(['PassengerId', 'Name', 'Cabin', 'Transported'], axis=1)
y = train_data['Transported'].astype(int)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_val_scaled, y_val), verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_val_scaled, y_val)
print(f"Validation Accuracy: {accuracy}")

# Make predictions on test data
X_test = test_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
predictions_binary = (predictions > 0.5).astype(int)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions_binary.flatten().astype(bool)
})
submission.to_csv('data/submission_keras.csv', index=False)

# Plot confusion matrix
y_pred = model.predict(X_val_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_val, y_pred_binary)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Transported', 'Transported'])
plt.yticks(tick_marks, ['Not Transported', 'Transported'])
plt.ylabel('True label')
plt.xlabel('Predicted label')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig('data/confusion_matrix_keras.png')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred_binary))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('data/training_history_keras.png')