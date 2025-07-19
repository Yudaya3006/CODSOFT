# iris_classifier.py

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ✅ Step 2: Load Your Dataset (Step 3 you're missing)
data = pd.read_csv("IRIS.csv")  # Make sure this file is in the same folder

# Step 3: Preview the data (Optional)
print("First 5 rows:\n", data.head())
print("\nColumn Names:", data.columns)

# Step 4: Visualize the Data (Optional)
sns.pairplot(data, hue='species')
plt.show()

# Step 5: Prepare Features and Target
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Step 6: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Train the Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the Model
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
