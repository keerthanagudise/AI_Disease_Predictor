import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Features
X = data.drop("disease", axis=1)

# Target
y = data["disease"]

# Train Model
model = DecisionTreeClassifier()

model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")