from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully")