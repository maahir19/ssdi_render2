import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# --------------------
# Flask App
# --------------------
app = Flask(__name__)

# --------------------
# Train Model
# --------------------
iris = load_iris()

X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

model = GaussianNB()
model.fit(X_train, Y_train)

# --------------------
# Home Page (HTML)
# --------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------
# Prediction Route
# --------------------
@app.route("/predict", methods=["POST"])
def predict():

    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    features = np.array([[sepal_length,
                          sepal_width,
                          petal_length,
                          petal_width]])

    prediction = model.predict(features)[0]
    result = iris.target_names[prediction]

    return render_template("index.html", prediction_text=f"Prediction: {result}")


# --------------------
# Local Run Only
# --------------------
if __name__ == "__main__":
    app.run()