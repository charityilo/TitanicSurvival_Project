from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained pipeline (includes preprocessing + SVM)
model = joblib.load("model/titanic_survival_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""

    if request.method == "POST":
        # Get inputs from form
        pclass = int(request.form["pclass"])
        sex = request.form["sex"]
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        embarked = request.form["embarked"]

        # Create DataFrame matching training columns
        input_df = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "Fare": fare,
            "Embarked": embarked
        }])

        # Predict
        pred = model.predict(input_df)[0]
        prediction_text = "Survived" if pred == 1 else "Did Not Survive"

    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port, debug=True)
