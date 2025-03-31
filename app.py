from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Charger le modèle
model = joblib.load("model_heart_disease.pkl")

# Dictionnaire pour interpréter le résultat
dicotarget = {
    1: "Le patient n'est pas malade",
    2: "Le patient est malade"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Récupérer les données du formulaire
            mv = int(request.form["mv"])
            eia = int(request.form["eia"])
            thal = int(request.form["thal"])
            oldpeak = float(request.form["oldpeak"])
            sc = int(request.form["sc"])
            cpt = int(request.form["cpt"])

            # Créer un tableau NumPy
            Donnees = np.array([[mv, eia, thal, oldpeak, sc, cpt]])

            # Faire une prédiction
            prediction = dicotarget[model.predict(Donnees)[0]]

        except Exception as e:
            prediction = "Erreur dans les données fournies."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
myapp = app 