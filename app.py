from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle entraîné
model = joblib.load('model_heart_disease.pkl')

# Dictionnaire pour les résultats
dicotarget = {
    1: "Le patient n'est pas malade",
    2: "Le patient est malade"
}

# Route pour afficher le formulaire
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

            # Convertir en tableau numpy
            Donnees = np.array([mv, eia, thal, oldpeak, sc, cpt]).reshape(1, -1)

            # Prédire avec le modèle
            resultat = model.predict(Donnees)[0]

            # Trouver le message correspondant
            prediction = dicotarget[resultat]

        except Exception as e:
            prediction = "Erreur dans la saisie des données !"

    return render_template("index.html", prediction=prediction)

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True)

myapp = app 