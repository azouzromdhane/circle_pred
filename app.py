from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Charger le modèle sauvegardé
model = tf.keras.models.load_model('C:/Users/hp/PycharmProjects/ia_interface/circle_classifier_model.keras')

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            x = float(request.form["x"])
            y = float(request.form["y"])
            point = np.array([[x, y]])  # Transformer en format compatible avec le modèle

            # Faire la prédiction
            prediction = model.predict(point)

            # La prédiction donne généralement une probabilité, donc on doit vérifier
            if prediction >= 0.5:  # Si la probabilité est supérieure à 0.5, le point est à l'intérieur
                result = "Le point est à l'intérieur du cercle."
            else:
                result = "Le point est à l'extérieur du cercle."
        except ValueError:
            result = "Veuillez entrer des nombres valides."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
