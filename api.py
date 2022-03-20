from flask import Flask, jsonify, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=["POST", "GET"])
def question():
    if request.method == "POST":
        age = request.form['age'] if request.form['age'] else 90
        gender = 1 if request.form['gender'] == "M" else 2
        smoke = 2 if request.form['smoke'] == "Y" else 1
        yf = 2 if request.form['yf'] == "Y" else 1
        anxiety = 2 if request.form['anxiety'] == "Y" else 1
        disease = 2 if request.form['disease'] == "Y" else 1
        tired = 2 if request.form['tired'] == "Y" else 1
        allergy = 2 if request.form['allergy'] == "Y" else 1
        wheeze = 2 if request.form['wheeze'] == "Y" else 1
        drink = 2 if request.form['drink'] == "Y" else 1
        cough = 2 if request.form['cough'] == "Y" else 1
        breath = 2 if request.form['breath'] == "Y" else 1
        swallow = 2 if request.form['swallow'] == "Y" else 1
        chest = 2 if request.form['chest'] == "Y" else 1
        values = int(age), int(smoke), int(yf), int(anxiety), int(disease), int(tired), int(allergy), int(wheeze), int(drink), int(cough), int(breath), int(swallow), int(chest), int(gender)
        return redirect(url_for("predict", val = values))
    else:
        return render_template("test.html")


@app.route("/<val>", methods=["GET"])
def predict(val):
    with open('model', 'rb') as file:
        model = pickle.load(file)

    val = val[1:len(val) - 1]
    val = val.split(",")
    val = list(map(int, val))
    val = np.asarray(val)
    val = val.reshape(1, -1)

    pred = model.predict(val)

    if (pred[0] == 'NO'):
        res = "LOW"
    else:
        res = "HIGH"
    result = {"chances": res}
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(debug=True)
