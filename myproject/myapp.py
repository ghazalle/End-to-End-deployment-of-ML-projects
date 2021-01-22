from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
pickle_in = open('myproject/iris.pkl', 'rb')
model = pickle.load(pickle_in)
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/result', methods=['POST'])
def result():
    sepal_length = request.form['sl']
    sepal_width = request.form['sw']
    petal_length = request.form['pl']
    petal_width = request.form['pw']
    input_array = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    p = model.predict(input_array)
    if p[0, 0] == 1 and p[0, 1] == 0 and p[0, 2] == 0:
        return render_template("result.html", data='0')
    if p[0, 0] == 0 and p[0, 1] == 1 and p[0, 2] == 0:
        return render_template("result.html", data='1')
    if p[0, 0] == 0 and p[0, 1] == 0 and p[0, 2] == 1:
        return render_template("result.html", data='2')


if __name__ == "__main__":
    app.run(debug=True)
