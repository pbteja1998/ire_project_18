from flask import Flask
from flask import request
from flask import render_template
import os
import subprocess
import pickle
from naive_bayes import NaiveBayes

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    # return 'Index Page'

@app.route('/summary', methods=['POST','GET'])
def get_summary():
    if request.method == "POST":
        model = request.form['model']
        file_name = request.form['file_name']
        path_to_app_dir = '/'.join(__file__.split("/")[:-1])
        if path_to_app_dir:
            os.chdir(path_to_app_dir)
        print "Loading " + model + " Naive Bayes Model"
        if model == "G":
            model = "Guassian"
            with open('gnb.pkl', 'rb') as fd:
                nb = pickle.load(fd)
        elif model == "C":
            model = "Complement"
            with open('cnb.pkl', 'rb') as fd:
                nb = pickle.load(fd)
        elif model == "M":
            model = "Multinomail"
            with open('mnb.pkl', 'rb') as fd:
                nb = pickle.load(fd)
        elif model == "B":
            model = "Bernoulli"
            with open('bnb.pkl', 'rb') as fd:
                nb = pickle.load(fd)

        print "Completed Loading " + model + " Naive Bayes Model"
        subpath = '../data/tagged/' + file_name
        return render_template('summary.html', summary=nb.getSummary(subpath), model=model)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)