from flask import Flask
from flask import request
import os
import subprocess
import pickle
from naive_bayes import NaiveBayes

app = Flask(__name__)

gnb = None

@app.route('/')
def index():
    from naive_bayes import NaiveBayes
    path_to_app_dir = '/'.join(__file__.split("/")[:-1])
    if path_to_app_dir:
        os.chdir(path_to_app_dir)
    global gnb
    print "here"
    with open('gnb.pkl', 'rb') as fd:
        gnb = pickle.load(fd)
    print "here1"
    return 'Index Page'

@app.route('/summary/<path:subpath>')
def get_summary(subpath):
    from naive_bayes import NaiveBayes
    global gnb
    subpath = '../data/tagged/' + subpath
    return gnb.getSummary(subpath)

if __name__ == '__main__':
    from naive_bayes import NaiveBayes
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True)