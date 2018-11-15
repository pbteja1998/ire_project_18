from flask import Flask
from flask import request
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/summary/<path:subpath>')
def get_summary(subpath):    
    # Get summary of given file
    subpath = '../data/tagged/' + subpath
    print subpath
    path_to_app_dir = '/'.join(__file__.split("/")[:-1])
    if path_to_app_dir:
        os.chdir(path_to_app_dir)
    proc = subprocess.Popen(['python', 'summary.py',  subpath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)    
    return proc.communicate()[0]

if __name__ == '__main__':
    print os.getcwd()
    print '/'.join(__file__.split("/")[:-1])
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)