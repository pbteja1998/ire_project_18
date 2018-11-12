from flask import Flask
from flask import request
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/summary/<path:subpath>')
def show_user_profile(subpath):    
    # show the user profile for that user
    proc = subprocess.Popen(['python', 'summary.py',  subpath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)    
    return proc.communicate()[0]