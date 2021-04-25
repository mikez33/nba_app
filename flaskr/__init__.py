import os
import sys

from flask import Flask, render_template, redirect, request, flash, url_for
from .functions import model, simulate

app = Flask(__name__)
app.secret_key=os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    model.generate_model()
    return render_template('index.html')

@app.route('/teams', methods=['GET', 'POST'])
def teams():
    return render_template('teams.html')