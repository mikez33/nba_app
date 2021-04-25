import os
import sys

from flask import Flask, render_template, redirect, request, flash, url_for

app = Flask(__name__)
app.secret_key=os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    render_template('index.html')