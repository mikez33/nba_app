import os
import sys

from flask import Flask, render_template, redirect, request, flash, url_for
from .functions import model, simulate

app = Flask(__name__)
app.secret_key=os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['submit_button'] == 'simulate':
            return redirect('/teams/')
    return render_template('index.html')


@app.route('/teams/', methods=['GET', 'POST'])
def teams():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Submit':
            teamA = str(request.form['TeamA'])
            win_pctA = float(request.form['win_pctA'])
            teamB = str(request.form['TeamB'])
            win_pctB = float(request.form['win_pctB'])
            home = str(request.form['Home'])
            simulations = int(request.form['simulations'])
            if home == 'TeamA':
                home = 1
            else:
                home = 0
            return redirect(url_for('results', teamA=teamA, win_pctA=win_pctA, 
                teamB=teamB, win_pctB=win_pctB, home=home, simulations=simulations))
    return render_template('teams.html')


@app.route('/results/', methods=['GET', 'POST'])
def results():
    teamA = request.args.get('teamA')
    win_pctA = request.args.get('win_pctA')
    teamB = request.args.get('teamB')
    win_pctB = request.args.get('win_pctB')
    home = request.args.get('home')
    simulations = request.args.get('simulations')

    results = []
    for i in range(int(simulations)):
        data = {}
        result = simulate.simulate(teamA, float(win_pctA), teamB, float(win_pctB), home)
        if result:
            result =  teamA + ' Wins!'
        else:
            result = teamB + ' Wins!'
        data['index'] = i + 1
        data['winner'] = result
        results.append(data)
    return render_template('results.html', results=results)