import sklearn
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# from ..static.csvs ixsmport nba_games_all.csv

model = None

def preprocess(df):
    df = df.dropna()
    # Normalize Data: x - xmin / xmax - xmin
    arr = ['team_id', 'w_pct', 'min', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a',
            'fg3_pct', 'ftm', 'fta', 'ft_pct', 'a_team_id']
    for field in arr:
        scaler = MinMaxScaler() 
        df[field] = scaler.fit_transform(df[[field]])
    df = df.replace('W', 1)
    df = df.replace('L', 0)
    df = df.replace('t', 1)
    df = df.replace('f', 0)
    df['wl'] = df['wl'].astype('int') 
    df['is_home'] = df['is_home'].astype('int') 
    return df

def generate_model():
    global model
    df = pd.read_csv('flaskr/static/csvs/nba_games_all.csv')
    df = preprocess(df)
    X = df[['team_id', 'w_pct', 'a_team_id', 'is_home']]
    y = df['wl']
    model = RandomForestClassifier()
    model.fit(X, y)

