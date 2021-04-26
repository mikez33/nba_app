import sklearn
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

teams = {
    'Miami Heat': 1610612748,
    'Los Angelos Lakers': 1610612747,
    'New York Knicks': 1610612752
}

model = None
copied = None
df_all = None

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
    global df_all
    global copied
    df = pd.read_csv('flaskr/static/csvs/nba_games_all.csv')
    copied = df.copy(deep=True)
    df_all = preprocess(df)
    X = df_all[['team_id', 'w_pct', 'a_team_id', 'is_home']]
    y = df_all['wl']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

def get_indices(teamA, teamB, df):
    index1 = 0
    index2 = 0
    global df_all
    global copied
    for index, row in copied.iterrows():
        if index1 != 0 and index2 != 0:
            return index1, index2
        if row['team_id'] == teamA:
            index1 = df_all.iloc[index]['team_id']
        if row['a_team_id'] == teamB:
            index2 = df_all.iloc[index]['a_team_id']
    return index1, index2


def simulate(teamA, win_pctA, teamB, win_pctB, home):
    generate_model()
    teamA = teams[teamA]
    teamB = teams[teamB]

    global df_all
    teamA, teamB = get_indices(teamA, teamB, df_all)
    new_point = { 'team_id': [teamA], 'w_pct': [win_pctA], 'a_team_id': [teamB], 'is_home': [home] }
    new_df = pd.DataFrame(data=new_point)

    global model
    r = model.predict(new_df)
    return r[0] == 0

if __name__ == '__main__':
    print(simulate('Miami Heat', 0.57, 'Los Angelos Lakers', 0.44, 1))