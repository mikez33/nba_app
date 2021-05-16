import sklearn
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

teams = {
    'Miami Heat': 1610612748,
    'Los Angelos Lakers': 1610612747,
    'New York Knicks': 1610612752,
    'Atlanta Hawks': 1610612737,
    'Boston Celtics': 1610612738,
    'Brooklyn Nets': 1610612751,
    'Charlotte Hornets': 1610612766,
    'Chicago Bulls': 1610612741,
    'Cleveland Cavaliers': 1610612739,
    'Dallas Maverick': 1610612742,
    'Denver Nuggets': 1610612743,
    'Detroit Pistons': 1610612765,
    'Golden State Warriors': 1610612744,
    'Houston Rockets': 1610612745,
    'Indiana Pacers': 1610612754,
    'Los Angeles Clippers': 1610612746,
    'Memphis Grizzlies': 1610612763,
    'Milwaukee Bucks': 1610612749,
    'Minnesota Timberwolves': 1610612750,
    'New Orleans Pelicans': 1610612740,
    'Oklahoma City Thunder': 1610612760,
    'Orlando Magic': 1610612753,
    'Philadelphia 76ers': 1610612755,
    'Phoenix Suns': 1610612756,
    'Portland Trail Blazers': 1610612757,
    'Sacramento Kings': 1610612758,
    'San Antonio Spurs': 1610612759,
    'Toronto Raptors': 1610612761,
    'Utah Jazz': 1610612762,
    'Washington Wizard': 1610612764
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
    # model = RandomForestClassifier()
    model = make_pipeline(StandardScaler(), 
                        SVC(gamma='auto', 
                            tol=0.03, 
                            kernel='linear',
                            class_weight='balanced'))
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