import dash
from sqlalchemy import create_engine
import sqlite3
import os

if os.environ.get('STACK'):
    print('RUNNING ON HEROKU')
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://gist.githubusercontent.com/mchatham-fk/852b69b35c20add71cb438a8d98b37cd/raw/3f1f3030762f0a29904538255ef5b49bd3d562f4/stylesheet.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions'] = True

dbconn = create_engine(DATABASE_URL) if os.environ.get(
    'STACK') else sqlite3.connect('output.db')
