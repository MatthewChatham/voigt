"""
Creating the app in this file (instead of app.py)
allows us to separate callbacks from layout.
"""
import dash
import dash_bootstrap_components as dbc
import os

if os.environ.get('STACK'):
    print('RUNNING ON HEROKU')
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions'] = True
app.scripts.config.serve_locally = True
