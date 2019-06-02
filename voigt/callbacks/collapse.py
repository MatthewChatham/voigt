"""
Callbacks that control opening/closing of collapsable list items.
"""
import os
from rq import Queue
from sqlalchemy import create_engine
from dash.dependencies import Input, Output, State

from ..worker import conn
from ..server import app

if os.environ.get('STACK'):
    print('RUNNING ON HEROKU')
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
    eng = create_engine(DATABASE_URL)
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

q = Queue(connection=conn)


@app.callback(
    Output("collapse1", "is_open"),
    [Input("collapse-button1", "n_clicks")],
    [State("collapse1", "is_open")],
)
def toggle_collapse1(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse2", "is_open"),
    [Input("collapse-button2", "n_clicks")],
    [State("collapse2", "is_open")],
)
def toggle_collapse2(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse3", "is_open"),
    [Input("collapse-button3", "n_clicks")],
    [State("collapse3", "is_open")],
)
def toggle_collapse3(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse4", "is_open"),
    [Input("collapse-button4", "n_clicks")],
    [State("collapse4", "is_open")],
)
def toggle_collapse4(n, is_open):
    if n:
        return not is_open
    return is_open
