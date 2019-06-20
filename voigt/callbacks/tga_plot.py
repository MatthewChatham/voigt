"""
Callbacks for TGA chart and dropdown.
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
    Output('tga-plot-dropdown', 'options'),
    [Input('tga-parse-data-and-refresh-chart', 'n_clicks')
     ], [State('session-id', 'children')]
)
def set_tga_file_options(n_clicks, session_id):
    if n_clicks is None:
        return []
    print('clicked button')
    input_dir = os.path.join(BASE_DIR, 'input', f'input_{session_id}', 'fitting')
    if len(os.listdir(input_dir)) == 0:
        print('input dir is empty')
        return []

    return [{'label': f, 'value': f} for f in os.listdir(input_dir)]


@app.callback(
    Output('tga-plot', 'figure'),
    [Input('tga-plot-dropdown', 'value')]
)
def update_plot(file):
    pass
