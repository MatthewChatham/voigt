"""
Hidden-div session state callbacks.
"""
import os
import json
from rq import Queue
from sqlalchemy import create_engine
from dash.dependencies import Input, Output, State

from ..worker import conn
from ..server import app

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
    eng = create_engine(DATABASE_URL)
else:
    env = 'Dev'
    BASE_DIR = 'C:\\Users\\Administrator\\Desktop\\voigt'

q = Queue(connection=conn)


@app.callback(
    [Output('areas-state', 'children'), Output('bin-width-state', 'children')],
    [Input('bin-width', 'value')],
    [
        State('bin-width-state', 'children'),
        State('areas-state', 'children'),
        State('type', 'value'),
        State('plot', 'figure')
    ]
)
def areas_state(bin_width, bin_width_state, areas_state, chart_type, figure):
    if bin_width == 10 and chart_type == 'sumcurve' and areas_state is None:
        return '{"areas": {}}', 100

    areas_state = json.loads(areas_state)

    if chart_type == 'area':

        areas = figure['data'][0]['y']
        areas_state['areas'].update({str(bin_width_state): areas})

    return json.dumps(areas_state), bin_width


@app.callback(
    Output('state', 'children'),
    [
        Input('add-split', 'n_clicks_timestamp'),
        Input('remove-split', 'n_clicks_timestamp'),
        Input('split-point', 'value'),
    ],
    [
        State('state', 'children'),
        State('plot', 'figure'),
        State('bin-width', 'value'),
        State('type', 'value'),
    ]
)
def split_state(add, remove, split_point,
                state, figure, bin_width, chart_type):

    # Initial load
    if add == 0 and remove == 0 and chart_type == 'sumcurve':
        return '{"splits":[], "add":0, "remove":0}'

    state = json.loads(state)

    if add > remove and add > state['add']:
        if split_point not in state['splits'] and split_point is not None:
            state['splits'].append(split_point)
    elif remove > add and remove > state['remove']:
        if len(state['splits']) > 0:
            state['splits'].pop()

    state['splits'] = sorted(state['splits'])
    state['add'] = add
    state['remove'] = remove

    # if chart_type == 'area':
    #     areas = figure['data'][0].get('y')
    #     if areas is not None:
    #         state['areas'][bin_width] = areas

    # print(state)

    return json.dumps(state)
