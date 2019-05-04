
from dash.dependencies import Input, Output, State
from flask import send_file

from os.path import join
import os
import json
import base64
from calendar import timegm
from time import gmtime

from voigt.drawing import countplot, areaplot, curveplot, construct_shapes
from voigt.aggregate import aggregate_all_files

from .server import app
from .extract import read_input

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

ALLOWED_EXTENSIONS = set(['txt'])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def upload(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        for i, c in enumerate(list_of_contents):
            s = c.split(',')[1]
            s = base64.b64decode(s).decode()
            subfolder = 'input' if env == 'Heroku' else 'test_input'
            with open(join(BASE_DIR, subfolder, list_of_names[i]), 'w') as f:
                f.write(s)
        return list_of_names


@app.callback(Output('selection', 'children'), [Input('split-point', 'value')])
def update_selection_prompt(split):
    if split is None:
        return 'Select a split point.'
    else:
        return f'You have selected {split}.'


@app.callback(
    Output('state', 'children'),
    [
        Input('add-split', 'n_clicks_timestamp'),
        Input('remove-split', 'n_clicks_timestamp'),
        Input('split-point', 'value')
    ],
    [State('state', 'children')]
)
def update_state(add, remove, split_point, state):
    if add == 0 and remove == 0:
        return '{"splits":[], "add":0, "remove":0}'

    state = json.loads(state)

    if add > remove and add > state['add']:
        if split_point not in state['splits']:
            state['splits'].append(split_point)
    elif remove > add and remove > state['remove']:
        if len(state['splits']) > 0:
            state['splits'].pop()

    state['splits'] = sorted(state['splits'])
    state['add'] = add
    state['remove'] = remove

    return json.dumps(state)


@app.callback(
    Output('plot', 'figure'),
    [
        Input('bin-width', 'value'),
        Input('scale', 'value'),
        Input('type', 'value'),
        Input('split-point', 'value')
    ],
    [State('state', 'children')]
)
def update_plot(bin_width, scale, chart_type, split_point, state):
    funcs = {'count': countplot, 'area': areaplot, 'curve': curveplot}
    return funcs[chart_type](
        bin_width,
        DATA=read_input(),
        scale=scale,
        shapes=construct_shapes(split_point=split_point)
    )


@app.callback(
    Output('dl_link', 'style'),
    [Input('submit', 'n_clicks')],
    [State('state', 'children')]
)
def submit(n_clicks, state):
    if n_clicks is None:
        return {'display': 'none'}
    if n_clicks is not None:
        splits = json.loads(state)['splits']
        aggregate_all_files(splits, read_input())
        return {}


@app.server.route('/dash/download')
def download_csv():
    return send_file(join(BASE_DIR, 'output', 'output.csv'),
                     mimetype='text/csv',
                     attachment_filename=f'output_{int(timegm(gmtime()))}.csv',
                     as_attachment=True)
