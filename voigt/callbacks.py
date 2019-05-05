
from dash.dependencies import Input, Output, State
from flask import send_file

from os.path import join
import os
import json
import base64
from calendar import timegm
from time import gmtime

from voigt.drawing import countplot, areaplot, curveplot, construct_shapes
from voigt.aggregate import generate_output_file

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

        # Delete previous files
        for existing_file in os.listdir(join(BASE_DIR, 'input')):
            if existing_file != '.hold':
                os.remove(join(BASE_DIR, 'input', existing_file))

        for i, c in enumerate(list_of_contents):
            s = c.split(',')[1]
            s = base64.b64decode(s).decode()
            with open(join(BASE_DIR, 'input', list_of_names[i]), 'w') as f:
                f.write(s)

        return str(list_of_names)


@app.callback(
    Output('selection', 'children'),
    [Input('state', 'children')],
)
def update_selection_prompt(state):
    if state is None:
        return 'Select a split point.'
    else:
        state = json.loads(state)
        return f'Current splits: {str(state["splits"])}.'


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
def update_areas_state(bin_width, bin_width_state, areas_state, chart_type, figure):
    if bin_width == 100 and chart_type == 'count' and areas_state is None:
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
def update_state(add, remove, split_point, state, figure, bin_width, chart_type):

    # Initial load
    if add == 0 and remove == 0 and chart_type == 'count':
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

    # if chart_type == 'area':
    #     areas = figure['data'][0].get('y')
    #     if areas is not None:
    #         state['areas'][bin_width] = areas

    # print(state)

    return json.dumps(state)


@app.callback(
    Output('plot', 'figure'),
    [
        Input('bin-width', 'value'),
        Input('scale', 'value'),
        Input('type', 'value'),
        # Input('split-point', 'value'),
        Input('parse-data-and-refresh-chart', 'n_clicks')
    ],
    [
        State('state', 'children'),
        State('areas-state', 'children')
    ]
)
def update_plot(bin_width, scale, chart_type, refresh_clicks, state, areas_state):
    funcs = {'count': countplot, 'area': areaplot, 'curve': curveplot}

    kwargs = dict(
        bin_width=bin_width,
        DATA=read_input(),
        scale=scale,
        shapes=[],  # construct_shapes(split_point=split_point),
    )

    if areas_state is not None and chart_type == 'area':

        areas_state = json.loads(areas_state)

        cached = areas_state['areas'].get(str(bin_width)) is not None
        if cached:
            print('FOUND CACHED AREA')
            areas = areas_state['areas'][str(bin_width)]
            kwargs['areas'] = areas

    return funcs[chart_type](**kwargs)


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
        generate_output_file(splits, read_input())
        return {}


@app.server.route('/dash/download')
def download_csv():
    return send_file(join(BASE_DIR, 'output', 'output.csv'),
                     mimetype='text/csv',
                     attachment_filename=f'output_{int(timegm(gmtime()))}.csv',
                     as_attachment=True)
