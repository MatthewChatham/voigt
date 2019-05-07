

from dash.dependencies import Input, Output, State
import dash_html_components as html
from flask import send_file

from os.path import join, isfile
import os
import json
import base64
from calendar import timegm
from time import gmtime

from voigt.drawing import countplot, areaplot, curveplot, sumcurveplot
from voigt.aggregate import generate_output_file
from .extract import parse_file
from .worker import conn
from .server import app
from .extract import read_input

from rq import Queue


if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

# Redis queue for asynchronous processing
q = Queue(connection=conn)


@app.callback(
    Output('result-status', 'children'),
    [Input('interval', 'n_intervals')]
)
def poll_and_update_on_processing(n_intervals):
    if n_intervals == 0:
        return None

    def _check_for_output(n_intervals):
        # return os.exists(join(BASE_DIR, 'output', 'output.csv'))
        if isfile(join(BASE_DIR, 'output', 'output.csv')):
            return True
        else:
            return False

    if _check_for_output(n_intervals):
        return 'The output file is ready!'
    else:
        return 'The output file isn\'t ready yet :('


@app.callback(
    Output('files', 'options'),
    [Input('parse-data-and-refresh-chart', 'n_clicks')]
)
def set_file_options(n_clicks):

    models = read_input()
    filenames = models.filename.unique().tolist()

    return [{'label': fn, 'value': fn} for fn in filenames]


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def upload(list_of_contents, list_of_names, list_of_dates):
    """
    Takes uploaded .txt files as input and writes them to disk.

    Provides feedback to the user.

    """

    def _clean():
        for existing_file in os.listdir(join(BASE_DIR, 'input')):
            if existing_file != '.hold':
                os.remove(join(BASE_DIR, 'input', existing_file))

    try:
        if list_of_contents:

            _clean()

            written = list()

            # Write uploaded files to BASE_DIR/input
            for i, c in enumerate(list_of_contents):

                if not list_of_names[i].endswith('.txt'):
                    raise Exception(f'File {list_of_names[i]} must be .txt')

                s = c.split(',')[1]

                try:
                    s = base64.b64decode(s).decode()
                except UnicodeDecodeError:
                    raise Exception(f'Error uploading file {list_of_names[i]}. Please check file format and try again.')

                with open(join(BASE_DIR, 'input', list_of_names[i]), 'w') as f:
                    f.write(s)

                try:
                    parse_file(join(BASE_DIR, 'input', list_of_names[i]))
                except Exception:
                    raise Exception(f'Cannot parse file {list_of_names[i]}')

                written.append(list_of_names[i])

            res = [html.Li(x) for x in written]
            res.insert(0, html.P(f'Success! {len(written)} .txt files were uploaded.'))
            return res

    except Exception as e:
        _clean()
        return f'An error occurred while uploading files: {e}'


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
    if bin_width == 100 and chart_type == 'curve' and areas_state is None:
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
    if add == 0 and remove == 0 and chart_type == 'curve':
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
        Input('parse-data-and-refresh-chart', 'n_clicks'),
        Input('include-negative', 'value'),
        Input('files', 'value')
    ],
    [
        State('state', 'children'),
        State('areas-state', 'children'),
    ]
)
def update_plot(bin_width, scale, chart_type, refresh_clicks, include_negative, filename, state, areas_state):
    funcs = {'count': countplot, 'area': areaplot,
             'curve': curveplot, 'sumcurve': sumcurveplot}

    models = read_input()
    if filename:
        models = models.loc[models.filename == filename]

    kwargs = dict(
        bin_width=bin_width,
        DATA=models,
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

    if include_negative:
        print('DISPLAYING NEGATIVE MODELS')
        kwargs['exclude_negative'] = False
    else:
        print('NOT displaying negative models')

    return funcs[chart_type](**kwargs)


@app.callback(
    Output('dl_link', 'content'),
    [Input('submit', 'n_clicks')],
    [State('state', 'children')]
)
def submit(n_clicks, state):
    if n_clicks is None:
        return None
    if n_clicks is not None:
        splits = json.loads(state)['splits']
        result = q.enqueue(generate_output_file, splits, read_input())
        return 'Please wait while data is processed...'


@app.server.route('/dash/download')
def download_csv():
    return send_file(join(BASE_DIR, 'output', 'output.csv'),
                     mimetype='text/csv',
                     attachment_filename=f'output_{int(timegm(gmtime()))}.csv',
                     as_attachment=True)
