from dash.dependencies import Input, Output, State
import dash_html_components as html
from flask import send_file

from os.path import join, isdir, isfile
import os
import json
import base64
import pandas as pd
import flask
import shutil
from urllib.parse import quote
import sqlite3
from sqlalchemy import create_engine

from voigt.drawing import countplot, areaplot, curveplot, sumcurveplot, emptyplot
from voigt.aggregate import generate_output_file
from .extract import parse_file
from .worker import conn
from .server import app
from .extract import read_input
from .amazon import get_s3

from rq import Queue
from rq.registry import StartedJobRegistry

# from flask_sqlalchemy import SQLAlchemy


if os.environ.get('STACK'):
    print('RUNNING ON HEROKU')
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'


# Redis queue for asynchronous processing
q = Queue(connection=conn)


@app.callback(
    [
        Output('dl_link', 'href'),
        Output('dl_link', 'style'),
        Output('dl_link_images', 'href'),
        Output('dl_link_images', 'style'),
        Output('feedback', 'children')
    ],
    [Input('interval', 'n_intervals')], [State('session-id', 'children')]
)
def poll_and_update_on_processing(n_intervals, session_id):
    if n_intervals == 0:
        return '#', {'display': 'none'}, '#', {'display': 'none'}, ''

    dbconn = create_engine(DATABASE_URL).connect() if os.environ.get(
        'STACK') else sqlite3.connect('output.db')

    query = f'select distinct table_name as name from information_schema.tables' if os.environ.get('STACK') else 'select distinct name from sqlite_master'

    def _check_for_output(n_intervals, dbconn):

        df = pd.read_sql(query, con=dbconn, columns=['name'])
        # print(df.columns)
        # print('NAMES', df)

        if f'output_{session_id}' in df.name.values:
            # print('True!')
            return True
        else:
            # print('False')
            return False

    if _check_for_output(n_intervals, dbconn):
        df = pd.read_sql(f'select * from output_{session_id}', con=dbconn)
        df.rename({'index': 'filename'}, axis=1, inplace=True)
        csv_string = df.to_csv(index=False)
        dbconn.close()

        imagedir = join(BASE_DIR, 'output', f'output_{session_id}', 'images')
        outputdir = join(BASE_DIR, 'output', f'output_{session_id}')
        # don't download if imagedir already full
        # download s3 images
        if len(os.listdir(imagedir)) > 0:
            # print(os.listdir(imagedir))
            if not isfile(join(imagedir, 'images.zip')):
                shutil.make_archive(join(outputdir, 'images'),
                                    'zip', imagedir)
            # {'display': 'none'}
            return "data:text/csv;charset=utf-8," + quote(csv_string), {}, f'/dash/download?session_id={session_id}', {}, 'Your results are ready!'
        if not os.environ.get('STACK'):
            with open(join(BASE_DIR, '.aws'), 'r') as f:
                print('getting aws creds')
                creds = json.loads(f.read())
                AWS_ACCESS = creds['access']
                AWS_SECRET = creds['secret']
        else:
            AWS_ACCESS = os.environ['AWS_ACCESS']
            AWS_SECRET = os.environ['AWS_SECRET']
        s3 = get_s3(AWS_ACCESS, AWS_SECRET)
        # select bucket
        bucket = s3.Bucket('voigt')
        # download file into current directory
        for s3_object in bucket.objects.all():
            # Need to split s3_object.key into path and file name, else it will
            # give error file not found.
            if f'output_{session_id}' not in s3_object.key:
                continue
            print(f'downloading file {s3_object.key}')
            path, filename = os.path.split(s3_object.key)
            bucket.download_file(s3_object.key, join(imagedir, filename))
        # zip images

        shutil.make_archive(join(outputdir, 'images'), 'zip', imagedir)

        # {'display': 'none'}
        return "data:text/csv;charset=utf-8," + quote(csv_string), {}, f'/dash/download?session_id={session_id}', {}, 'Your results are ready!'
    else:
        # q = Queue(connection=conn)
        dbconn.close()
        registry = StartedJobRegistry('default', connection=conn)
        if len(registry.get_job_ids()) == 0:
            print('found nothing in queue')
            return '#', {'display': 'none'}, '#', {'display': 'none'}, 'Ready.'
        else:
            print('found something in queue')
            return '#', {'display': 'none'}, '#', {'display': 'none'}, 'Please wait while your request is processed....'


@app.callback(
    Output('files', 'options'),
    [Input('parse-data-and-refresh-chart', 'n_clicks')
     ], [State('session-id', 'children')]
)
def set_file_options(n_clicks, session_id):

    models = read_input(session_id)
    if len(models) == 0:
        return []

    filenames = models.filename.unique().tolist()

    res = [{'label': fn, 'value': fn} for fn in filenames]
    return res


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('session-id', 'children')
               ]
              )
def upload(list_of_contents, list_of_names, list_of_dates, session_id):
    """
    Takes uploaded .txt files as input and writes them to disk.

    Provides feedback to the user.

    """

    if list_of_contents is None:
        return ''

    # make a subdirectory for this session if one doesn't already exist
    input_dir = join(BASE_DIR, 'input', f'input_{session_id}')
    if not isdir(input_dir):
        try:
            os.mkdir(input_dir)
        except FileExistsError:
            pass

    output_dir = join(BASE_DIR, 'output', f'output_{session_id}')
    if not isdir(output_dir):
        try:
            os.mkdir(output_dir)
            os.mkdir(join(output_dir, 'images'))
        except FileExistsError:
            pass

    def _clean():
        for existing_file in os.listdir(input_dir):
            if existing_file != '.hold':
                os.remove(join(input_dir, existing_file))

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

                with open(join(input_dir, list_of_names[i]), 'w') as f:
                    f.write(s)

                try:
                    parse_file(join(input_dir, list_of_names[i]))
                except Exception:
                    raise Exception(f'Cannot parse file {list_of_names[i]}')

                written.append(list_of_names[i])

            res = [html.Li(x) for x in written]
            res.insert(0, html.P(f'Success! {len(written)} .txt files were uploaded.'))
            return res

    except Exception as e:
        _clean()
        return f'An error occurred while uploading files: {e}', {'display': 'none'}


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
def update_state(add, remove, split_point, state, figure, bin_width, chart_type):

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
        State('session-id', 'children')
    ]
)
def update_plot(bin_width, scale, chart_type, refresh_clicks, include_negative, filename, state, areas_state, session_id):
    funcs = {
        'count': countplot,
        # 'area': areaplot,
        'curve': curveplot,
        'sumcurve': sumcurveplot
    }

    models = read_input(session_id)
    if len(models) == 0:
        # print('returning emptyplot')
        return emptyplot()
    if filename:
        models = models.loc[models.filename == filename]

    kwargs = dict(
        bin_width=bin_width,
        DATA=models,
        scale=scale,
        shapes=[],  # construct_shapes(split_point=split_point),
    )

    # if areas_state is not None and chart_type == 'area':

    #     areas_state = json.loads(areas_state)

    #     cached = areas_state['areas'].get(str(bin_width)) is not None
    #     if cached:
    #         print('FOUND CACHED AREA')
    #         areas = areas_state['areas'][str(bin_width)]
    #         kwargs['areas'] = areas

    if include_negative:
        # print('DISPLAYING NEGATIVE MODELS')
        kwargs['exclude_negative'] = False
    else:
        # print('NOT displaying negative models')
        pass

    return funcs[chart_type](**kwargs)


# @app.callback(
#     Output('submit-state', 'children'),
#     [Input('submit', 'n_clicks'), Input('interval', 'n_intervals')]

# )
# def update_submit_state(n_clicks, n_intervals):
#     if n_clicks is None:
#         return 'ready'
#     elif len(q.job_ids) > 0:
#         return 'processing'
#     else:
#         return 'done'

# @app.callback()
# def update_feedback():
#     pass


@app.callback(
    Output('submit-state', 'children'),
    [Input('submit', 'n_clicks')],
    [
        State('state', 'children'),
        State('session-id', 'children'),
        State('submit-state', 'children')
    ]
)
def submit(n_clicks, state, session_id, submit_state):
    # if submit_state is None:
    #     return ''
    # elif submit_state == 'processing':
    #     return 'Please wait until the current job is finished!'
    # elif submit_state == 'done':
    #     return 'Your results are ready!'
    # elif submit_state == 'ready':
    registry = StartedJobRegistry('default', connection=conn)
    if n_clicks is not None and len(registry.get_job_ids()) == 0:
        splits = json.loads(state)['splits']
        models = read_input(session_id)
        if len(models) == 0:
            return 'Please upload some TGA files first!'
        q.enqueue(generate_output_file, splits, models, session_id)
        return 'processing'


@app.server.route('/dash/download')
def download_csv():
    session_id = flask.request.args.get('session_id')
    return send_file(join(BASE_DIR, 'output', f'output_{session_id}', 'images.zip'),
                     mimetype='application/zip',
                     attachment_filename=f'images_{session_id}.zip',
                     as_attachment=True
                     )


# @app.callback(
#     dash.dependencies.Output('download-link', 'href'),
#     [dash.dependencies.Input('field-dropdown', 'value')])
# def update_download_link(filter_value):
#     dff = filter_data(filter_value)
#     csv_string = dff.to_csv(index=False, encoding='utf-8')
#     csv_string = "data:text/csv;charset=utf-8," + urllib.quote(csv_string)
#     return csv_string
