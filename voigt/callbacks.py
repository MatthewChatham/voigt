from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_bootstrap_components as dbc
from flask import send_file

from os.path import join, isdir, isfile, exists
import os
import json
import base64
import pandas as pd
import flask
import shutil
from urllib.parse import quote
import sqlite3
from sqlalchemy import create_engine
import time

from voigt.drawing import countplot, curveplot, sumcurveplot, emptyplot
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
    eng = create_engine(DATABASE_URL)
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
    [Input('interval', 'n_intervals')],
    [
        State('session-id', 'children'),
        State('jobs', 'children')
    ]
)
def poll_and_update_on_processing(n_intervals, session_id, jobs):
    if n_intervals == 0:
        return '#', {'display': 'none'}, '#', {'display': 'none'}, ''

    res = None

    dbconn = eng.connect() if os.environ.get(
        'STACK') else sqlite3.connect('output.db')

    query = f'select distinct table_name \
    as name from information_schema.tables' \
        if os.environ.get('STACK') \
        else 'select distinct name from sqlite_master'

    def _check_for_output(n_intervals, dbconn):

        df = pd.read_sql(query, con=dbconn, columns=['name'])
        # print(df.columns)
        # print('NAMES', df)

        if jobs and any(df.name.str.contains(f'output_{session_id}_{jobs[-1]}')):
            return True
        else:
            return False

    if _check_for_output(n_intervals, dbconn):
        # Allow user to download the results of the most recent job
        df = pd.read_sql(f'select * from output_{session_id}_{jobs[-1]}',
                         con=dbconn)
        df.rename({'index': 'filename'}, axis=1, inplace=True)
        csv_string = df.to_csv(index=False)

        outputdir = join(BASE_DIR, 'output',
                         f'output_{session_id}', f'job_{jobs[-1]}')
        imagedir = join(outputdir, 'images')
        # don't download if imagedir already full
        # download s3 images
        if len(os.listdir(imagedir)) > 0:
            # print(os.listdir(imagedir))
            if not isfile(join(imagedir, 'images.zip')):
                shutil.make_archive(join(outputdir, 'images'),
                                    'zip', imagedir)
                # for f in os.listdir(imagedir):
                #     os.unlink(join(imagedir, f))
            # {'display': 'none'}
            res = ("data:text/csv;charset=utf-8," + quote(csv_string), {},
                   f'/dash/download?session_id={session_id}&job_id={jobs[-1]}', {},
                   'Your results are ready!')
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
        print(f'found S3 buckets {bucket.objects.all()}')
        for s3_object in bucket.objects.all():
            do_zip = False
            # Need to split s3_object.key into path and file name, else it will
            # give error file not found.
            if f'output_{session_id}_{jobs[-1]}' not in s3_object.key:
                continue
            path, filename = os.path.split(s3_object.key)

            fpth = join(imagedir, filename)
            
            if not isfile(fpth):
                print(f'downloading file {s3_object.key}')
                bucket.download_file(s3_object.key, fpth)
                do_zip = True

        if do_zip:
            shutil.make_archive(join(outputdir, 'images'), 'zip', imagedir)

        res = ("data:text/csv;charset=utf-8," + quote(csv_string), {},
               f'/dash/download?session_id={session_id}&job_id={jobs[-1]}', {},
               dbc.Alert('Your results are ready!', color='success'))
    else:
        registry = StartedJobRegistry('default', connection=conn)
        input_dir = join(BASE_DIR, 'input', f'input_{session_id}')
        # TODO concurrency? what if mutliple ppl use app at same time?
        if (jobs and jobs[-1] not in registry.get_job_ids()) or not jobs:
            msg = dbc.Alert('Ready.', color='primary') if isdir(
                input_dir) \
                else dbc.Alert('Upload some TGA files first!', color='warning')
            res = ('#', {'display': 'none'}, '#', {'display': 'none'}, msg)
        elif jobs and jobs[-1] in registry.get_job_ids():
            res = ('#', {'display': 'none'}, '#', {'display': 'none'},
                   dbc.Alert(
                [
                    'Please wait while your request is processed.',
                    dbc.Spinner(type='grow')
                ],
                color='danger')
            )

    dbconn.close()
    return res


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
               State('session-id', 'children'),
               State('jobs', 'children')
               ]
              )
def upload(list_of_contents, list_of_names, list_of_dates, session_id, job_id):
    """
    Takes uploaded .txt files as input and writes them to disk.

    Provides feedback to the user.

    """

    print('UPLOAD')

    if session_id is not None and list_of_contents is None:
        print(f'Running in session {session_id}')

    # make a subdirectory for this session if one doesn't exist
    input_dir = join(BASE_DIR, 'input', f'input_{session_id}')
    if not isdir(input_dir):
        try:
            os.mkdir(input_dir)
        except FileExistsError:
            pass

    # Create an output directory for this session if it doesn't exist
    output_dir = join(BASE_DIR, 'output', f'output_{session_id}')
    if not isdir(output_dir):
        try:
            os.mkdir(output_dir)
            os.mkdir(join(output_dir, 'images'))
        except FileExistsError:
            pass

    def _clean_input_dir():
        """
        Clean the input directory by removing every existing file.
        """
        for existing_file in os.listdir(input_dir):
            if existing_file != '.hold':
                os.remove(join(input_dir, existing_file))

    try:

        # If the user isn't uplaoding anything and
        # hasn't uploaded anything, ask them to do so.
        if list_of_contents is None and len(os.listdir(input_dir)) == 0:
            return 'Please upload some files.'

        # if the user is uploading something, first clean the input directory,
        # then write the uploaded files to BASE_DIR/input/input_{session_id}
        if list_of_contents:

            _clean_input_dir()

            # Save successfully uploaded filenames here
            written = list()

            # Write uploaded files to BASE_DIR/input/input_{session_id}
            # If any of the files do not end in .txt,
            # or cannot be decoded properly, or cannot be parsed
            # into Voigt models, then clean the input directory and print
            # the error message. Otherwise, show a bullet list of files
            # uploaded to the input directory.

            for i, c in enumerate(list_of_contents):

                if not list_of_names[i].endswith('.txt'):
                    raise Exception(f'File {list_of_names[i]} must be .txt')

                s = c.split(',')[1]

                try:
                    s = base64.b64decode(s).decode()
                except UnicodeDecodeError:
                    raise Exception(f'Error uploading file {list_of_names[i]}.\
                     Please check file format and try again.')

                with open(join(input_dir, list_of_names[i]), 'w') as f:
                    f.write(s)

                try:
                    parse_file(join(input_dir, list_of_names[i]))
                except Exception:
                    raise Exception(f'Cannot parse file {list_of_names[i]}')

                written.append(list_of_names[i])

            res = [html.Li(x) for x in written]
            res.insert(0, html.P(f'Success! {len(written)} \
                .txt files were uploaded.'))
            return res

    except Exception as e:
        # If any of the files raise an error (wrong extension,
        # decoding error, error parsing into models),
        # then print the error message.
        _clean_input_dir()
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
def update_plot(bin_width, scale, chart_type, refresh_clicks,
                include_negative, filename, state, areas_state, session_id):
    include_negative = include_negative == 'True'

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
        print('DISPLAYING NEGATIVE MODELS')
        kwargs['exclude_negative'] = False
    else:
        print('NOT displaying negative models')
        pass

    return funcs[chart_type](**kwargs)


# @app.callback(
#     Output('jobs', 'children'),
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
    Output('jobs', 'children'),
    [Input('submit', 'n_clicks')],
    [
        State('state', 'children'),
        State('session-id', 'children'),
        State('jobs', 'children')
    ]
)
def submit(n_clicks, state, session_id, jobs):
    job_id = str(int(round(time.time() * 1000)))
    registry = StartedJobRegistry('default', connection=conn)
    running_jobs = registry.get_job_ids()
    if job_id in running_jobs:
        print('JOB FOUND IN QUEUE')
        return jobs

    input_dir = join(BASE_DIR, 'input', f'input_{session_id}')
    user_has_uploaded = exists(input_dir) and len(os.listdir(input_dir)) > 0
    if n_clicks is not None and user_has_uploaded:

        splits = json.loads(state)['splits']
        models = read_input(session_id)

        if len(models) > 0:
            q.enqueue(generate_output_file, splits,
                      models, session_id, job_id, job_id=job_id)
            jobs.append(job_id)

            outputdir = join(BASE_DIR, 'output',
                             f'output_{session_id}', f'job_{job_id}')
            os.mkdir(outputdir)
            os.mkdir(join(outputdir, 'images'))

    # print('STARTED JOBS:', jobs)
    return jobs


@app.server.route('/dash/download')
def download_csv():
    session_id = flask.request.args.get('session_id')
    job_id = flask.request.args.get('job_id')
    f = join(BASE_DIR, 'output', f'output_{session_id}', f'job_{job_id}', 'images.zip')
    if isfile(f):
        return send_file(f,
                         mimetype='application/zip',
                         attachment_filename=f'images_{session_id}.zip',
                         as_attachment=True
                         )
    else:
        print(f'File not found: {f}')


# @app.callback(
#     dash.dependencies.Output('download-link', 'href'),
#     [dash.dependencies.Input('field-dropdown', 'value')])
# def update_download_link(filter_value):
#     dff = filter_data(filter_value)
#     csv_string = dff.to_csv(index=False, encoding='utf-8')
#     csv_string = "data:text/csv;charset=utf-8," + urllib.quote(csv_string)
#     return csv_string
