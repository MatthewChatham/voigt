"""
Callbacks for upload and download (I/O, Input/Output).
"""
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_bootstrap_components as dbc
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

from ..common.extract import parse_file
from ..worker import conn
from ..server import app
from ..common.amazon import get_s3
from ..common.extract import read_input
from ..peak_fitting import read_data

from rq import Queue
from rq.registry import StartedJobRegistry

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
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')],
    [
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified'),
        State('session-id', 'children'),
        State('jobs', 'children')
    ]
)
def upload_analysis(list_of_contents, list_of_names, list_of_dates, session_id, job_id):
    """
    Takes uploaded .txt files as input and writes them to disk.

    Provides feedback to the user.

    """

    print('UPLOAD')

    if session_id is not None and list_of_contents is None:
        print(f'Running in session {session_id}')

    # make a subdirectory for this session if one doesn't exist
    input_dir = join(BASE_DIR, 'input', f'input_{session_id}')
    try:
        os.mkdir(input_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir(join(input_dir, 'analysis'))
    except FileExistsError:
        pass

    # Create an output directory for this session if it doesn't exist
    output_dir = join(BASE_DIR, 'output', f'output_{session_id}')
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir(join(output_dir, 'analysis'))
    except FileExistsError:
        pass

    try:
        os.mkdir(join(output_dir, 'analysis', 'images'))
    except FileExistsError:
        pass

    def _clean_input_dir():
        """
        Clean the input directory by removing every existing file.
        """
        for existing_file in os.listdir(join(input_dir, 'analysis')):
            if existing_file != '.hold':
                os.remove(join(input_dir, 'analysis', existing_file))

    try:

        # If the user isn't uplaoding anything and
        # hasn't uploaded anything, ask them to do so.
        # print(os.listdir(input_dir))
        if list_of_contents is None and len(os.listdir(join(input_dir, 'analysis'))) == 0:
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

            peaks = pd.DataFrame()

            for i, c in enumerate(list_of_contents):

                if not list_of_names[i].endswith('.txt'):
                    raise Exception(f'File {list_of_names[i]} must be .txt')

                s = c.split(',')[1]

                try:
                    s = base64.b64decode(s).decode()
                except UnicodeDecodeError:
                    raise Exception(f'Error uploading file {list_of_names[i]}.\
                     Please check file format and try again.')

                with open(join(input_dir, 'analysis', list_of_names[i]), 'w') as f:
                    f.write(s)

                try:
                    parsed_file = parse_file(join(input_dir, 'analysis', list_of_names[i]))
                    peaks = pd.concat([peaks, parsed_file], sort=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise Exception(f'Cannot parse file {list_of_names[i]}: {e}')

                written.append(list_of_names[i])

            res = [html.Li(x) for x in written]
            res.insert(0, html.P(f'Success! {len(written)} \
                .txt files were uploaded.'))

            # peaks = read_input(session_id)

            def compute_models(DATA):
                res = pd.DataFrame([], columns=['filename', 'peak_name', 'peak_position', 'amplitude'])
                for idx, (_, model) in enumerate(DATA.iterrows()):

                    row = pd.Series()
                    row['filename'] = model.filename
                    row['peak_name'] = model.variable
                    row['peak_position'] = model.value
                    
                    amp_col = model.variable[:model.variable.index('_')] + '_amplitude'
                    row['amplitude'] = model[amp_col]

                    res.loc[idx] = row

                return res

            models = compute_models(peaks)

            models.to_csv(join(input_dir, 'models.csv'))

            peaks.to_csv(join(input_dir, 'peaks.csv'))

            return res

    except Exception as e:
        # If any of the files raise an error (wrong extension,
        # decoding error, error parsing into models),
        # then print the error message.
        _clean_input_dir()
        return f'An error occurred while uploading files: {e}'


@app.callback(
    [
        Output('output-data-upload-fitting', 'children'),
        Output('tga-plot-dropdown', 'options'),
        Output('run-start-temp', 'value'),
        Output('mass-loss-to-temp', 'value')
    ],
    [Input('upload-data-fitting', 'contents')],
    [
        State('upload-data-fitting', 'filename'),
        State('upload-data-fitting', 'last_modified'),
        State('session-id', 'children'),
        State('jobs', 'children'),
        State('file-format', 'value')
    ]
)
def upload_fitting(list_of_contents, list_of_names, list_of_dates, session_id, job_id, format):
    """
    Takes uploaded .txt files as input and writes them to disk.

    Provides feedback to the user.

    """

    min_ = 60
    max_ = 950

    print('UPLOAD')

    if session_id is not None and list_of_contents is None:
        print(f'Running in session {session_id}')

    # make a subdirectory for this session if one doesn't exist
    input_dir = join(BASE_DIR, 'input', f'input_{session_id}')
    input_dir_session = os.path.join(BASE_DIR, 'input', f'input_{session_id}', 'fitting')
    try:
        os.mkdir(input_dir)
        os.mkdir(join(input_dir, 'fitting'))
    except FileExistsError:
        pass

    try:
        os.mkdir(join(input_dir, 'fitting'))
    except FileExistsError:
        pass

    # Create an output directory for this session if it doesn't exist
    output_dir = join(BASE_DIR, 'output', f'output_{session_id}')
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    try:
        os.mkdir(join(output_dir, 'fitting'))
        os.mkdir(join(output_dir, 'fitting', 'images'))
    except FileExistsError:
        pass

    try:
        os.mkdir(join(output_dir, 'fitting', 'images'))
    except FileExistsError:
        pass

    def _clean_input_dir():
        """
        Clean the input directory by removing every existing file.
        """
        for existing_file in os.listdir(join(input_dir, 'fitting')):
            if existing_file != '.hold':
                os.remove(join(input_dir, 'fitting', existing_file))

    try:

        # If the user isn't uplaoding anything and
        # hasn't uploaded anything, ask them to do so.
        if list_of_contents is None and len(os.listdir(join(input_dir, 'fitting'))) == 0:
            return 'Please upload some files. Note: Refreshing page will \
                                    remove input files. \
                                    Uploading multiple times will first \
                                    remove all existing files.', [], min_, max_

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

                    s = base64.b64decode(s)
                    import chardet
                    encoding = chardet.detect(s)['encoding']
                    if encoding == 'UTF-16':
                        s = s.decode('utf-16')
                    elif encoding == 'ascii':
                        s = s.decode('utf-8')
                    else:
                        s = s.decode(encoding)
                except UnicodeDecodeError as e:
                    print(e)
                    raise Exception(f'Error uploading file {list_of_names[i]}.\
                     Please check file format and try again.')

                with open(join(input_dir, 'fitting', list_of_names[i]), 'w') as f:
                    f.write(s)

                try:
                    d = read_data(join(input_dir, 'fitting',
                                       list_of_names[i]), format)
                    temp = d[2]
                    print(temp[0])
                    min_ = max(min_, int(temp[0]) + 1)
                    max_ = min(max_, int(temp[-1]) - 1)
                except Exception as e:
                    raise Exception(f'Cannot parse file {list_of_names[i]}: {e}')

                written.append(list_of_names[i])

            res = [html.Li(x) for x in written]
            res.insert(0, html.P(f'Success! {len(written)} \
                .txt files were uploaded.'))
            return res, [{'label': f, 'value': f} for f in os.listdir(input_dir_session)], min_, max_

    except Exception as e:
        # If any of the files raise an error (wrong extension,
        # decoding error, error parsing into models),
        # then print the error message.
        _clean_input_dir()
        import traceback
        traceback.print_exc()
        return f'An error occurred while uploading files: {e}', [], min_, max_


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
            print('found output')
            return True
        else:
            print('didn\'t find output')
            return False

    if _check_for_output(n_intervals, dbconn):
        # Allow user to download the results of the most recent job
        df = pd.read_sql(f'select * from output_{session_id}_{jobs[-1]}',
                         con=dbconn)
        df.rename({'index': 'filename'}, axis=1, inplace=True)
        csv_string = df.to_csv(index=False)

        output = join(BASE_DIR, 'output')
        session = join(output, f'output_{session_id}')
        analysis = join(session, 'analysis')
        job = join(analysis, f'job_{jobs[-1]}')
        images = join(job, 'images')

        for dir_ in [output, session, analysis, job, images]:
            if not isdir(dir_):
                os.mkdir(dir_)

        # don't download if imagedir already full
        # download s3 images
        if len(os.listdir(images)) > 0:
            # print(os.listdir(imagedir))
            if not isfile(join(job, 'images.zip')):
                tmp = shutil.make_archive(images, 'zip', images)
                print(f'made archive {tmp}')
                # for f in os.listdir(imagedir):
                #     os.unlink(join(imagedir, f))
            # {'display': 'none'}
            res = ("data:text/csv;charset=utf-8," + quote(csv_string), {},
                   f'/dash/download?session_id={session_id}&job_id={jobs[-1]}', {},
                   dbc.Alert('Your results are ready!', color='success'))

        else:

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
                print(f'found S3 object {s3_object.key}')
                do_zip = False
                # Need to split s3_object.key into path and file name, else it will
                # give error file not found.
                if f'output_{session_id}/job_{jobs[-1]}' not in s3_object.key:
                    continue
                path, filename = os.path.split(s3_object.key)

                fpth = join(images, filename)

                print(f'checking for file {s3_object.key}')
                if not isfile(fpth):
                    print(f'downloading file {s3_object.key}')
                    bucket.download_file(s3_object.key, fpth)
                    do_zip = True

            if do_zip:
                tmp = shutil.make_archive(images, 'zip', images)
                print(f'made archive {tmp}')

            res = ("data:text/csv;charset=utf-8," + quote(csv_string), {},
                   f'/dash/download?session_id={session_id}&job_id={jobs[-1]}', {},
                   dbc.Alert('Your results are ready!', color='success'))
    else:
        registry = StartedJobRegistry('default', connection=conn)
        input_dir = join(BASE_DIR, 'input', f'input_{session_id}', 'analysis')
        # TODO concurrency? what if mutliple ppl use app at same time?
        if (jobs and jobs[-1] not in registry.get_job_ids()) or not jobs:
            msg = dbc.Alert('Ready.', color='primary') if len(os.listdir(input_dir)) > 0 \
                else dbc.Alert('Upload some peak files first!', color='warning')
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


@app.server.route('/dash/download')
def download_img():
    session_id = flask.request.args.get('session_id')
    job_id = flask.request.args.get('job_id')
    f = join(BASE_DIR, 'output', f'output_{session_id}', 'analysis', f'job_{job_id}', 'images.zip')
    if isfile(f):
        return send_file(f,
                         mimetype='application/zip',
                         attachment_filename=f'images_{session_id}.zip',
                         as_attachment=True
                         )
    else:
        print(f'File not found: {f}')


@app.server.route('/dash/download-fit')
def download_fitting():
    session_id = flask.request.args.get('session_id')
    job_id = flask.request.args.get('job_id')
    f = join(BASE_DIR, 'output', f'output_{session_id}', 'fitting', f'job_{job_id}.zip')
    if isfile(f):
        return send_file(f,
                         mimetype='application/zip',
                         attachment_filename=f'fitting_{session_id}.zip',
                         as_attachment=True
                         )
    else:
        print(f'File not found: {f}')


@app.callback(
    [Output('hist-download-1', 'href'), Output('hist-download-1', 'download'), Output('hist-download-2', 'href'), Output('hist-download-2', 'download')],
    [Input('refresh-download', 'n_clicks')], [State('session-id', 'children')]
)
def download_peaks_and_areas(n_clicks, session_id):

    df = pd.read_csv(join(BASE_DIR, 'input', f'input_{session_id}', 'peaks.csv'))
    res = pd.DataFrame([], columns=['filename', 'peak_name', 'peak_position', 'amplitude'])
    for idx, (_, model) in enumerate(df.iterrows()):

        row = pd.Series()
        row['filename'] = model.filename
        row['peak_name'] = model.variable
        row['peak_position'] = model.value
        
        amp_col = model.variable[:model.variable.index('_')] + '_amplitude'
        row['amplitude'] = model[amp_col]

        res.loc[idx] = row

    csv_string = res.to_csv(index=False)

    res = "data:text/csv;charset=utf-8," + quote(csv_string)

    return res, 'peaks.csv', f'/dash/download-hist?session_id={session_id}', 'histogram.csv'


@app.server.route('/dash/download-hist')
def download_histogram():
    session_id = flask.request.args.get('session_id')
    
    f = join(BASE_DIR, 'output', f'output_{session_id}', 'histogram.csv')
    
    if isfile(f):
        return send_file(f,
                         mimetype='application/csv',
                         attachment_filename='histogram.csv',
                         as_attachment=True
                         )
    else:
        print(f'File not found: {f}')