"""
Callbacks that control peak-fitting parameters.
"""
from dash.dependencies import Input, Output, State
import dash_html_components as html
import os
from sqlalchemy import create_engine
import sqlite3
import pandas as pd
from os.path import join, exists, isfile
import dash_bootstrap_components as dbc
import time
import shutil
import json

from ..worker import conn
from ..server import app
from ..peak_fitting import main, params, read_data
from ..common.amazon import get_s3

from rq import Queue
from rq.registry import StartedJobRegistry
from rq.job import Job

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
    Output('neg-peak-range', 'style'),
    [Input('negative-peaks', 'values')],
    [State('neg-peak-range', 'style')]
)
def toggle_neg_peak_range(neg_peaks, style):
    print(style)
    if len(neg_peaks) == 0:
        style.update({'display': 'none'})
        return style
    else:
        style.pop('display', None)
        return style


@app.callback(
    Output('pos-peak-slider', 'disabled'),
    [Input('temp-range-pos-full', 'values')],
    [State('pos-peak-slider', 'style')]
)
def toggle_neg_peak_range(full, style):
    if len(full) == 0:
        return False
    else:
        return True


@app.callback(
    Output('neg-range-values', 'children'),
    [Input('neg-peak-slider', 'value')],
)
def update_neg_peak_range(value):
    return ['Temperature range to bound', html.Strong(' negative '), f'curve fitting: {value[0]}, {value[1]}']


@app.callback(
    Output('pos-range-values', 'children'),
    [Input('pos-peak-slider', 'value')],
)
def update_pos_peak_range(value):
    return ['Temperature range to bound', html.Strong(' positive '), f'curve fitting: {value[0]}, {value[1]}']


# TODO!
@app.callback(
    [
        Output('dl_link_fitting', 'href'),
        Output('dl_link_fitting', 'style'),
        Output('feedback_fitting', 'children')
    ],
    [Input('interval', 'n_intervals')],
    [
        State('session-id', 'children'),
        State('fit-jobs', 'children')
    ]
)
def poll_and_update_on_processing(n_intervals, session_id, fit_jobs):
    if n_intervals == 0:
        return '#', {'display': 'none'}, ''

    res = None

    def _check_for_output():

        if len(fit_jobs) == 0:
            return False

        job = Job.fetch(fit_jobs[-1], connection=conn)
        # print(df.columns)
        # print('NAMES', df)

        # Todo: account for failure and provide feedback to user
        print(f'Job status: {job.get_status()}')
        if job.get_status() in ['finished', 'failed']:
            return True
        else:
            return False

    if _check_for_output():

        outputdir = join(BASE_DIR, 'output',
                         f'output_{session_id}', 'fitting', f'job_{fit_jobs[-1]}')
        if not exists(outputdir):
            os.mkdir(outputdir)

        # already downloaded results
        if len(os.listdir(outputdir)) > 0:
            if not isfile(join(outputdir, 'results.zip')):
                tmp = shutil.make_archive(outputdir, 'zip', outputdir)
                print(f'made archive {tmp}')
            res = (f'/dash/download-fit?session_id={session_id}&job_id={fit_jobs[-1]}', {},
                   dbc.Alert('Your results are ready!', color='success'))

        # download results
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
            # print(f'found S3 buckets {bucket.objects.all()}')
            for s3_object in bucket.objects.all():

                do_zip = False
                # Need to split s3_object.key into path and file name, else it will
                # give error file not found.
                if f'output_{session_id}/fitting/job_{fit_jobs[-1]}' not in s3_object.key:
                    continue
                print(f'found S3 object {s3_object.key}')
                path, filename = os.path.split(s3_object.key)

                fpth = join(outputdir, filename)

                print(f'checking for file {s3_object.key}')
                if not isfile(fpth):
                    print(f'downloading file {s3_object.key}')
                    bucket.download_file(s3_object.key, fpth)
                    do_zip = True

            if do_zip:
                tmp = shutil.make_archive(outputdir, 'zip', outputdir)
                print(f'made archive {tmp}')

            res = (f'/dash/download-fit?session_id={session_id}&job_id={fit_jobs[-1]}', {},
                   dbc.Alert('Your results are ready!', color='success'))
    else:
        registry = StartedJobRegistry('default', connection=conn)
        input_dir = join(BASE_DIR, 'input', f'input_{session_id}', 'fitting')
        # TODO concurrency? what if mutliple ppl use app at same time?
        if (fit_jobs and fit_jobs[-1] not in registry.get_job_ids()) or not fit_jobs:
            msg = dbc.Alert('Ready.', color='primary') if len(os.listdir(input_dir)) > 0 \
                else dbc.Alert('Upload some TGA measurements first!', color='warning')
            res = ('#', {'display': 'none'}, msg)
        elif fit_jobs and fit_jobs[-1] in registry.get_job_ids():
            res = ('#', {'display': 'none'},
                   dbc.Alert(
                [
                    'Please wait while your request is processed.',
                    dbc.Spinner(type='grow')
                ],
                color='danger')
            )

    return res


@app.callback(
    Output('fit-jobs', 'children'),
    [Input('submit_fitting', 'n_clicks')],
    [
        State('negative-peaks', 'values'),
        State('neg-peak-slider', 'value'),
        State('pos-peak-slider', 'value'),
        State('max-peak-num', 'value'),
        State('mass-defect-warning', 'value'),
        State('mass-loss-from-temp', 'value'),
        State('mass-loss-to-temp', 'value'),
        State('run-start-temp', 'value'),
        State('file-format', 'value'),
        State('amorphous-carbon-temp', 'value'),
        State('temp-range-pos-full', 'values'),

        State('session-id', 'children'),
        State('fit-jobs', 'children')
    ]
)
def submit(n_clicks, neg_peaks, neg_peak_range,
           pos_peak_range, max_peak_num, mass_defect_warning,
           mass_loss_from, mass_loss_to, run_start_temp, file_format,
           amorph_carb_temp, full, session_id, fit_jobs):

    # runs on page load
    if n_clicks is None or n_clicks == 0:
        return fit_jobs

    fit_job_id = str(int(round(time.time() * 1000)))
    job = join(BASE_DIR, 'output', f'output_{session_id}', 'fitting', f'job_{fit_job_id}')
    if not exists(job):
        os.mkdir(job)

    # created by upload callback when page is loaded
    input_dir = join(BASE_DIR, 'input', f'input_{session_id}', 'fitting')
    user_has_uploaded = exists(input_dir) and len(os.listdir(input_dir)) > 0

    if not user_has_uploaded:
        return fit_jobs

    # create params file
    try:

        # convert checklist to boolean
        full = len(full) != 0

        # translate Dash layout to params file
        neg_peaks = 'no' if len(neg_peaks) == 0 else 'yes'
        neg_peak_range = ','.join([str(x) for x in neg_peak_range]) \
            if neg_peaks == 'yes' else 'None'
        pos_peak_range = ','.join([str(x) for x in pos_peak_range]) if not full else 'full'

        params_file_path = join(BASE_DIR, 'output', f'output_{session_id}',
                                'fitting', 'params_file.txt')

        with open(params_file_path, 'w') as f:
            f.write(params.format(neg_peaks, neg_peak_range,
                                  pos_peak_range, max_peak_num,
                                  mass_defect_warning, mass_loss_from,
                                  mass_loss_to, run_start_temp, file_format,
                                  amorph_carb_temp))
    except Exception as e:
        print(f'Error when generating parameter file: {e}')
        return fit_jobs

    # prepare data for worker process
    fnames = os.listdir(input_dir)
    data = [read_data(join(input_dir, fn), format=file_format)
            for fn in fnames]

    # send the job to the worker for processing
    # TODO : distinguish fit job IDs from analysis job IDs
    q.enqueue(main,
              args=(params_file_path,
                    fnames,
                    data,
                    input_dir,
                    session_id,
                    fit_job_id,
                    parse_params(params_file_path)),
              job_id=fit_job_id, job_timeout=1000,
              )
    fit_jobs.append(fit_job_id)

    return fit_jobs