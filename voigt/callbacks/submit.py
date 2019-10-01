"""
Submit button callback.
"""
import os
import time
import json
from rq import Queue
from os.path import join, exists
from sqlalchemy import create_engine
from rq.registry import StartedJobRegistry
from dash.dependencies import Input, Output, State

from ..common.aggregate import generate_output_file
from ..worker import conn
from ..server import app
from ..common.extract import read_input

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
                      models, session_id, job_id, job_id=job_id, job_timeout=60*60)
            jobs.append(job_id)

            outputdir = join(BASE_DIR, 'output',
                             f'output_{session_id}', f'job_{job_id}')
            os.mkdir(outputdir)
            os.mkdir(join(outputdir, 'images'))

    # print('STARTED JOBS:', jobs)
    return jobs
