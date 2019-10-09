"""
Callbacks for chart and chart controls/feedback.
"""
import os
from os.path import join
import json
from rq import Queue
from sqlalchemy import create_engine
from dash.dependencies import Input, Output, State

from ..worker import conn
from ..server import app
from ..common.extract import read_input
from ..common.drawing import countplot, areaplot, curveplot, sumcurveplot, emptyplot

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
    Output('files', 'options'),
    [Input('parse-data-and-refresh-chart', 'n_clicks')
     ], [State('session-id', 'children')]
)
def set_file_options(n_clicks, session_id):

    pth = join(BASE_DIR, 'input', f'input_{session_id}', 'models.csv')
    try:
        models = pd.read_csv(pth)
    except Exception:
        models = read_input(session_id)
        # models.to_csv()
    if len(models) == 0:
        return []

    filenames = models.filename.unique().tolist()

    res = [{'label': fn, 'value': fn} for fn in filenames]
    return res


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
        'area': areaplot,
        'curve': curveplot,
        'sumcurve': sumcurveplot
    }

    pth = join(BASE_DIR, 'input', f'input_{session_id}', 'models.csv')
    try:
        models = pd.read_csv(pth)
    except Exception:
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
        session_id=session_id,
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
