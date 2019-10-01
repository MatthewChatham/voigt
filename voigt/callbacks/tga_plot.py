"""
Callbacks for TGA chart and dropdown.
"""
import os
from rq import Queue
from sqlalchemy import create_engine
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from ..worker import conn
from ..server import app
from ..peak_fitting import read_data

if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
    DATABASE_URL = os.environ['DATABASE_URL']
    eng = create_engine(DATABASE_URL)
else:
    env = 'Dev'
    BASE_DIR = 'C:\\Users\\Administrator\\Desktop\\voigt'

q = Queue(connection=conn)


# @app.callback(
#     Output('tga-plot-dropdown', 'options'),
#     [Input('tga-parse-data-and-refresh-chart', 'n_clicks')
#      ], [State('session-id', 'children')]
# )
# def set_tga_file_options(n_clicks, session_id):
#     if n_clicks is None:
#         return []
#     print('clicked button')
#     input_dir = os.path.join(BASE_DIR, 'input', f'input_{session_id}', 'fitting')
#     if len(os.listdir(input_dir)) == 0:
#         print('input dir is empty')
#         return []

#     return [{'label': f, 'value': f} for f in os.listdir(input_dir)]


@app.callback(
    Output('tga-plot', 'figure'),
    [
        Input('tga-plot-dropdown', 'value'),
        Input('pos-range-min', 'value'), Input('pos-range-max', 'value'),
        Input('neg-range-min', 'value'), Input('neg-range-max', 'value'),
        Input('negative-peaks', 'values')
    ],
    [State('session-id', 'children'), State('file-format',
                                            'value'), State('tga-plot', 'figure'), ]
)
def update_plot(file, pos_min, pos_max, neg_min, neg_max, neg_peaks, session_id, format, figure):
    if file is None:
        return {}
    input_dir = os.path.join(BASE_DIR, 'input', f'input_{session_id}', 'fitting')
    fn = os.path.join(input_dir, file)

    tmp = read_data(fn, format=format)
    temp, mass = tmp[2], tmp[3]

    data = list()

    trace = go.Scatter(
        x=temp,
        y=mass,
        mode='lines',
        # name=m.filename.strip('.txt') + f'/{prefix}'
    )

    data.append(trace)

    figure = {
        'data': data,
        'layout': {
            'yaxis': dict(
                type='linear',
                autorange=True
            ),
            'shapes': [
                {
                    'type': 'rect',
                    'x0': pos_min,
                    'y0': 0,
                    'x1': pos_max,
                    'y1': max(mass),
                    'line': {
                        'color': 'rgba(200, 247, 197, 1)',
                        'width': 2,
                    },
                    'fillcolor': 'rgba(200, 247, 197, 0.5)',
                }
            ]
        }
    }

    show_neg_range = len(neg_peaks) != 0
    if show_neg_range:
        figure['layout']['shapes'].append(

            {
                'type': 'rect',
                'x0': neg_min,
                'y0': 0,
                'x1': neg_max,
                'y1': max(mass),
                'line': {
                        'color': 'rgba(241, 169, 160, 1)',
                        'width': 2,
                },
                'fillcolor': 'rgba(241, 169, 160, 0.5)',
            }

        )

    return figure
