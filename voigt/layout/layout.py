"""
Frontend layout composed of two tabs: Fitting and Analysis.
"""
import time
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from analysis import analysis
from fitting import fitting


footer = html.Div(
    [
        html.Span(['Copyright © 2019 ',
                   html.A('Matthew Chatham',
                          href='https://www.linkedin.com/in/matthewchatham/',
                          target="_blank")],
                  style={'top': '50%'})
    ],
    style={'width': '100%', 'height': '50px', 'text-align': 'center',
           'font-size': '14px', 'padding-top': '15px'}
)


def _layout():

    session_id = int(round(time.time() * 1000))

    state = [dcc.Interval(id='interval', interval=1 * 1000,  # milliseconds
                          n_intervals=0),
             html.Div(id='state', style={'display': 'none'}),
             html.Div(id='areas-state', style={'display': 'none'}),
             html.Div(id='bin-width-state', style={'display': 'none'}),
             html.Div(children=[], id='jobs', style={'display': 'none'}),
             html.Div(children=session_id,
                      id='session-id',
                      style={'display': 'none'}
                      ), ]

    body = dbc.Container([
        dcc.Tabs([
            dcc.Tab(analysis, label='Analysis'),
            dcc.Tab(fitting, label='Fitting'),
        ], style={'margin': '0 0 10px 0'})] + state,
        className="mt-4",
        style={'background-color': '#e8eaf6',
               'padding': '10px', 'font-size': '16px', 'min-height': '250px'}
    )

    return html.Div([body, footer])


layout = _layout
