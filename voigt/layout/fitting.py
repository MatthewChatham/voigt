"""
Frontend layout for Fitting tab.
"""
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

fitting = [

    html.H1("Step 1: Set Fit Parameters", style={'margin-left': '10px'}),
    # negative peaks
    dcc.Checklist(
        options=[
            {'label': 'Negative peaks', 'value': 'neg'},
        ],
        values=[]
    ),
    # Temperature range to bound negative curve fitting
    dcc.RangeSlider(
        count=1,
        min=30,
        max=1000,
        step=1,
        value=[30, 1000]
    ),
    # Temperature range to bound positive curve fitting
    dcc.RangeSlider(
        count=1,
        min=30,
        max=1000,
        step=1,
        value=[30, 1000]
    ),
    # max peak num
    dcc.Slider(min=1, max=20, step=1, value=10),
    # mass defect warning
    dcc.Slider(min=0, max=100, step=1, value=10),
    # Temperature to calculate mass loss from
    dcc.Slider(min=30, max=1000, step=1, value=60),
    # Temperature to calculate mass loss to
    dcc.Slider(min=30, max=1000, step=1, value=950),
    # run start temp
    dcc.Input(min=30, max=1000, step=1, value=60),
    # file format
    dcc.Dropdown(options=[
        {'label': 'Q500', 'value': 'Q500'},
        {'label': 'DMSM', 'value': 'DMSM'}
    ], value='Q500'),
    # amorphous carbon temperature
    dcc.Slider(min=30, max=1000, step=1, value=450),


    html.Hr(),
]


# negative peaks: no
# Temperature range to bound negative curve fitting: None
# Temperature range to bound positive curve fitting: full
# max peak num: 10
# mass defect warning: 10
# Temperature to calculate mass loss from: 60
# Temperature to calculate mass loss to: 950
# run start temp: 60
# file format: Q500/DMSM
# amorphous carbon temperature: 450
