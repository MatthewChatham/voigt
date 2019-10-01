"""
Frontend layout for Fitting tab.
"""
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

step1 = [
    html.H1('Step 1: Upload TGA Measurements', style={'margin-left': '10px'}),

    dbc.Row(
        [
            dbc.Col(
                [
                    # file format
                    html.Div(['File Format: ', dcc.Dropdown(options=[
                        {'label': 'Q500/DMSM', 'value': 'Q500/DMSM'},
                        {'label': 'TGA 5500', 'value': 'TGA 5500'},
                        {'label': 'TGA 5500 v2', 'value': 'TGA 5500 v2'},
                        {'label': 'Just Temp and Mass',
                            'value': 'Just Temp and Mass'}
                    ], value='Q500/DMSM', style={'width': '100%'}, id='file-format')], style={'margin-left': '5px'}),
                    dcc.Upload(
                        id='upload-data-fitting',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '5px',
                            'background-color': '#f2f2f2'
                        },
                        multiple=True
                    ),
                ],
                md=4,
                style={'padding': '10px'}
            ),

            dbc.Col([
                dbc.Card(dbc.CardBody(
                    "",
                    id='output-data-upload-fitting',
                    style={'min-height': '50px',
                           'overflow': 'auto',
                           'max-height': '600px',
                           'resize': 'vertical'})),
                # dbc.Button('Parse files and refresh chart & options',
                #            id='tga-parse-data-and-refresh-chart',
                #            color='primary',
                #            style={'margin-top': '5px', 'font-size': '14px'})
            ], style={'padding': '10px', 'margin-top': '25px'}),

        ], style={'margin': '25px'}),

    dbc.Row([dbc.Col([
        dcc.Dropdown(id='tga-plot-dropdown', options=[], style={
                     'width': '250px', 'margin-bottom': '15px'}, placeholder='Select a file.'),
        dcc.Graph(id='tga-plot')
    ])], style={'margin': '25px'}),

    html.Hr(),
]

step2 = [
    html.H1('Step 2: Set Fit Parameters', style={'margin-left': '10px'}),

    dbc.Row([

        dbc.Col([

            # negative peaks
            dcc.Checklist(
                options=[
                    {'label': 'Fit negative peaks?', 'value': 'neg'},
                ],
                value=[], inputStyle={'margin-right': '5px'},
                id='negative-peaks'
            ),

            # Temperature range to bound negative curve fitting
            html.Div([
                html.P('Negative Peak Range:', id='neg-range-values'),
                dcc.Checklist(
                    options=[
                        {'label': 'Full (disables input)', 'value': 'full'},
                    ],
                    value=[], inputStyle={'margin-right': '5px'},
                    id='temp-range-neg-full'
                ),
                dcc.Input(id='neg-range-min', type='number',
                          min=30, max=1000, step=1, value=200),
                dcc.Input(id='neg-range-max', type='number',
                          min=30, max=1000, step=1, value=450),
            ],
                style={'padding': '10px', 'width': '250px',
                       'margin-bottom': '5px'},
                id='neg-peak-range'),

        ], width=3),


        dbc.Col([

            # Temperature range to bound positive curve fitting
            html.Div([
                html.P('Positive Peak Range:', id='pos-range-values'),
                dcc.Checklist(
                    options=[
                        {'label': 'Full (disables input)', 'value': 'full'},
                    ],
                    value=[], inputStyle={'margin-right': '5px'},
                    id='temp-range-pos-full'
                ),
                dcc.Input(id='pos-range-min', type='number',
                          min=30, max=1000, step=1, value=450, inputMode='numeric'),
                dcc.Input(id='pos-range-max', type='number',
                          min=30, max=1000, step=1, value=850),
            ], style={'padding': '10px', 'width': '250px', 'margin-bottom': '5px', 'margin-top': '25px'}),

        ], width=3),

        dbc.Col([

            # max peak num
            dbc.Row([dbc.Col('Max Peak Num: ', width=8), dbc.Col(dcc.Input(
                min=1, max=20, step=1, value=10, type='number', id='max-peak-num'))]),

            # mass defect warning
            dbc.Row([dbc.Col('Mass Defect Warning: ', width=8), dbc.Col(dcc.Input(
                min=0, max=100, step=1, value=10, type='number', id='mass-defect-warning'))]),

            # run start temp
            dbc.Row([dbc.Col('Run Start Temp: ', width=8), dbc.Col(dcc.Input(
                min=30, max=1000, step=1, value=60, type='number', id='run-start-temp', disabled=False))]),

            # Temperature to calculate mass loss to
            dbc.Row([dbc.Col('Run End Temp: ', width=8), dbc.Col(dcc.Input(
                min=30, max=1000, step=1, value=950, type='number', id='mass-loss-to-temp', disabled=False))]),

            # amorphous carbon temperature
            dbc.Row([dbc.Col('Amorphous Carbon Temp: ', width=8), dbc.Col(dcc.Input(
                min=30, max=1000, step=1, value=450, type='number', id='amorphous-carbon-temp', disabled=False))]),


        ], width=6),

    ], style={'margin': '25px'}),

    html.Hr(),
]

step3 = [
    html.H1('Step 3: Run Peak-Fitting Routine', style={'margin-left': '10px'}),

    dbc.Row([

        dbc.Col([
            html.Div(['Job timeout: ', dcc.Input(
                min=1, max=60, step=1, value=15, type='number', id='job-timeout-mins'), ' minutes'], style={'margin-bottom': '25px'}),

            html.Span(id='feedback_fitting', style={
                "background-color": "#d3d3d3", "width": "250px"}),
            html.Span(id='feedback_run_start_end', style={
                "background-color": "#d3d3d3", "width": "250px"}),
            html.Span(dbc.Button('Fit Peaks', id='submit_fitting',
                                 style={'margin-bottom': '25px',
                                        'font-size': '16px'},
                                 color='primary')),

        ], md=12),
        dbc.Col([html.A(dbc.Button('Download Results',
                                   color='success',
                                   style={'font-size': '16px',
                                          'margin': '10px 5px 0 0'}),
                        href='/', id='dl_link_fitting'),
                 ]),

    ], style={'margin': '25px'}),

    html.Hr(),
]

fitting = step1 + step2 + step3


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
