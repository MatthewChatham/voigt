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
                    html.P('Note: Refreshing page will \
                                    remove input files. \
                                    Uploading multiple times will first \
                                    remove all existing files.'),
                    # file format
                    html.Div(['File Format: ', dcc.Dropdown(options=[
                        {'label': 'Q500/DMSM', 'value': 'Q500/DMSM'},
                        {'label': 'TGA 5500', 'value': 'TGA 5500'},
                        {'label': 'Just Temp and Mass',
                            'value': 'Just Temp and Mass'}
                    ], value='Q500/DMSM', style={'width': '200px'}, id='file-format')]),
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
                dbc.Button('Parse files and refresh chart & options',
                           id='tga-parse-data-and-refresh-chart',
                           color='primary',
                           style={'margin-top': '5px', 'font-size': '14px'})
            ], style={'padding': '10px'}),

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
                    {'label': 'Negative peaks', 'value': 'neg'},
                ],
                values=[], inputStyle={'margin-right': '5px'},
                id='negative-peaks'
            ),

            # Temperature range to bound negative curve fitting
            html.Div([
                html.P(['Temperature range to bound', html.Strong(
                    ' negative '), 'curve fitting'], id='neg-range-values'),
                dcc.RangeSlider(

                    min=30,
                    max=1000,
                    step=1,
                    value=[30, 1000],
                    dots=False,
                    marks={
                        30: {'label': '30°C', 'style': {'color': '#77b0b1'}},
                        1000: {'label': '1000°C', 'style': {'color': '#f50'}}
                    },
                    id='neg-peak-slider',
                    updatemode='drag',
                ),
            ], style={'padding': '10px', 'width': '250px', 'margin-bottom': '5px'}, id='neg-peak-range'),

            # Temperature range to bound positive curve fitting
            html.Div([
                html.P(['Temperature range to bound', html.Strong(
                    ' positive '), 'curve fitting'], id='pos-range-values'),
                dcc.Checklist(
                    options=[
                        {'label': 'Full', 'value': 'full'},
                    ],
                    values=[], inputStyle={'margin-right': '5px'},
                    id='temp-range-pos-full'
                ),
                dcc.RangeSlider(
                    count=1,
                    min=30,
                    max=1000,
                    step=1,
                    value=[450, 850],
                    marks={
                        30: {'label': '30°C', 'style': {'color': '#77b0b1'}},
                        1000: {'label': '1000°C', 'style': {'color': '#f50'}}
                    },
                    updatemode='drag',
                    id='pos-peak-slider',
                    disabled=True
                ),
            ], style={'padding': '10px', 'width': '250px', 'margin-bottom': '5px'}),

        ]),


        dbc.Col([
            # max peak num
            html.Div(['Max Peak Num: ', dcc.Input(
                min=1, max=20, step=1, value=10, type='number', id='max-peak-num')]),

            # mass defect warning
            html.Div(['Mass Defect Warning: ', dcc.Input(
                min=0, max=100, step=1, value=10, type='number', id='mass-defect-warning'), ]),

            # Temperature to calculate mass loss from
            # html.Div(['Temp to calculate mass loss from: ', dcc.Input(
            #     min=30, max=1000, step=1, value=60, type='number', id='mass-loss-from-temp', disabled=False), ]),

        ]),

        dbc.Col([
            # Temperature to calculate mass loss to
            html.Div(['Temp to calculate mass loss to: ', dcc.Input(
                min=30, max=1000, step=1, value=950, type='number', id='mass-loss-to-temp', disabled=False), ]),

            # run start temp
            html.Div(['Run Start Temp / Temp to calculate mass loss from: ', dcc.Input(
                min=30, max=1000, step=1, value=60, type='number', id='run-start-temp', disabled=False)]),
            # amorphous carbon temperature
            html.Div(['Amorphous Carbon Temperature: ', dcc.Input(
                min=30, max=1000, step=1, value=450, type='number', id='amorphous-carbon-temp', disabled=False)]),


        ])

    ], style={'margin': '25px'}),

    html.Hr(),
]

step3 = [
    html.H1('Step 3: Run Peak-Fitting Routine', style={'margin-left': '10px'}),

    dbc.Row([

        dbc.Col([
            html.Div(['Job timeout (minutes): ', dcc.Input(
                min=1, max=20, step=1, value=15, type='number', id='job-timeout-mins')]),

            html.Span(id='feedback_fitting', style={
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
