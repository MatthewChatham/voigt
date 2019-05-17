import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from .drawing import countplot, emptyplot
from .extract import read_input
from .aggregate import fwhm, composition, peak_position

import time
import inspect

# tooltip texts
tmp = inspect.getsource(fwhm)
fwhm_tooltiptext = [html.Pre(x) for x in str.split(tmp, '\n') if '#' not in x]
tmp = inspect.getsource(peak_position)
wapp_tooltiptext = [html.Pre(x) for x in str.split(tmp, '\n') if '#' not in x]
tmp = inspect.getsource(composition)
comp_tooltiptext = [html.Pre(x) for x in str.split(tmp, '\n') if '#' not in x]

INSTRUCTIONS = '''
                    Enter a bin width, then select up to 10 partitions.
                    When you're done, click "Generate Output File".
                '''


def _layout():

    session_id = int(round(time.time() * 1000))

    body = dbc.Container(
        [
            html.H1("Step 1: Upload TGA Files", style={'margin-left': '10px'}),
            dbc.Row(
                [
                    dbc.Col(
                        [

                            dcc.Upload(
                                id='upload-data',
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
                                "", id='output-data-upload', style={'min-height': '50px', 'overflow': 'auto', 'max-height': '600px', 'resize': 'vertical'})),
                            dbc.Button('Parse files and refresh chart & options',
                                       id='parse-data-and-refresh-chart', color='primary', style={'margin-top': '5px', 'font-size': '14px'}),
                            ], style={'padding': '10px'}),

                ], style={'margin': '10px'}),

            html.Hr(),

            html.H1('Step 2: Adjust Chart', style={'margin-left': '10px'}),
            dbc.Row([

                dbc.Col([
                    html.Div('Chart type: ', style={
                        'border-bottom': '1px dashed black', 'cursor':'help', 'width': '85px', 'padding-bottom': '1px', 'margin-bottom': '5px'}, id='target'),
                    dbc.Tooltip(html.Div([
                        html.Li(
                            'Sum of Fitted Peaks: The overall '
                            'fitted function for the uploaded TGA '
                            'results. The area under this curve is equal to '
                            'the sum of the areas of each individual fitted peak.'),
                        html.Li(
                            'Fitted Peaks: Shows each fitted peak.'),
                        html.Li(
                            'Peak Histogram: Histogram of peak centers.'),
                    ], style={'width': '500px'}), offset=100, placement='bottom-start', autohide=False, target='target', style={'font-size': '14px', 'max-width': '1000px', 'text-align': 'left'}),
                    dcc.Dropdown(
                        id='type',
                        options=[
                            {'label': 'Sum of Fitted Peaks',
                             'value': 'sumcurve'},
                            {'label': 'Fitted Peaks',
                             'value': 'curve'},
                            {'label': 'Peak Histogram',
                             'value': 'count'},
                            # {'label': 'Area Histogram', 'value': 'area'},
                        ],
                        value='sumcurve',
                        placeholder='Select chart type',
                        style={'width': '250px', 'margin-bottom': '15px'}
                    ),
                    html.Span('Y-axis scale: ', style={'margin-top': '50px'}),
                    dcc.Dropdown(
                        id='scale',
                        options=[
                            {'label': 'Linear Scale',
                             'value': 'linear'},
                            {'label': 'Log Scale', 'value': 'log'},
                        ],
                        value='linear',
                        placeholder='Select scale',
                        style={'width': '250px', 'margin-bottom': '15px'}
                    ),
                ]),

                dbc.Col([
                    html.Span('File: ', style={'magin-top': '0'}),
                    dcc.Dropdown(
                        id='files',
                        placeholder='View models from a single file',
                        style={'width': '250px', 'margin-bottom': '15px'}
                    ),
                    html.Span('Include negative models? ',
                              style={'margin-top': '50px'}),
                    dcc.Dropdown(
                        id='include-negative',
                        options=[
                            {'label': 'Include negative models',
                             'value': 'True'},
                            {'label': 'Exclude negative models',
                             'value': 'False'},
                        ],
                        value='False',
                        style={'width': '250px', 'margin-bottom': '15px'},
                        placeholder='Include negative models'
                    ),
                ]),
                dbc.Col([
                    html.Span('Bin width: '),
                    dcc.Input(
                        id='bin-width',
                        value=10,
                        placeholder='Enter bin width (default: 100)',
                        type='number',
                        min=10, max=100, step=10
                    ),
                ]),


                dbc.Col([
                    dcc.Graph(id='plot', figure=emptyplot())
                ], md=12, style={'margin': '25px 0'})

            ], style={'margin': '25px'}, className="justify-content-center"),

            html.Hr(),

            html.H1('Step 3: Select Splits', style={'margin-left': '10px'}),
            dbc.Row([

                dbc.Col([

                    html.P('Select a partition', id='selection'),
                    dcc.Input(
                        id='split-point',
                        placeholder='Enter next split point',
                        type='number',
                        style={'width': '250px', 'title': 'asdf'},
                        min=0, max=1000, step=10
                    ),
                    html.Button('Add Split', id='add-split',
                                n_clicks_timestamp=0),
                    html.Button('Remove Last Split', id='remove-split',
                                n_clicks_timestamp=0),
                ])
            ], style={'margin': '25px'}),

            html.Hr(),

            html.H1('Step 4: Generate File', style={'margin-left': '10px'}),
            dbc.Row([

                dbc.Col([


                    dbc.Card(dbc.CardBody([
                        html.H3('Aggregates'),
                        html.Hr(),
                        html.P('The output file will compute 3 aggregations'
                               'for each region defined by the selected split points.'),
                        html.P(
                            'Click the text of the definitions below to view Python code.'),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Span('Full Width Half Maximum: Given a region, the'
                                          ' FWHM of the sum of fitted peaks with '
                                          'centers in the region.', id='collapse-button1', style={'cursor': 'pointer'}),
                                dbc.Collapse(
                                    dbc.Card(dbc.CardBody(fwhm_tooltiptext), style={
                                             'margin': '10px'}),
                                    id="collapse1",
                                ),
                            ], action=True),

                            dbc.ListGroupItem([
                                html.Span(
                                    'Composition: Total area in the region.', id='collapse-button2', style={'cursor': 'pointer'}),
                                dbc.Collapse(
                                    dbc.Card(dbc.CardBody(comp_tooltiptext), style={
                                             'margin': '10px'}),
                                    id="collapse2",
                                ),
                            ], action=True),
                            dbc.ListGroupItem([
                                html.Span('Full Width Half Maximum: Average of peak centers'
                                          ' in the region, weighted by their total area (not '
                                          'just area in that region).',
                                          id='collapse-button3', style={'cursor': 'pointer'}),
                                dbc.Collapse(
                                    dbc.Card(dbc.CardBody(fwhm_tooltiptext), style={
                                             'margin': '10px'}),
                                    id="collapse3",
                                ),
                            ], action=True)
                        ]),
                    ]), style={'margin': '10px 0'}),


                    dbc.Card(dbc.CardBody([

                        html.H3('Schema of output file'),
                        html.Hr(),
                        html.P(
                            'The columns of the output file fall into the following categories:'),


                        dbc.ListGroup([


                            dbc.ListGroupItem(
                                'Index: filename (each row corresponds to one TGA file)'),
                            dbc.ListGroupItem(
                                'TGA results: Overall results and positive '
                                'fit statistics. Consists of mass_30_mg, mass_950_mg, '
                                'loss_amorph_pct, pos_chi_square, pos_reduced_chi_square, '
                                'mass_loss_pct, peak_integration_pct'),
                            dbc.ListGroupItem(
                                'Positive model aggregates for each partition '
                                'region. Consists of columns of form '
                                '`[aggregate]_pos_lowerbound_upperbound`.'),
                            dbc.ListGroupItem(
                                'Negative model aggregates, colums of form `[aggregate]_neg`.'),


                        ])




                    ]), style={'margin': '10px 0'}),

                ]),



            ], style={'margin': '25px'}),

            dbc.Row([

                dbc.Col([

                    html.Span(id='feedback', style={
                        "background-color": "#d3d3d3", "width": "250px"}),
                    html.Span(dbc.Button('Generate Output File', id='submit',
                                         style={'margin-bottom': '25px', 'font-size': '16px'}, color='primary')),

                ], md=12),
                dbc.Col([html.A(dbc.Button('Download CSV', color='success', style={'font-size': '16px', 'margin': '10px 5px 0 0'}),
                                href='/dash/download', id='dl_link'),
                         html.A(dbc.Button('Download Images', color='success', style={'font-size': '16px', 'margin': '10px 5px 0 0'}),
                                href='/dash/download', id='dl_link_images'),
                         ]),

            ], style={'margin': '25px'}),

            dcc.Interval(id='interval', interval=1 * 1000,  # milliseconds
                         n_intervals=0),
            html.Div(id='state', style={'display': 'none'}),
            html.Div(id='areas-state', style={'display': 'none'}),
            html.Div(id='bin-width-state', style={'display': 'none'}),
            html.Div(id='submit-state', style={'display': 'none'}),
            html.Div(children=session_id,
                     id='session-id',
                     style={'display': 'none'}
                     ),

        ],  # END CONTAINER

        className="mt-4",
        style={'background-color': '#e8eaf6',
               'padding': '10px', 'font-size': '16px'}
    )


    footer = html.Div(
        [
            html.Span(['Copyright © 2019 ', 
            html.A('Matthew Chatham', href='http://www.matthewchatham.com')], style={'top':'50%'})
        ],
        style={'width':'100%', 'height':'50px', 'text-align':'center', 'font-size':'14px', 'padding-top':'15px'}
    )


  #     <!-- Copyright -->
  # <div class="footer-copyright text-center py-3">© 2018 Copyright:
  #   <a href="https://mdbootstrap.com/education/bootstrap/"> MDBootstrap.com</a>
  # </div>
  # <!-- Copyright -->

    return html.Div([body, footer])


layout = _layout
