import dash_core_components as dcc
import dash_html_components as html

from .drawing import countplot, emptyplot
from .extract import read_input
from .aggregate import fwhm, composition, peak_position

import time
import inspect

# tooltip texts
tmp = inspect.getsource(fwhm)
fwhm_tooltiptext = [html.P(x) for x in str.split(tmp, '\n') if '#' not in x]
tmp = inspect.getsource(peak_position)
wapp_tooltiptext = [html.P(x) for x in str.split(tmp, '\n') if '#' not in x]
tmp = inspect.getsource(composition)
comp_tooltiptext = [html.P(x) for x in str.split(tmp, '\n') if '#' not in x]

INSTRUCTIONS = '''
                    Enter a bin width, then select up to 10 partitions.
                    When you're done, click "Generate Output File".
                '''


def _layout():

    session_id = int(round(time.time() * 1000))

    return html.Div([

        # SESSION ID
        html.Div(children=session_id,
                 id='session-id',
                 style={'display': 'none'}
                 ),


        # Step 1: Upload TGA files
        html.Div([
            html.H1('Step 1: Upload TGA Files'),
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
                    'margin': '10px',
                },
                multiple=True
            ),
            html.Div(id='output-data-upload', style={'padding': '10px'}),
            html.Button('Parse files and refresh chart & options',
                        id='parse-data-and-refresh-chart'),
        ], style={'width': '50%'}),


        html.H1('Step 2: Adjust Chart & Select Splits',
                style={'margin-top': '25px'}),


        html.Li('Sum of Fitted Peaks: The overall fitted function for the uploaded TGA results. The area under this curve is equal to the sum of the areas of each individual fitted peak.'),
        html.Li('Fitted Peaks: Shows each fitted peak.'),
        html.Li('Peak Histogram: Histogram of peak centers.'),


        dcc.Loading([
            html.Table([

                html.Tr([
                        # html.P('Chart type'),
                        html.Td(dcc.Dropdown(
                            id='type',
                            options=[
                                {'label': 'Sum of Fitted Peaks',
                                    'value': 'sumcurve'},
                                {'label': 'Fitted Peaks', 'value': 'curve'},
                                {'label': 'Peak Histogram', 'value': 'count'},
                                # {'label': 'Area Histogram', 'value': 'area'},
                            ],
                            value='sumcurve',
                            placeholder='Select chart type',
                            style={'width': '250px'}
                        )),
                        html.Td(dcc.Dropdown(
                            id='scale',
                            options=[
                                {'label': 'Linear Scale', 'value': 'linear'},
                                {'label': 'Log Scale', 'value': 'log'},
                            ],
                            value='linear',
                            placeholder='Select scale',
                            style={'width': '250px'}
                        )),
                        html.Td(dcc.Dropdown(
                            id='files', placeholder='View models from a single file', style={'width': '250px'})),
                        html.Td(dcc.Dropdown(
                            id='include-negative',
                            options=[
                                {'label': 'Include negative models', 'value': True},
                                {'label': 'Exclude negative models', 'value': False},
                            ],
                            value=False,
                            style={'width': '250px'},
                            placeholder='Include negative models'
                        ))
                        ]),  # end row 1

                html.Tr([

                        html.Td(['Bin width: ', dcc.Input(
                            id='bin-width',
                            value=10,
                            placeholder='Enter bin width (default: 100)',
                            type='number',
                            min=10, max=100, step=10
                        )])
                        ])  # end row 2

            ]),  # end table

            dcc.Graph(id='plot', figure=emptyplot()),

        ]),  # end loading



        # html.H1('Step 3: Select Splits'),
        html.P('Select a partition', id='selection'),
        dcc.Input(
            id='split-point',
            placeholder='Enter next split point',
            type='number',
            style={'width': '250px', 'title': 'asdf'},
            min=0, max=1000, step=10
        ),
        html.Button('Add Split', id='add-split', n_clicks_timestamp=0),
        html.Button('Remove Last Split', id='remove-split',
                    n_clicks_timestamp=0),


        html.H1('Step 3: Generate File', style={'margin-top': '10px'}),
        html.P('The output file will compute 3 aggregations for each region defined by the selected split points.'),
        html.Li([html.Span(className='tooltip', children=['Full Width Half Maximum (FWHM)', html.Span(
            className='tooltiptext', children=fwhm_tooltiptext)]),
            ': Given a region, the FWHM of the sum of fitted peaks with centers in the region.']),
        html.Li([html.Span(className='tooltip', children=['Weighted-Average Peak Position (WAPP)', html.Span(
            className='tooltiptext', children=wapp_tooltiptext)]),
            ': Average of peak centers in the region, weighted by their total area (not just area in that region).']),
        html.Li([html.Span(className='tooltip', children=['Composition', html.Span(
            className='tooltiptext', children=comp_tooltiptext)]),
            ': Total area in the region.']),

        html.Strong('Schema of output file:'),
        html.P('The columns of the output file fall into the following categories:'),
        html.Li('Index: filename (each row corresponds to one TGA file)'),
        html.Li('TGA results: Overall results and positive fit statistics. Consists of mass_30_mg, mass_950_mg, loss_amorph_pct, pos_chi_square, pos_reduced_chi_square, mass_loss_pct, peak_integration_pct'),
        html.Li('Positive model aggregates for each partition region. Consists of columns of form `[aggregate]_pos_lowerbound_upperbound`.'),
        html.Li('Negative model aggregates, colums of form `[aggregate]_neg`.'),

        dcc.Interval(
            id='interval',
            interval=10 * 1000,  # milliseconds
            n_intervals=0,
        ),
        html.P(id='feedback'),
        html.Button('Generate Output File', id='submit',
                    style={'margin-bottom': '25px'}),
        html.A(html.Button('Download CSV'),
               href='/dash/download', id='dl_link'),
        html.A(html.Button('Download Images'),
               href='/dash/download', id='dl_link_images'),
        html.Div(id='state', style={'display': 'none'}),
        html.Div(id='areas-state', style={'display': 'none'}),
        html.Div(id='bin-width-state', style={'display': 'none'})
    ])


layout = _layout
