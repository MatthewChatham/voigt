import dash_core_components as dcc
import dash_html_components as html

from .drawing import countplot
from .extract import read_input

INSTRUCTIONS = '''
                    Enter a bin width, then select up to 10 partitions.
                    When you're done, click "Generate Output File".
                '''

layout = html.Div([
    html.Div([
        html.H6('Instructions'),
        html.P(INSTRUCTIONS)
    ]),
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
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    dcc.Input(
        id='bin-width',
        value=100,
        placeholder='Enter bin width (default: 100)',
        type='number',
        style={'width': '250px', 'title': 'asdf'},
        min=10, max=100, step=10
    ),
    dcc.Dropdown(
        id='scale',
        options=[
            {'label': 'Linear', 'value': 'linear'},
            {'label': 'Log', 'value': 'log'},
        ],
        value='linear',
        placeholder='Select scale',
        style={'width': '250px'}
    ),
    dcc.Dropdown(
        id='type',
        options=[
            {'label': 'Count', 'value': 'count'},
            {'label': 'Area', 'value': 'area'},
            {'label': 'Curve', 'value': 'curve'}
        ],
        value='count',
        placeholder='Select histogram type',
        style={'width': '250px'}
    ),
    dcc.Input(
        id='split-point',
        placeholder='Enter next split point',
        type='number',
        style={'width': '250px', 'title': 'asdf'},
        min=0, max=1000, step=10
    ),
    html.Button('Add Split', id='add-split', n_clicks_timestamp=0),
    html.Button('Remove Last Split', id='remove-split', n_clicks_timestamp=0),
    html.Button('Generate Output File', id='submit'),
    html.A('Download CSV', href='/dash/download', id='dl_link'),
    html.P('Select a partition', id='selection'),
    dcc.Graph(id='plot', figure=countplot(DATA=read_input())),

    html.Div(id='state'),  # , style={'display': 'none'})
    html.Div(id='areas-state'),
    html.Div(id='bin-width-state')
])
