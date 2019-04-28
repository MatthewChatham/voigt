import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from extract import get_data
from drawing import figure
from aggregate import aggregate_all_files

from flask import send_file

from os.path import join

BASE_DIR = '/Users/matthew/freelance/voigt/'
BASE_DIR = '.'  # FOR DEV ON HEROKU

DATA = get_data()
INSTRUCTIONS = '''
                    Enter a bin width, then select up to 10 partitions.
                    When you're done, click "Generate Output File".
                '''

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


def construct_shapes(scale='linear', split_point=None):
    """
    Construct a Plotly shape object for each partition.
    """
    shapes = []
    # if scale == 'log':
    #     return shapes
    for p in partitions:
        shapes.append({
            'type': 'rect',
            'x0': p[0], 'x1': p[1],
            'y0': 1, 'y1': 20,
            'line': {'color': 'rgba(128, 0, 128, 1)'},
            'fillcolor': 'rgba(128, 0, 128, 0.2)',
        })

    if split_point:
        shapes.append({
            'type': 'line',
            'x0': split_point,
            'y0': 0,
            'x1': split_point,
            'y1': 100,
            'line': {
                'color': 'rgb(55, 128, 191)',
                'width': 3,
            },
        })

    return shapes


app.layout = html.Div([
    html.Div([
        html.H6('Instructions'),
        html.P(INSTRUCTIONS)
    ]),
    dcc.Input(
        id='bin-width',
        value=100,
        placeholder='Enter bin width (default: 100)',
        type='number',
        inputmode='numeric',
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
        ],
        value='count',
        placeholder='Select histogram type',
        style={'width': '250px'}
    ),
    dcc.Input(
        id='split-point',
        placeholder='Enter next split point',
        type='number',
        inputmode='numeric',
        style={'width': '250px', 'title': 'asdf'},
        min=0, max=1000, step=10
    ),
    html.Button('Add Split', id='add-split'),
    html.Button('Remove Last Split', id='remove-split'),
    html.Button('Generate Output File', id='submit'),
    html.A('Download CSV', href='/dash/download', id='dl_link'),
    html.P('Select a partition', id='selection'),
    dcc.Graph(id='plot', figure=figure(DATA=DATA)),

    html.Div(id='split-points', style={'display': 'none'})
])


@app.callback(
    Output('split-points', 'children'),
    [
        Input('add-split', 'n_clicks_timestamp'),
        Input('remove-split', 'n_clicks_timestamp'),
        Input('split-point', 'value')
    ],
    [State('split-points', 'children')]
)
def update_state(add, remove, split_point, partitions):
    partitions = from_json(partitions)
    if add > remove:
        partitions.append(split_point)
    if remove > add:
        partitions.pop()
    else:
        raise RuntimeError('Add and remove buttons clicked at the same time!')
    return to_json(partitions)


@app.callback(
    Output('dl_link', 'style'),
    [Input('submit', 'n_clicks')]
)
def submit(n_clicks):
    if n_clicks is None:
        return {'display': 'none'}
    if n_clicks is not None:
        aggregate_all_files(partitions, get_data())
        return {}


@app.server.route('/dash/download')
def download_csv():
    return send_file(join(BASE_DIR, 'output.csv'),
                     mimetype='text/csv',
                     attachment_filename='output.csv',
                     as_attachment=True)


@app.callback(
    Output('plot', 'figure'),
    [
        Input('bin-width', 'value'),
        Input('scale', 'value'),
        Input('type', 'value'),
        Input('split-point', 'value')
    ]
)
def update_plot(bin_width, scale, chart_type, split_point):
    return figure(bin_width, DATA=DATA, scale=scale, _type=chart_type, shapes=construct_shapes(split_point=split_point))


@app.callback(
    Output('selection', 'children'),
    [Input('plot', 'selectedData')]
)
def update_selection_prompt(selectedData):
    if selectedData is None:
        return 'Select a partition.'
    else:
        # print(selectedData["range"]["x"])
        txt = ", ".join([str(int(x)) for x in selectedData["range"]["x"]])
        return f'You have selected [{txt}].'


if __name__ == '__main__':
    app.run_server(debug=True)
