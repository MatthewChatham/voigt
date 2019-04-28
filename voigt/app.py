import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from extract import get_data
from drawing import countplot, areaplot
from aggregate import aggregate_all_files
from flask import send_file
from os.path import join
import os
import json


if os.environ.get('STACK'):
    env = 'Heroku'
    BASE_DIR = '/app'
else:
    env = 'Dev'
    BASE_DIR = '/Users/matthew/freelance/voigt'

print(f'Running in {os.getcwd()} on {env} environment.')


DATA = get_data()
INSTRUCTIONS = '''
                    Enter a bin width, then select up to 10 partitions.
                    When you're done, click "Generate Output File".
                '''

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


def construct_shapes(scale='linear', split_point=None, max_=10):
    shapes = []

    if split_point:
        shapes.append({
            'type': 'line',
            'x0': split_point,
            'y0': 0,
            'x1': split_point,
            'y1': max_,
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
    html.Button('Add Split', id='add-split', n_clicks_timestamp=0),
    html.Button('Remove Last Split', id='remove-split', n_clicks_timestamp=0),
    html.Button('Generate Output File', id='submit'),
    html.A('Download CSV', href='/dash/download', id='dl_link'),
    html.P('Select a partition', id='selection'),
    dcc.Graph(id='plot', figure=countplot(DATA=DATA)),

    html.Div(id='splits')  # , style={'display': 'none'})
])


@app.callback(Output('selection', 'children'), [Input('split-point', 'value')])
def update_selection_prompt(split):
    if split is None:
        return 'Select a split point.'
    else:
        return f'You have selected {split}.'


@app.callback(
    Output('splits', 'children'),
    [
        Input('add-split', 'n_clicks_timestamp'),
        Input('remove-split', 'n_clicks_timestamp'),
        Input('split-point', 'value')
    ],
    [State('splits', 'children')]
)
def update_state(add, remove, split_point, state):
    if add == 0 and remove == 0:
        return '{"splits":[], "add":0, "remove":0}'

    state = json.loads(state)

    if add > remove and add > state['add']:
        if split_point not in state['splits']:
            state['splits'].append(split_point)
    elif remove > add and remove > state['remove']:
        if len(state['splits']) > 0:
            state['splits'].pop()

    state['add'] = add
    state['remove'] = remove

    return json.dumps(state)


@app.callback(
    Output('plot', 'figure'),
    [
        Input('bin-width', 'value'),
        Input('scale', 'value'),
        Input('type', 'value'),
        Input('split-point', 'value')
    ],
    [State('splits', 'children')]
)
def update_plot(bin_width, scale, chart_type, split_point, splits):
    funcs = {'count': countplot, 'area': areaplot}
    return funcs[chart_type](
        bin_width,
        DATA=DATA,
        scale=scale,
        shapes=construct_shapes(split_point=split_point)
    )


@app.callback(
    Output('dl_link', 'style'),
    [Input('submit', 'n_clicks')],
    [State('splits', 'children')]
)
def submit(n_clicks, splits):
    if n_clicks is None:
        return {'display': 'none'}
    if n_clicks is not None:
        aggregate_all_files(splits, get_data())
        return {}


@app.server.route('/dash/download')
def download_csv():
    return send_file(join(BASE_DIR, 'output', 'output.csv'),
                     mimetype='text/csv',
                     attachment_filename='output.csv',
                     as_attachment=True)


if __name__ == '__main__':
    app.run_server(debug=True)
