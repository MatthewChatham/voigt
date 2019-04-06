import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


from extract import get_data
from drawing import figure, in_partitions, construct_shapes
from aggregate import aggregate_all_files

DATA = get_data()
INSTRUCTIONS = '''
                    Enter a bin width, then select up to 10 partitions.
                    When you're done, click "Generate Output File".
                '''
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


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
    html.P(),
    html.P('Select a partition', id='selection'),
    dcc.Graph(id='plot', figure=figure(DATA=DATA)),
    html.P(id='partitions'),
    html.Button('Add Partition', id='add-partition'),
    html.Button('Remove Last Partition', id='remove-partition'),
    html.Button('Generate Output File', id='submit'),

    html.Div(id='hidden1')  # display: none
])


@app.callback(
    Output('hidden1', 'children'),
    [Input('submit', 'n_clicks')]
)
def submit(n_clicks):
    aggregate_all_files()


@app.callback(
    Output('plot', 'figure'),
    [
        Input('add-partition', 'n_clicks'),
        Input('plot', 'selectedData'),
        Input('remove-partition', 'n_clicks'),
        Input('bin-width', 'value'),
        Input('scale', 'value'),
        Input('type', 'value')
    ]
)
def update_plot(add_clicks, selectedData, remove_clicks, bin_width, scale, _type):

    # CLICKED ADD BUTTON
    if add_clicks and add_clicks > vals[0] and selectedData and \
            not in_partitions(selectedData['range']['x']):
        vals[0] = add_clicks
        partitions.append([int(x) for x in selectedData['range']['x']])

    # CLICKED REMOVE BUTTON
    elif remove_clicks and remove_clicks > vals[1] \
            and len(partitions) > 0:
        vals[1] = remove_clicks
        partitions.pop()

    return figure(scale=scale, bin_width=bin_width, selectedData=selectedData, shapes=construct_shapes(), _type=_type, DATA=DATA)


@app.callback(
    Output('selection', 'children'),
    [Input('plot', 'selectedData')]
)
def update_selection(selectedData):
    if selectedData is None:
        return 'Select a partition.'
    else:
        print(selectedData["range"]["x"])
        txt = ", ".join([str(int(x)) for x in selectedData["range"]["x"]])
        return f'You have selected [{txt}].'


if __name__ == '__main__':
    app.run_server(debug=True)
