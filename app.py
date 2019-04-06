import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from scipy.special import wofz
from scipy.integrate import quad

from extract import get_data

DATA = get_data()
INSTRUCTIONS = '''
                    Enter a bin width, then select up to 10 partitions.
                    When you're done, click "Generate Output File".
                '''
MIN, MAX = 0, 1000
partitions = []
vals = [0, 0, 0]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


def Voigt(x, sigma, gamma):
    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / sigma\
        / np.sqrt(2 * np.pi)


def figure(bin_width=50, shapes=[], scale='linear', selectedData=None, _type=None):
    """
    Given a bin_width and a shapes object, construct a Plotly figure.
    """

    figure = {
        'data': [go.Histogram(x=DATA.value,
                              xbins=dict(
                                  start=MIN,
                                  end=MAX,
                                  size=bin_width
                              ),
                              marker=dict(
                                  color='#FFD7E9',
                              ),
                              opacity=0.75
                              )
                 ],
        'layout': go.Layout({
            'shapes': shapes,
            'dragmode': 'select',
            'yaxis': dict(
                type=scale,
                autorange=True
            )
        })
    }

    if _type == 'count':

        # Display a rectangle to highlight the previously selected region
        shape = {
            'type': 'rect',
            'line': {
                'width': 1,
                'dash': 'dot',
                'color': 'darkgrey'
            }
        }
        if selectedData:
            figure['layout']['shapes'] = figure['layout']['shapes'] + (dict({
                'x0': selectedData['range']['x'][0],
                'x1': selectedData['range']['x'][1],
                'y0': selectedData['range']['y'][0],
                'y1': selectedData['range']['y'][1]
            }, **shape),)
        else:
            pass

    elif _type == 'area':
        nbins = (MAX - MIN) / int(bin_width)
        bins = np.linspace(MIN, MAX, nbins + 1)
        bins = [(x, bins[i + 1])
                for i, x in enumerate(bins) if i < len(bins) - 1]
        areas = [0] * len(bins)
        for i, b in enumerate(bins):
            for idx, model in DATA.iterrows():
                model_prefix = str.split(model.variable, '_')[0]
                sigma = model[model_prefix + '_sigma']
                gamma = model[model_prefix + '_gamma']
                a, e = quad(lambda x: Voigt(x, sigma, gamma), b[0], b[1])
                areas[i] += 0 if np.isnan(a) else a

        figure = {
            'data': [go.Bar(x=[x[0] for x in bins], y=areas, width=[bin_width] * len(bins),
                            marker=dict(
                                color='#FFD7E9',
            ),
                opacity=0.75
            )
            ],
            'layout': go.Layout({
                'shapes': shapes,
                'dragmode': 'select',
                'yaxis': dict(
                    type=scale,
                    autorange=True
                )
            })
        }

    return figure


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
    dcc.Graph(id='plot', figure=figure()),
    html.P(id='partitions'),
    html.Button('Add Partition', id='add-partition'),
    html.Button('Remove Last Partition', id='remove-partition'),
    html.Button('Generate Output File', id='submit'),

    html.Div(id='hidden1')  # display: none
])


def generate_output_file():
    pass


def in_partitions(l):
    """
    Given a partition candidate, check whether
    it overlaps with any existing partitions.
    """
    print(l)
    _l = [int(x) for x in l]
    res = False
    for p in partitions:
        p = [int(x) for x in p]
        if (_l[1] > p[1] and _l[0] <= p[1]) \
                or (_l[0] <= p[0] and _l[1] > p[0]) \
                or (_l[1] <= p[1] and _l[0] >= p[0]):
            res = True
    return res


def construct_shapes():
    """
    Construct a Plotly shape object for each partition.
    """
    shapes = []
    for p in partitions:
        shapes.append({
            'type': 'rect',
            'x0': p[0], 'x1': p[1],
            'y0': 0, 'y1': 20,
            'line': {'color': 'rgba(128, 0, 128, 1)'},
            'fillcolor': 'rgba(128, 0, 128, 0.2)',
        })
    return shapes


@app.callback(
    Output('hidden1', 'children'),
    [Input('submit', 'n_clicks')]
)
def submit(n_clicks):
    generate_output_file()


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

    return figure(scale=scale, bin_width=bin_width, selectedData=selectedData, shapes=construct_shapes(), _type=_type)


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
