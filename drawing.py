import plotly.graph_objs as go


MIN, MAX = 0, 1000


def figure(bin_width=50, shapes=[], scale='linear', selectedData=None, _type=None, DATA=None):
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
