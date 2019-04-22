import plotly.graph_objs as go
from aggregate import compute_bin_areas
import numpy as np


MIN, MAX = 0, 1000

# todo: only recompute area if bin width changes
# todo: find and fix bugs with partition selection on area plot


def figure(bin_width=50, shapes=[],
           scale='linear', selectedData=None, _type=None, DATA=None):
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
                # range=range_
            )
        })
    }

    if _type == 'area':
        nbins = (MAX - MIN) / int(bin_width)
        bins = np.linspace(MIN, MAX, nbins + 1)
        bins = [(x, bins[i + 1])
                for i, x in enumerate(bins) if i < len(bins) - 1]
        areas = compute_bin_areas(bins, DATA)

        # max_ = max(areas)
        # range_ = [0,max([500, max_])] if scale == 'linear' else [0,max([500, max_])]

        figure = {
            'data': [go.Bar(x=[x[0] for x in bins],
                            y=areas, width=[bin_width] * len(bins),
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
                    # range=range_

                )
            })
        }

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
            'y0': max([selectedData['range']['y'][0], 1]),
            'y1': min([selectedData['range']['y'][1], 20])
        }, **shape),)
    else:
        pass

    return figure
