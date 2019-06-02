"""
Plotly plotting functions.
"""
import plotly.graph_objs as go
from .aggregate import compute_bin_areas, Voigt
import numpy as np


MIN, MAX = 0, 1000

# todo: only recompute area if bin width changes
# todo: find and fix bugs with partition selection on area plot


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


def countplot(bin_width=50, shapes=[],
              scale='linear', selectedData=None, DATA=None, exclude_negative=True):

    if exclude_negative:
        DATA = DATA.loc[~DATA.variable.str.contains('nm')]

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
            # 'dragmode': 'select',
            'yaxis': dict(
                type=scale,
                autorange=True
                # range=range_
            )
        })
    }

    return figure


def areaplot(bin_width=50, shapes=[],
             scale='linear', selectedData=None, DATA=None, exclude_negative=True, areas=None):

    if exclude_negative:
        DATA = DATA.loc[~DATA.variable.str.contains('nm')]

    nbins = (MAX - MIN) / int(bin_width)
    bins = np.linspace(MIN, MAX, nbins + 1)
    bins = [(x, bins[i + 1])
            for i, x in enumerate(bins) if i < len(bins) - 1]

    if areas is None:
        print('COMPUTING BIN AREAS')
        areas = compute_bin_areas(bins, DATA)

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
            # 'dragmode': 'select',
            'yaxis': dict(
                type=scale,
                autorange=True
                # range=range_

            )
        })
    }

    return figure


def curveplot(bin_width=50, shapes=[],
              scale='linear', selectedData=None, DATA=None, exclude_negative=True):

    if exclude_negative:
        DATA = DATA.loc[~DATA.variable.str.contains('nm')]

    models = DATA

    X = np.linspace(30, 1000, 1000 - 30 + 1)

    data = list()

    for idx, m in models.iterrows():

        prefix = str.split(m.variable, '_')[0]

        sigma = m.loc[prefix + '_sigma']
        gamma = sigma

        amplitude = m.loc[prefix + '_amplitude']

        trace = go.Scatter(
            x=X,
            y=Voigt(X, center=m.value, sigma=sigma,
                    gamma=gamma, amplitude=amplitude),
            mode='lines',
            name=m.filename.strip('.txt') + f'/{prefix}'
        )

        data.append(trace)

    figure = {
        'data': data,
        'layout': go.Layout({
            'shapes': shapes,
            # 'dragmode': 'select',
            'yaxis': dict(
                type=scale,
                autorange=True
                # range=range_

            )
        })
    }

    return figure


def sumcurveplot(bin_width=50, shapes=[],
                 scale='linear', selectedData=None, DATA=None, exclude_negative=True):

    if exclude_negative:
        DATA = DATA.loc[~DATA.variable.str.contains('nm')]

    models = DATA

    def F(x):
        vals = np.array([0]*len(x), ndmin=2)

        for idx, model in models.iterrows():
            prefix = model.variable[:model.variable.index('_')]
            sigma = model.loc[prefix + '_sigma']
            gamma = sigma
            amplitude = model.loc[prefix + '_amplitude']
            if 'nm' in prefix:
                amplitude = -amplitude

            res =  np.array(Voigt(x, center=model.value, amplitude=amplitude, sigma=sigma, gamma=gamma), ndmin=2)
            # print(vals, res)
            vals = np.concatenate([vals, res], axis=0)
        return vals.sum(axis=0)

    X = np.linspace(30, 1000, 1000 - 30 + 1)

    data = list()

    trace = go.Scatter(
        x=X,
        y=F(X),
        mode='lines',
    )

    data.append(trace)

    figure = {
        'data': data,
        'layout': go.Layout({
            'shapes': shapes,
            # 'dragmode': 'select',
            'yaxis': dict(
                type=scale,
                autorange=True
                # range=range_

            )
        })
    }

    return figure


def emptyplot():
    data = list()

    figure = {
        'data': data,
        'layout': go.Layout({
            # 'dragmode': 'select',
            'yaxis': dict(
                type='linear',
                autorange=True
                # range=range_

            )
        })
    }

    return figure
