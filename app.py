import dash
import base64
import io

from dash.dependencies import Input, Output, State
import dash_table
import plotly
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import math
import operator

from datetime import datetime
from sklearn.metrics import classification_report
from utils import text_to_labels, make_ranges, minmax_scale, normalize, KNN

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

CONST_NOW = str(datetime.now())

out_columns = ['variety', 'Hrabstwo', 'class']
metrics = ['euclidean', 'manhattan', 'czebyszew', 'mahalanobis']

def string_to_datetime(date_time_str):
    return datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        return pd.read_excel(io.BytesIO(decoded))

def csv_handle(decoded):
    return pd.read_csv(
        io.StringIO(decoded.decode('utf-8')))

def xls_handle(decoded):
    return pd.read_excel(io.BytesIO(decoded))


app.layout = html.Div([
    dcc.Upload(
        id='df-upload',
        children=html.Div([
            html.A('Wybierz plik')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
    ),
    html.Div([
        # adding new column
        dcc.Input(
            id='new-column',
            placeholder='Podaj nazwę',
            value='',
            style={'padding': 10, 'width': '150px', 'margin': 10, 'borderColor': '#1890ff'}
        ),
        html.Button('Nowa kolumna', id='add-column', n_clicks=0,
                    style={'margin': 10, 'width': '150px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'}),
        html.Span('\U000000b7'),
        # adding new row
        html.Button('Nowy wiersz', id='add-row', n_clicks=0,
                    style={'margin': 10, 'width': '150px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'}),
    ], style={'height': 70, 'display': 'none'}),

    html.Div([
        # label existing column
        dcc.Input(
            id='label-column',
            placeholder='Nazwa kolumny',
            value='',
            style={'padding': 10, 'width': '150px', 'margin': 10, 'borderColor': '#1890ff'}
        ),
        html.Button('Zamiana danych', id='lab-btn', n_clicks=0,
                    style={'margin': 10, 'width': '180px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'}),
        html.Span('\U000000b7'),
        # make ranges for column
        dcc.Input(
            id='range-column',
            placeholder='Nazwa kolumny',
            value='',
            style={'padding': 10, 'width': '150px', 'margin': 10, 'borderColor': '#1890ff'}
        ),
        dcc.Input(
            id='range-amount',
            placeholder='Liczba przedziałów',
            value='',
            style={'padding': 10, 'width': '200px', 'margin': 10, 'borderColor': '#1890ff'}
        ),
        html.Button('Dyskretyzacja', id='rng-btn', n_clicks=0,
                    style={'margin': 10, 'width': '170px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'}),
    ]),

    html.Div([
        # minmax scale column
        dcc.Input(
            id='minmax-column',
            placeholder='Nazwa kolumny',
            value='',
            style={'padding': 10, 'width': '150px', 'margin': 10, 'borderColor': '#1890ff'}
        ),
        dcc.Input(
            id='a-val',
            placeholder='Min',
            value='',
            style={'padding': 10, 'width': '60px', 'margin': 10, 'borderColor': '#1890ff'}
        ),
        dcc.Input(
            id='b-val',
            placeholder='Max',
            value='',
            style={'padding': 10, 'width': '60px', 'margin': 10, 'borderColor': '#1890ff'}
        ),
        html.Button('Zmiana przedziału min-max', id='minmax-btn', n_clicks=0,
                    style={'margin': 10, 'width': '280px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'}),
        html.Span('\U000000b7'),
        dcc.Input(
            id='normalize-column',
            placeholder='Nazwa kolumny',
            value='',
            style={'padding': 10, 'width': '150px', 'margin': 10, 'borderColor': '#1890ff'}
        ),
        html.Button('Normalizacja', id='normalize-btn', n_clicks=0,
                    style={'margin': 10, 'width': '220px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'})
    ]),

    html.Div([
        html.Div([
            dcc.Input(
                id='col-decision',
                placeholder='Kolumna decyzyjna',
                value='',
                style={'padding': 10, 'width': '200px', 'margin': 10, 'borderColor': '#1890ff'}
            ),
            html.Span('\U000000b7')
        ]),
# generate 2d graph
html.Div([
    dcc.Input(
        id='col-a-2d',
        placeholder='Kolumna a',
        value='',
        style={'padding': 10, 'width': '100px', 'margin': 10, 'borderColor': '#1890ff'}
    ),
    dcc.Input(
        id='col-b-2d',
        placeholder='Kolumna b',
        value='',
        style={'padding': 10, 'width': '100px', 'margin': 10, 'borderColor': '#1890ff'}
    ),
    html.Button('Generuj wykres 2d', id='generate-2d-btn', n_clicks=0,
                style={'margin': 10, 'width': '250px', 'background': '#1890ff', 'borderColor': '#1890ff',
                       'color': '#fff'}),
    html.Span('\U000000b7')
]),
# generate 3d graph
html.Div([
    dcc.Input(
        id='col-a-3d',
        placeholder='Kolumna a',
        value='',
        style={'padding': 10, 'width': '100px', 'margin': 10, 'borderColor': '#1890ff'}
    ),
    dcc.Input(
        id='col-b-3d',
        placeholder='Kolumna b',
        value='',
        style={'padding': 10, 'width': '100px', 'margin': 10, 'borderColor': '#1890ff'}
    ),
    dcc.Input(
        id='col-c-3d',
        placeholder='Kolumna c',
        value='',
        style={'padding': 10, 'width': '100px', 'margin': 10, 'borderColor': '#1890ff'}
    ),
    html.Button('Generuj wykres 3d', id='generate-3d-btn', n_clicks=0,
                style={'margin': 10, 'width': '250px', 'background': '#1890ff', 'borderColor': '#1890ff',
                       'color': '#fff'})
]),
# generate histograms
html.Div([
    dcc.Input(
        id='hist-column',
        placeholder='Nazwa kolumny',
        value='',
        style={'padding': 10, 'width': '150px', 'margin': 10, 'borderColor': '#1890ff'}
    ),
    dcc.Input(
        id='hist-range',
        placeholder='Liczba przedziałów ',
        value='',
        style={'padding': 10, 'width': '200px', 'margin': 10, 'borderColor': '#1890ff'}
    ),
    html.Button('Generuj histogram', id='generate-histogram-btn', n_clicks=0,
                style={'margin': 10, 'width': '200px', 'background': '#1890ff', 'borderColor': '#1890ff',
                       'color': '#fff'})
]),
]),

html.Div([
    dcc.Graph(id='graph-2d', style={'width': '700px', 'margin': 10, 'display': 'inline-block'}),
    dcc.Graph(id='graph-3d', style={'width': '700px', 'margin': 10, 'display': 'inline-block'}),
    dcc.Graph(id='generate-histogram', style={'width': '700px', 'margin': 10, 'display': 'inline-block'}),
]),
html.Div([
    html.Span('Sekcja KNN', style={'font-size': '30px'})
],
style={'text-align': 'center'}),
html.Div([
    html.Button('euclidean', id='euclidean-btn', n_clicks=0,
                style={'margin': 10, 'width': '200px', 'background': '#1890ff', 'borderColor': '#1890ff',
                       'color': '#fff'}),
    html.Button('manhattan', id='manhattan-btn', n_clicks=0,
                    style={'margin': 10, 'width': '200px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'}),
    html.Button('czebyszew', id='czebyszew-btn', n_clicks=0,
                    style={'margin': 10, 'width': '200px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'}),
    html.Button('mahalanobis', id='mahalanobis-btn', n_clicks=0,
                    style={'margin': 10, 'width': '200px', 'background': '#1890ff', 'borderColor': '#1890ff',
                           'color': '#fff'}),
]),
html.Div([
    dcc.Graph(id='euclidean', style={'width': '700px', 'margin': 10, 'display': 'inline-block'}),
    dcc.Graph(id='manhattan', style={'width': '700px', 'margin': 10, 'display': 'inline-block'}),
    dcc.Graph(id='czebyszew', style={'width': '700px', 'margin': 10, 'display': 'inline-block'}),
    dcc.Graph(id='mahalanobis', style={'width': '700px', 'margin': 10, 'display': 'inline-block'}),
]),

dash_table.DataTable(id='df-holder', style_table={'display': 'none'}),
dash_table.DataTable(
    id='table',
    columns=(),
    data=[],
    editable=True
),

html.P(str(CONST_NOW), id='timestamp-load', style={'display': 'none'}),
html.P(str(CONST_NOW), id='timestamp-new-col', style={'display': 'none'}),
html.P(str(CONST_NOW), id='timestamp-new-row', style={'display': 'none'}),
html.P(str(CONST_NOW), id='timestamp-label-enc', style={'display': 'none'}),
html.P(str(CONST_NOW), id='timestamp-range-col', style={'display': 'none'}),
html.P(str(CONST_NOW), id='timestamp-minmax-col', style={'display': 'none'}),
html.P(str(CONST_NOW), id='timestamp-normal-col', style={'display': 'none'}),
html.Div(id='black-hole', style={'display': 'none'}),

html.Div([
    dcc.Input(
        id='save-filename',
        placeholder='Nazwa pliku',
        value='',
        style={'padding': 10, 'width': '200px', 'margin': 10, 'borderColor': '#1890ff'}
    ),
    html.Button('Zapisz csv', id='save-file', n_clicks=0,
                style={'margin': 10, 'width': '150px', 'background': '#1890ff', 'borderColor': '#1890ff',
                       'color': '#fff'}),
]),
])


@app.callback(Output('black-hole', 'children'),
              [Input('save-file', 'n_clicks')],
              [State('save-filename', 'value'), State('table', 'data')])
def save_file(n_clicks, filename, table_data):
    if n_clicks > 0:
        df = pd.DataFrame(table_data)
        df.to_csv(filename + '.csv', index=False)
    return None


# GRAPHS

@app.callback(Output('euclidean', 'figure'),
              [Input('euclidean-btn', 'n_clicks')],
              [State('table', 'data')])
def generate_euclidean(n_clicks, table_data):
    if n_clicks > 0:
        df = pd.DataFrame(table_data)
        df = df.fillna(df.mean())
        if 'variety' in df:
            X = df.drop('variety', axis=1)
            Y = df['variety']
            model = KNN(metric='euclidean')
            model.fit(X,Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='euclidean'
                )
            }
        elif 'Hrabstwo' in df:
            X = df.drop('Hrabstwo', axis=1)
            Y = df['Hrabstwo']
            model = KNN(metric='euclidean')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='euclidean'
                )
            }
        elif 'class' in df:
            X = df.drop('class', axis=1)
            Y = df['class']
            model = KNN(metric='euclidean')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='euclidean'
                )
            }
    return {
        'data': [],
        'layout': go.Layout(title='euclidean'),
    }

@app.callback(Output('manhattan', 'figure'),
              [Input('manhattan-btn', 'n_clicks')],
              [State('table', 'data')])
def generate_manhattan(n_clicks, table_data):
    if n_clicks > 0:
        df = pd.DataFrame(table_data)
        df = df.fillna(df.mean())
        if 'variety' in df:
            X = df.drop('variety', axis=1)
            Y = df['variety']
            model = KNN(metric='manhattan')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='manhattan'
                )
            }
        elif 'Hrabstwo' in df:
            X = df.drop('Hrabstwo', axis=1)
            Y = df['Hrabstwo']
            model = KNN(metric='manhattan')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='manhattan'
                )
            }
        elif 'class' in df:
            X = df.drop('class', axis=1)
            Y = df['class']
            model = KNN(metric='manhattan')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='manhattan'
                )
            }
    return {
        'data': [],
        'layout': go.Layout(title='manhattan'),
    }

@app.callback(Output('czebyszew', 'figure'),
              [Input('czebyszew-btn', 'n_clicks')],
              [State('table', 'data')])
def generate_czebyszew(n_clicks, table_data):
    if n_clicks > 0:
        df = pd.DataFrame(table_data)
        df = df.fillna(df.mean())
        if 'variety' in df:
            X = df.drop('variety', axis=1)
            Y = df['variety']
            model = KNN(metric='czebyszew')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='czebyszewv'
                )
            }
        elif 'Hrabstwo' in df:
            X = df.drop('Hrabstwo', axis=1)
            Y = df['Hrabstwo']
            model = KNN(metric='czebyszew')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='czebyszew'
                )
            }
        elif 'class' in df:
            X = df.drop('class', axis=1)
            Y = df['class']
            model = KNN(metric='czebyszew')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='czebyszew'
                )
            }
    return {
        'data': [],
        'layout': go.Layout(title='czebyszew'),
    }

@app.callback(Output('mahalanobis', 'figure'),
              [Input('mahalanobis-btn', 'n_clicks')],
              [State('table', 'data')])
def generate_mahalanobis(n_clicks, table_data):
    if n_clicks > 0:
        df = pd.DataFrame(table_data)
        df = df.fillna(df.mean())
        if 'variety' in df:
            X = df.drop('variety', axis=1)
            Y = df['variety']
            model = KNN(metric='mahalanobis')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='mahalanobis'
                )
            }
        elif 'Hrabstwo' in df:
            X = df.drop('Hrabstwo', axis=1)
            Y = df['Hrabstwo']
            model = KNN(metric='mahalanobis')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='mahalanobis'
                )
            }
        elif 'class' in df:
            X = df.drop('class', axis=1)
            Y = df['class']
            model = KNN(metric='mahalanobis')
            model.fit(X, Y)
            data = model.report('test')

            return {
                'data': [data],
                'layout': go.Layout(
                    xaxis={'title': 'K nearest neighbors'},
                    yaxis={'title': 'Accuracy'},
                    margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                    clickmode='event+select',
                    title='mahalanobis'
                )
            }
    return {
        'data': [],
        'layout': go.Layout(title='mahalanobis'),
    }


@app.callback(Output('generate-histogram', 'figure'),
              [Input('generate-histogram-btn', 'n_clicks')],
              [State('hist-column', 'value'),
               State('hist-range', 'value'), State('table', 'data')])
def generate_histogram(n_clicks, decision, range, table_data):
    if n_clicks > 0 and decision and range:
        df = pd.DataFrame(table_data)
        data = df[decision]
        min_val = data.min()
        max_val = data.max()
        size = (max_val - min_val) / int(range)

        return {
            'data': [go.Histogram(x=df[decision], xbins=dict(
                      start=min_val,
                      end=max_val,
                      size=size),
                      autobinx=False)],
            'layout': go.Layout(
                xaxis={'title': decision},
                yaxis={'title': 'Quantity'},
                margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                clickmode='event+select',
                title='histogram'
            )
        }
    elif n_clicks > 0 and decision:
        df = pd.DataFrame(table_data)

        return {
            'data': [go.Histogram(x=df[decision])],
            'layout': go.Layout(
                xaxis={'title': decision},
                yaxis={'title': 'Ilość'},
                margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                clickmode='event+select',
                title='histogram'
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(title='histogram'),
        }

@app.callback(Output('graph-2d', 'figure'),
              [Input('generate-2d-btn', 'n_clicks')],
              [State('col-decision', 'value'),
               State('col-a-2d', 'value'), State('col-b-2d', 'value'), State('table', 'data')])
def generate_2d(n_clicks, decision, column_a, column_b, table_data):
    if n_clicks > 0 and decision and column_a and column_b:
        df = pd.DataFrame(table_data)
        classes = df[decision].unique()

        data = []

        for d_class in classes:
            df_for_class = df[df[decision] == d_class]
            x = df_for_class[column_a]
            y = df_for_class[column_b]
            data.append(go.Scatter(
                x=x, y=y,
                name=str(d_class),
                opacity=0.8,
                mode='markers',
                marker={'size': 10}
            ))

        return {
            'data': data,
            'layout': go.Layout(
                xaxis={'title': str(column_a)},
                yaxis={'title': str(column_b)},
                margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                clickmode='event+select',
                title='wykres2d'
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(title='wykres2d')
        }

@app.callback(Output('graph-3d', 'figure'),
              [Input('generate-3d-btn', 'n_clicks')],
              [State('col-decision', 'value'),
               State('col-a-3d', 'value'), State('col-b-3d', 'value'), State('col-c-3d', 'value'),
               State('table', 'data')])
def generate_3d(n_clicks, decision, column_a, column_b, column_c, table_data):
    if n_clicks > 0 and decision and column_a and column_b and column_c:
        df = pd.DataFrame(table_data)
        classes = df[decision].unique()

        data = []

        for d_class in classes:
            df_for_class = df[df[decision] == d_class]

            x = df_for_class[column_a]
            y = df_for_class[column_b]
            z = df_for_class[column_c]

            data.append(go.Scatter3d(
                x=x, y=y, z=z,
                name=str(d_class),
                opacity=0.8,
                mode='markers',
                marker={'size': 6}
            ))

        return {
            'data': data,
            'layout': dict(
                xaxis={'title': str(column_a)},
                yaxis={'title': str(column_b)},
                zaxis={'title': str(column_c)},
                margin={'l': 40, 'b': 40, 't': 30, 'r': 0},
                clickmode='event+select',
                title='wykres3d'
            )
        }
    else:
        return {
            'data': [],
            'layout': go.Layout(title='wykres3d')
        }


# END: GRAPHS


# TIMESTAMPS


@app.callback([Output('df-holder', 'data'), Output('timestamp-load', 'children')],
              [Input('df-upload', 'contents')],
              [State('df-upload', 'filename')])
def read_df(contents, filename):
    if contents is None:
        return [{}], CONST_NOW
    df = parse_contents(contents, filename)
    return df.to_dict('records'), str(datetime.now())


@app.callback(Output('timestamp-new-col', 'children'),
              [Input('add-column', 'n_clicks')],
              [State('timestamp-new-col', 'children')])
def read_new_column(n_clicks, timestamp):
    if n_clicks > 0:
        return str(datetime.now())
    else:
        return timestamp


@app.callback(Output('timestamp-new-row', 'children'),
              [Input('add-row', 'n_clicks')],
              [State('timestamp-new-row', 'children')])
def read_new_row(n_clicks, timestamp):
    if n_clicks > 0:
        return str(datetime.now())
    else:
        return timestamp


@app.callback(Output('timestamp-label-enc', 'children'),
              [Input('lab-btn', 'n_clicks')],
              [State('timestamp-label-enc', 'children')])
def label_column(n_clicks, timestamp):
    if n_clicks > 0:
        return str(datetime.now())
    else:
        return timestamp


@app.callback(Output('timestamp-range-col', 'children'),
              [Input('rng-btn', 'n_clicks')],
              [State('timestamp-range-col', 'children')])
def label_column(n_clicks, timestamp):
    if n_clicks > 0:
        return str(datetime.now())
    else:
        return timestamp


@app.callback(Output('timestamp-minmax-col', 'children'),
              [Input('minmax-btn', 'n_clicks')],
              [State('timestamp-minmax-col', 'children')])
def label_column(n_clicks, timestamp):
    if n_clicks > 0:
        return str(datetime.now())
    else:
        return timestamp


@app.callback(Output('timestamp-normal-col', 'children'),
              [Input('normalize-btn', 'n_clicks')],
              [State('timestamp-normal-col', 'children')])
def label_column(n_clicks, timestamp):
    if n_clicks > 0:
        return str(datetime.now())
    else:
        return timestamp


# END: TIMESTAMPS


@app.callback([Output('table', 'data'), Output('table', 'columns')],
              [Input('df-holder', 'data'), Input('timestamp-load', 'children'),
               Input('timestamp-new-col', 'children'), Input('timestamp-new-row', 'children'),
               Input('timestamp-label-enc', 'children'), Input('timestamp-range-col', 'children'),
               Input('timestamp-minmax-col', 'children'), Input('timestamp-normal-col', 'children')],
              [State('new-column', 'value'), State('table', 'data'), State('table', 'columns'),
               State('label-column', 'value'), State('range-column', 'value'), State('range-amount', 'value'),
               State('minmax-column', 'value'), State('a-val', 'value'), State('b-val', 'value'),
               State('normalize-column', 'value')])
def update_output(
        rows,
        timestamp_load, timestamp_col, timestamp_row, timestamp_lab, timestanp_rng, timestamp_minmax, timestamp_normal,
        column_name, table_data, existing_columns, lab_column, rng_column, amount, minmax_column, a, b, normal_column
):
    if rows is None:
        return [{}], []
    df = pd.DataFrame(rows)

    time_load = string_to_datetime(timestamp_load)
    time_col = string_to_datetime(timestamp_col)
    time_row = string_to_datetime(timestamp_row)
    time_lab = string_to_datetime(timestamp_lab)
    time_rng = string_to_datetime(timestanp_rng)
    time_minmax = string_to_datetime(timestamp_minmax)
    time_normal = string_to_datetime(timestamp_normal)

    all_times = [time_load, time_col, time_row, time_lab, time_rng, time_minmax, time_normal]

    if time_load == max(all_times):
        columns = [
            {'id': column, 'name': column, 'deletable': True, 'renamable': True} for i, column in enumerate(df.columns)
        ]
        return df.to_dict('records'), columns
    elif time_col == max(all_times):
        if column_name:
            existing_columns.append({'id': column_name, 'name': column_name, 'deletable': True, 'renamable': True})
        return table_data, existing_columns
    elif time_row == max(all_times):
        table_data.append({c['id']: '' for c in existing_columns})
        return table_data, existing_columns
    elif time_lab == max(all_times):
        if lab_column:
            df = pd.DataFrame(table_data)
            df, new_col = text_to_labels(df, lab_column)
            table_data = df.to_dict('records')
            if new_col:
                existing_columns.append({'id': new_col, 'name': new_col, 'deletable': True, 'renamable': True})
        return table_data, existing_columns
    elif time_rng == max(all_times):
        if rng_column and amount and amount.isdigit():
            df = pd.DataFrame(table_data)
            df, new_col = make_ranges(df, rng_column, int(amount))
            table_data = df.to_dict('records')
            if new_col:
                existing_columns.append({'id': new_col, 'name': new_col, 'deletable': True, 'renamable': True})
        return table_data, existing_columns
    elif time_minmax == max(all_times):
        if minmax_column and a and b and a.isdigit() and b.isdigit():
            df = pd.DataFrame(table_data)
            df, new_col = minmax_scale(df, minmax_column, int(a), int(b))
            table_data = df.to_dict('records')
            if new_col:
                existing_columns.append({'id': new_col, 'name': new_col, 'deletable': True, 'renamable': True})
        return table_data, existing_columns
    elif time_normal == max(all_times):
        if normal_column:
            df = pd.DataFrame(table_data)
            df, new_col = normalize(df, normal_column)
            table_data = df.to_dict('records')
            if new_col:
                existing_columns.append({'id': new_col, 'name': new_col, 'deletable': True, 'renamable': True})
        return table_data, existing_columns
    else:
        return [{}], []


if __name__ == '__main__':
    app.run_server(debug=True)