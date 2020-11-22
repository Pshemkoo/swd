from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import math
import operator
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from datetime import datetime


def text_to_labels(data, column):
    if not is_string_dtype(data[column]):
        return data, None
    encoder = LabelEncoder()
    new_col = column + '_labeled'
    data[new_col] = encoder.fit_transform(data[column])
    return data, new_col


def make_ranges(data, column, amount):
    if not is_numeric_dtype(data[column]):
        return data, None
    min_val = data[column].min()
    max_val = data[column].max()

    ranges = []

    for i in range(amount):
        left = (max_val - min_val) * i / amount + min_val
        right = (max_val - min_val) * (i + 1) / amount + min_val if i + 1 < amount else 1e9
        ranges.append((left, right))

    def apply_ranges(value, ranges):
        for i, item in enumerate(ranges):
            if item[0] <= value < item[1]:
                return i
        return None

    new_col = column + '_ranged'

    data[new_col] = data[column].apply(lambda x: apply_ranges(x, ranges))

    return data, new_col


def minmax_scale(data, column, a, b):
    if not is_numeric_dtype(data[column]):
        return data, None
    scaler = MinMaxScaler(feature_range=(a, b))
    new_col = column + '_scaled'
    data[new_col] = 0
    data[[new_col]] = scaler.fit_transform(data[[column]])
    return data, new_col


def normalize(data, column):
    if not is_numeric_dtype(data[column]):
        return data, None
    mean = data.loc[:, column].mean()
    std = data.loc[:, column].std()

    def apply_normalization(value, mean, std):
        return (value - mean) / std

    new_col = column + '_normalized'

    data[new_col] = data[column].apply(lambda x: apply_normalization(x, mean, std))

    return data, new_col

class KNN:
    def __init__(self, k=1, metric='euclidean'):
        self.k = k
        self.metric = self._get_distance_method(metric)

    def _get_distance_method(self, metric):

        def _get_euclidean(u, v):
            return math.sqrt(sum([(f - s) ** 2 for f, s in zip(u, v)]))

        def _get_manhattan(u, v):
            return sum([abs(f - s) for f, s in zip(u, v)])

        def _get_czebyszew(u, v):
            return max([abs(f - s) for f, s in zip(u, v)])

        def _get_mahalanobis(u, v):
            u = np.asarray(u)
            v = np.asarray(v)
            VI = np.atleast_2d(self.VI)
            delta = u - v
            m = np.dot(np.dot(delta, VI), delta)
            return np.sqrt(m)

        if metric == 'euclidean':
            return _get_euclidean
        elif metric == 'manhattan':
            return _get_manhattan
        elif metric == 'czebyszew':
            return _get_czebyszew
        elif metric == 'mahalanobis':
            return _get_mahalanobis

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.VI = X.cov()

        self.distances = []

        for (i, v), c in zip(self.X.iterrows(), self.y):
            result = []
            for (j, row), decision in zip(self.X.iterrows(), self.y):
                if i == j:
                    continue
                distance = self.metric(v, row)
                result.append({'distance': distance, 'class': decision})
            self.distances.append(sorted(result, key=lambda x: x['distance']))

    def predict_single(self, v):
        result = []
        for (_, row), decision in zip(self.X.iterrows(), self.y):
            distance = self.metric(v, row)
            result.append({'distance': distance, 'class': decision})

        result = sorted(result, key=lambda x: x['distance'])[: self.k]
        return self._decide_prediction(result)

    def _decide_prediction(self, result):
        amount_per_classes = {}
        for item in result:
            if not item['class'] in amount_per_classes:
                amount_per_classes[item['class']] = 0
            amount_per_classes[item['class']] += 1

        if len(set([value for key, value in amount_per_classes.items()])) == 1:
            #             print(result)
            return result[0]['class']
        else:
            if self.k == 50:
                print(amount_per_classes)
            #             print(max(amount_per_classes.items(), key=operator.itemgetter(1))[0])
            return max(amount_per_classes.items(), key=operator.itemgetter(1))[0]

    def predict(self, X):
        return [self.predict_single(row) for i, row in X.iterrows()]

    def report(self, title):
        total = []
        for k in range(1, self.X.shape[0]):
            predictions = [self._decide_prediction(self.distances[i][: k]) for i in range(self.X.shape[0])]
            result = [real == pred for real, pred in zip(self.y, predictions)]
            total.append(sum(result))
        # fig = go.Figure(data=go.Scatter(x=list(range(1, self.X.shape[0])), y=total, mode='markers'))
        fig = go.Scatter(x=list(range(1, self.X.shape[0])), y=total, mode='markers')
        return fig
        # fig.update_layout(
        #     title={
        #         'text': title,
        #         'y': 0.9,
        #         'x': 0.5,
        #         'xanchor': 'center',
        #         'yanchor': 'top'
        #     },
        #     xaxis_title="K nearest neighbors",
        #     yaxis_title="Accuracy"
        # )
        # fig.show()

    def leave_one_out(self, k=5, report=False):
        predictions = [self._decide_prediction(self.distances[i][: k]) for i in range(self.X.shape[0])]
        result = [real == pred for real, pred in zip(self.y, predictions)]

        if report:
            return classification_report(y, predictions)
        else:
            return sum([real == pred for real, pred in zip(y, predictions)]) / len(y)
