"""Script for ensembling methods"""

import re
import time
from itertools import chain, combinations

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ensembler.mocking.helpers import *


class BaseEnsembler:
    """Base ensembler class"""

    def __init__(self, available, forecasters, metric="cpu", last_n_rows=100):
        """Initialization, available is always true for mean esnembler,
        other ensemblers firstly need to collect some historical data"""
        self.available = available
        self.forecasters = forecasters
        self.metric = metric
        self.last_n_rows = last_n_rows

    def train(self, historcal_df):
        """training function for ensemblers, for neural networks oridinary training
        is performed,
        for other ensemblers just forecasters weights are
        updated according to historical data"""
        pass

    def predict(self, msg_body):
        """prediction function for ensemblers"""
        pass

    @staticmethod
    def fill_nans(data_frame, columns):
        """Fill nans with rows from given columns average"""
        data_frame[columns] = (
            data_frame[columns].T.fillna(data_frame[columns].mean(axis=1)).T
        )
        data_frame = data_frame.fillna(0)
        return data_frame


class AverageEnsembler(BaseEnsembler):
    """Mean ensembler class"""

    def predict(self, msg_body):
        """prediction is just a mean"""
        predicted_values = [
            float(msg_body["predictionsToEnsemble"][forecaster])
            for forecaster in self.forecasters
            if msg_body["predictionsToEnsemble"][forecaster]
        ]
        return np.mean(np.array(predicted_values))


class BestSubsetEnsembler(BaseEnsembler):
    """Combination (from all possible combinantions of top k forecastres)
    of forecastres which boost the prediction performance the most"""

    def __init__(self, *args, top_k_limit=5, **kwargs):
        super(BestSubsetEnsembler, self).__init__(*args, **kwargs)
        self.top_k = top_k_limit
        self.last_update_time = int(time.time())
        self.selected_forecasters = None
        self.prediction_columns = None

    def select_top_k(self, data_frame):
        """Select top K forecasters basing on historcal mae"""
        data_frame[self.prediction_columns] = data_frame[self.prediction_columns].apply(
            lambda x: np.abs(x.values - data_frame["y"].values)
        )
        return list(
            data_frame[self.prediction_columns]
            .mean()
            .sort_values(ascending=True)
            .index[: self.top_k]
        )

    def train(self, historcal_df):
        """Select best subset basing on historical predictions"""
        self.prediction_columns = [
            col for col in historcal_df.df.columns if re.search(r"prediction$", col)
        ]
        historcal_df = self.fill_nans(historcal_df.df, self.prediction_columns)
        historcal_df = historcal_df.tail(self.last_n_rows)
        top_k_predictions = self.select_top_k(historcal_df)
        all_top_k_subsets = list(
            chain.from_iterable(
                combinations(top_k_predictions, r)
                for r in range(len(top_k_predictions) + 1)
            )
        )
        best_mae = None
        for subset in all_top_k_subsets:
            if subset:
                prediction = historcal_df[list(subset)].mean(axis=1)
                mae = mean_absolute_error(prediction, historcal_df["y"])
                if best_mae:
                    if mae < best_mae:
                        best_subset = subset
                        best_mae = mae
                else:
                    best_subset = subset
                    best_mae = mae
        self.selected_forecasters = [
            foreaster.split(".")[1] for foreaster in best_subset
        ]
        self.last_update_time = int(time.time())
        self.available = True

    def predict(self, msg_body):
        """prediction function for ensemblers"""
        predicted_values = pd.DataFrame(
            {
                forecaster: [msg_body["predictionsToEnsemble"][forecaster]]
                for forecaster in self.forecasters
            }
        )
        predicted_values = self.fill_nans(predicted_values, self.forecasters)
        prediction = predicted_values[self.selected_forecasters].mean(axis=1)
        return prediction.values[0]


class LinnearProgrammingEnsembler(BaseEnsembler):
    """Weights of ensemblers calculated with linnear
    porgramming (so that all weights sum up to 1)"""

    def __init__(self, *args, **kwargs):
        super(LinnearProgrammingEnsembler, self).__init__(*args, **kwargs)
        self.last_update_time = int(time.time())
        self.weights = None
        self.prediction_columns = None

    def train(self, historcal_df):
        """Select best subset basing on historical predictions"""
        self.prediction_columns = [
            col for col in historcal_df.df.columns if re.search(r"prediction$", col)
        ]
        historcal_df = self.fill_nans(historcal_df.df, self.prediction_columns)
        historcal_df = historcal_df.tail(self.last_n_rows)
        X = historcal_df[self.prediction_columns].to_numpy()
        y = historcal_df["y"].to_numpy()
        # Construct the problem.
        b = cp.Variable(X.shape[1])
        objective = cp.Minimize(cp.sum_squares(X @ b - y))
        constraints = [cp.sum(b) == 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        self.weights = b.value
        self.last_update_time = int(time.time())
        self.available = True

    def predict(self, msg_body):
        """prediction function for ensemblers"""
        predicted_values = pd.DataFrame(
            {
                forecaster: [msg_body["predictionsToEnsemble"][forecaster]]
                for forecaster in self.forecasters
            }
        )
        predicted_values = self.fill_nans(predicted_values, self.forecasters)
        prediction = predicted_values @ self.weights
        return prediction.values[0]
