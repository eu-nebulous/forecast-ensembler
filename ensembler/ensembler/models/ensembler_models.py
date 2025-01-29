"""Script for ensembling methods"""
import logging
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
        """
        Initialization, 'available' is always True for mean ensembler;
        other ensemblers need to collect historical data before training.
        """
        self.available = available
        self.forecasters = forecasters
        self.metric = metric
        self.last_n_rows = last_n_rows
        logging.info(
            f"[BaseEnsembler] Initialized with forecasters={forecasters}, "
            f"metric={metric}, last_n_rows={last_n_rows}, available={available}"
        )

    def train(self, historical_df):
        """
        Training function for ensemblers. For neural networks, ordinary training
        is performed. For other ensemblers, forecasters' weights are updated
        according to historical data.
        """
        pass

    def predict(self, msg_body):
        """Prediction function for ensemblers."""
        pass

    @staticmethod
    def fill_nans(data_frame: pd.DataFrame, columns):
        """Fill NaNs using the columns' row-wise average."""
        logging.debug(f"[fill_nans] Before filling, NaN counts per column:\n{data_frame[columns].isna().sum()}")
        data_frame[columns] = (
            data_frame[columns]
            .T.fillna(data_frame[columns].mean(axis=1))
            .T
        )
        data_frame = data_frame.fillna(0)
        logging.debug(f"[fill_nans] After filling, NaN counts per column:\n{data_frame[columns].isna().sum()}")
        return data_frame


class AverageEnsembler(BaseEnsembler):
    """Mean ensembler class"""

    def predict(self, msg_body):
        """Prediction is just an average of available forecasters."""
        logging.info("[AverageEnsembler] predict() called")
        logging.debug(f"[AverageEnsembler] msg_body: {msg_body}")

        predicted_values = [
            float(msg_body["predictionsToEnsemble"][forecaster])
            for forecaster in self.forecasters
            if msg_body["predictionsToEnsemble"].get(forecaster) is not None
        ]
        logging.debug(f"[AverageEnsembler] Extracted predictions: {predicted_values}")

        prediction = np.mean(np.array(predicted_values)) if predicted_values else 0.0
        logging.info(f"[AverageEnsembler] Final prediction: {prediction}")
        return prediction


class BestSubsetEnsembler(BaseEnsembler):
    """
    Combination (from all possible combinations of top-k forecasters)
    that yields the lowest MAE on historical data.
    """

    def __init__(self, *args, top_k_limit=5, **kwargs):
        super(BestSubsetEnsembler, self).__init__(*args, **kwargs)
        self.top_k = top_k_limit
        self.last_update_time = int(time.time())
        self.selected_forecasters = []
        self.prediction_columns = []
        logging.info(
            f"[BestSubsetEnsembler] Initialized with top_k_limit={self.top_k}"
        )

    def select_top_k(self, df: pd.DataFrame):
        """
        Select top-k forecasters based on historical MAE.
        This modifies df[prediction_columns] to store absolute errors for each forecaster.
        """
        logging.info("[BestSubsetEnsembler] select_top_k() called")
        # Convert each prediction column to absolute error = |pred - y|
        df[self.prediction_columns] = df[self.prediction_columns].apply(
            lambda col: np.abs(col.values - df["y"].values)
        )

        # Compute mean errors per column
        mean_errors = df[self.prediction_columns].mean().sort_values(ascending=True)
        logging.debug(f"[BestSubsetEnsembler] Mean errors:\n{mean_errors}")

        # Pick the top_k forecasters with smallest error
        top_k = list(mean_errors.index[: self.top_k])
        logging.info(f"[BestSubsetEnsembler] top_k forecasters: {top_k}")
        return top_k

    def train(self, historical_df):
        """Select the best subset of forecasters using MAE on historical predictions."""
        logging.info("[BestSubsetEnsembler] train() called")

        if not hasattr(historical_df, "df") or not hasattr(historical_df.df, "columns"):
            raise TypeError("Expected 'historical_df' to have a '.df' attribute that is a DataFrame.")

        # Extract the real DataFrame
        df = historical_df.df
        logging.debug(f"[BestSubsetEnsembler] DataFrame columns: {list(df.columns)}")

        # Find columns ending with '.prediction'
        self.prediction_columns = [col for col in df.columns if re.search(r"prediction$", col)]
        logging.info(f"[BestSubsetEnsembler] Found prediction columns: {self.prediction_columns}")

        # Fill missing values in these columns
        df = self.fill_nans(df, self.prediction_columns)

        # Consider only the last N rows
        df = df.tail(self.last_n_rows)
        logging.debug(
            f"[BestSubsetEnsembler] Using last {self.last_n_rows} rows. "
            f"Shape after tail: {df.shape}"
        )

        # Select top-k predictions
        top_k_predictions = self.select_top_k(df)

        # Generate all possible subsets from these top-k forecasters
        all_top_k_subsets = list(
            chain.from_iterable(
                combinations(top_k_predictions, r) for r in range(len(top_k_predictions) + 1)
            )
        )
        logging.debug(f"[BestSubsetEnsembler] Generated {len(all_top_k_subsets)} subsets from top_k")

        # Evaluate each subset by mean absolute error (MAE)
        best_mae = None
        best_subset = None
        for subset in all_top_k_subsets:
            if subset:  # skip empty subset
                # Average the columns in this subset
                prediction = df[list(subset)].mean(axis=1)
                mae = mean_absolute_error(prediction, df["y"])
                logging.debug(f"[BestSubsetEnsembler] Subset={subset}, MAE={mae}")

                if best_mae is None or mae < best_mae:
                    best_mae = mae
                    best_subset = subset

        logging.info(f"[BestSubsetEnsembler] Best subset: {best_subset}, MAE={best_mae}")

        if best_subset:
            # Convert column names like "metric.LSTM.prediction" => "LSTM"
            # Example: "AccumulatedSecondsPendingRequests.lstm.prediction".split('.') => ["AccumulatedSecondsPendingRequests", "lstm", "prediction"]
            # We'll take [1] or handle it more robustly in case of different splits
            final_list = []
            for col in best_subset:
                parts = col.split(".")
                if len(parts) == 3:
                    final_list.append(parts[1])  # "lstm"
                elif len(parts) == 2:
                    final_list.append(parts[0])  # "exponentialsmoothing"
                else:
                    # fallback
                    final_list.append(".".join(parts[:-1]))
            self.selected_forecasters = final_list
        else:
            self.selected_forecasters = []

        logging.info(f"[BestSubsetEnsembler] Selected forecasters: {self.selected_forecasters}")
        self.last_update_time = int(time.time())
        self.available = True

    def predict(self, msg_body):
        """Combine selected forecasters' predictions by average."""
        logging.info("[BestSubsetEnsembler] predict() called")
        logging.debug(f"[BestSubsetEnsembler] msg_body: {msg_body}")

        # Build a DataFrame from the new predictions
        # self.forecasters are your *all possible* forecasters
        predicted_values = pd.DataFrame(
            {
                forecaster: [msg_body["predictionsToEnsemble"].get(forecaster, 0.0)]
                for forecaster in self.forecasters
            }
        )
        logging.debug(f"[BestSubsetEnsembler] Constructed predictions DataFrame:\n{predicted_values}")

        # Fill NaNs in all forecasters
        predicted_values = self.fill_nans(predicted_values, self.forecasters)
        logging.debug(f"[BestSubsetEnsembler] DataFrame after fill_nans:\n{predicted_values}")

        # If no selected forecasters, return 0.0 or a default
        if not self.selected_forecasters:
            logging.info("[BestSubsetEnsembler] No selected forecasters, returning 0.0")
            return 0.0

        # Average only the selected forecasters
        prediction = predicted_values[self.selected_forecasters].mean(axis=1)
        final_pred = prediction.values[0]
        logging.info(f"[BestSubsetEnsembler] Final prediction: {final_pred}")
        return final_pred


class LinnearProgrammingEnsembler(BaseEnsembler):
    """
    Weights of forecasters calculated with a linear programming approach
    (so that all weights sum up to 1, minimizing squared error).
    """

    def __init__(self, *args, **kwargs):
        super(LinnearProgrammingEnsembler, self).__init__(*args, **kwargs)
        self.last_update_time = int(time.time())
        self.weights = None
        self.prediction_columns = []
        logging.info("[LinnearProgrammingEnsembler] Initialized")

    def train(self, historical_df):
        """Derive weights from historical predictions by solving a linear least-squares problem."""
        logging.info("[LinnearProgrammingEnsembler] train() called")

        if not hasattr(historical_df, "df") or not hasattr(historical_df.df, "columns"):
            raise TypeError("Expected 'historical_df' to have a '.df' attribute that is a DataFrame.")

        df = historical_df.df
        logging.debug(f"[LinnearProgrammingEnsembler] DataFrame columns: {list(df.columns)}")

        self.prediction_columns = [
            col for col in df.columns if re.search(r"prediction$", col)
        ]
        logging.info(f"[LinnearProgrammingEnsembler] Found prediction columns: {self.prediction_columns}")

        # Fill missing values, keep last N
        df = self.fill_nans(df, self.prediction_columns)
        df = df.tail(self.last_n_rows)
        logging.debug(f"[LinnearProgrammingEnsembler] Using last {self.last_n_rows} rows => shape: {df.shape}")

        # Prepare design matrix X (forecasters) and target vector y
        X = df[self.prediction_columns].to_numpy()
        y = df["y"].to_numpy()

        logging.debug(f"[LinnearProgrammingEnsembler] X shape={X.shape}, y shape={y.shape}")

        # Minimize sum of squared errors subject to sum(weights) == 1
        b = cp.Variable(X.shape[1])
        objective = cp.Minimize(cp.sum_squares(X @ b - y))
        constraints = [cp.sum(b) == 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        logging.info(
            f"[LinnearProgrammingEnsembler] Solved optimization, status={prob.status}, objective={result}"
        )

        self.weights = b.value
        logging.info(f"[LinnearProgrammingEnsembler] Learned weights: {self.weights}")
        self.last_update_time = int(time.time())
        self.available = True

    def predict(self, msg_body):
        """Use the learned linear weights to combine forecasters' predictions."""
        logging.info("[LinnearProgrammingEnsembler] predict() called")
        logging.debug(f"[LinnearProgrammingEnsembler] msg_body: {msg_body}")

        predicted_values = pd.DataFrame(
            {
                forecaster: [msg_body["predictionsToEnsemble"].get(forecaster, 0.0)]
                for forecaster in self.forecasters
            }
        )
        logging.debug(f"[LinnearProgrammingEnsembler] Predictions DataFrame:\n{predicted_values}")

        predicted_values = self.fill_nans(predicted_values, self.forecasters)
        logging.debug(f"[LinnearProgrammingEnsembler] After fill_nans:\n{predicted_values}")

        if self.weights is None:
            logging.warning("[LinnearProgrammingEnsembler] Weights are None, defaulting to mean.")
            return predicted_values.mean(axis=1).values[0]

        # Matrix multiply the row by the weights
        prediction = predicted_values.values @ self.weights
        final_pred = prediction[0]
        logging.info(f"[LinnearProgrammingEnsembler] Final prediction: {final_pred}")
        return final_pred