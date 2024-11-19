"""Script for pythorch dataset class"""
import random
import re

import numpy as np
import torch
import torch.utils.data as data


class DatasetHistMask(torch.utils.data.Dataset):
    """Defines a dataset for PyTorch, dataset
    networks_prediction_df - data frame with predictions and target
    models_list - list with present forecasters
    max_pred_len - maximum value for prediction steps
    series_len - ensemble series len (historical values len + curent prediction length)
    target_column - target column name in networks_prediction_df
    start_idx - start from this indices networks_prediction_df rows will be chosen
    last_idx - networks_prediction_df rows will be chosen to this index (might be None)
    nan_fill_value - value for filling missing data e.g target value mean
    """

    def __init__(
        self,
        networks_prediction_df,
        models_list=["arima", "cnn"],
        max_pred_len=1,
        series_len=15,
        target_column="y",
        start_idx=0,
        last_idx=None,
        nan_fill_value=100,
    ):
        "Initialization"
        self.df = networks_prediction_df
        self.get_rows(last_idx, start_idx)
        self.models_list = models_list
        assert len(self.models_list) > 1, "There must be more than one forecaster!"
        self.series_len = series_len
        self.max_pred_len = max_pred_len
        self.x_columns = self.get_x_col_names()
        self.series_lengths = self.get_series_lengths()
        self.valid_indices = self.get_valid_indices()
        self.df = self.df[self.x_columns + [target_column, "series_id"]]
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        self.target_column = target_column
        self.nan_fill_value = nan_fill_value

    def get_x_col_names(self):
        """Get forecasters predictions columns names"""
        predictions_col_pattern = "".join(
            [f"^{model}_value|" for model in self.models_list]
        )[:-1]
        return [
            col for col in self.df.columns if re.match(predictions_col_pattern, col)
        ]

    def get_series_lengths(self):
        """Get whole series lenghths using series_id column"""
        return self.df["series_id"].value_counts().sort_values("index").cumsum()

    def get_valid_indices(self):
        """Get valid inidices for series starts according to series_id column"""
        indices = [
            i
            for i in range(self.df.shape[0])
            if self.series_lengths.loc[self.df.iloc[i]["series_id"]] - i
            >= self.series_len + self.max_pred_len
        ]
        return indices

    def get_rows(self, last_idx, start_idx):
        """Get rows according to last_idx, start_idx"""
        if last_idx:
            self.df = self.df.iloc[start_idx:last_idx]
        else:
            self.df = self.df.iloc[start_idx:]

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.valid_indices)

    def get_one_series_df_part(self, idx):
        """Get single series (consecutive rows from
        data frame of length: series_len)"""
        series = self.df.iloc[idx : idx + self.series_len].reset_index()
        series = series.loc[:, ~series.columns.duplicated()]
        return series

    def get_target_value(self, df):
        """Get target value, single y true value which
        we want to approximate with ensembling"""
        assert df[self.target_column].values[-1] != np.nan
        return df[self.target_column].values[-1]

    def replace_future_values(self, col, hist_len):
        """Replace values connected with target from future"""
        col.iloc[hist_len:] = self.nan_fill_value
        return col

    def add_columns(self, x, hist_len):
        """Add extra columns: time_idx, nans masks, is_past"""
        for col in self.x_columns:
            x[f"{col}_res"] = x[col] - x[self.target_column]
            x[f"{col}_res"] = self.replace_future_values(x[f"{col}_res"], hist_len)
        x["time_idx"] = range(self.series_len)
        x["is_past"] = [1 for _ in range(hist_len)] + [
            0 for _ in range(self.series_len - hist_len)
        ]
        for col in self.x_columns:
            x[f"{col}_mask"] = x[col].notna().astype(int)
        return x

    def get_predictions_to_ensemble(self, x):
        """Get forcasters values which will be ensembled"""
        return (
            x[self.x_columns + [f"{col}_mask" for col in self.x_columns]].tail(1).copy()
        )

    @staticmethod
    def to_tensors(x, preds, y):
        """Convert network input to tensor"""
        return (
            torch.tensor(x.fillna(0).to_numpy().astype(np.float32)),
            torch.tensor(preds.fillna(0).values.astype(np.float32)).squeeze(),
            torch.tensor(y),
        )

    def get_one_series(self, idx, hist_len=10):
        """Get single input example"""
        x = self.get_one_series_df_part(idx)
        target = self.get_target_value(x)
        x[self.target_column] = self.replace_future_values(
            x[self.target_column], hist_len
        )
        x = self.add_columns(x, hist_len)
        predictions_to_ensemble = self.get_predictions_to_ensemble(x)
        x = x.fillna(self.nan_fill_value)
        return (x, predictions_to_ensemble, target)

    def __getitem__(self, idx, rand_hist_len=True):
        """Get item, optionaly rand histry length"""
        idx = self.valid_indices[idx]
        if rand_hist_len:
            hist_len = random.randint(
                self.series_len - self.max_pred_len, self.series_len - 1
            )
        else:
            hist_len = (
                self.series_len
                - self.get_one_series_df_part(idx)[self.target_column]
                .iloc[::-1]
                .last_valid_index()
            )
        x, predictions_to_ensemble, target = self.get_one_series(idx, hist_len=hist_len)
        return self.to_tensors(x, predictions_to_ensemble, target)
