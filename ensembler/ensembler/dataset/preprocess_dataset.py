import logging
import time

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder

pd.options.mode.chained_assignment = None

"""Script for preparing time series dataset from pythorch-forecasting package 
TODO: add checking whether data consists of multiple series, handle nans values"""


class Dataset(object):
    def __init__(
        self,
        dataset,
        target_column="value",
        time_column="ems_time",
        tv_unknown_reals=[],
        known_reals=[],
        tv_unknown_cat=[],
        static_reals=[],
        classification=0,
        context_length=40,
        prediction_length=5,
        publish_rate=10000,
    ):

        self.max_missing_values = (
            20  # max consecutive missing values allowed per series
        )
        self.target_column = target_column
        self.time_column = time_column
        self.tv_unknown_cat = tv_unknown_cat
        self.known_reals = known_reals
        self.tv_unknown_reals = tv_unknown_reals
        self.static_reals = static_reals
        self.classification = classification
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.publish_rate = publish_rate
        self.dataset = dataset
        self.dropped_recent_series = True  # default set to be true
        if self.dataset.shape[0] > 0:
            self.check_gap()
        self.n = dataset.shape[0]
        if self.dataset.shape[0] > 0:
            self.ts_dataset = self.create_time_series_dataset()

    def cut_nan_start(self, dataset):
        dataset.index = range(dataset.shape[0])
        first_not_nan_index = dataset[self.target_column].first_valid_index()
        if first_not_nan_index == first_not_nan_index:  # check is if it;s not np.nan
            if first_not_nan_index is not None:
                if first_not_nan_index > -1:
                    return dataset[dataset.index > first_not_nan_index]
        else:
            return dataset.dropna()

    def fill_na(self, dataset):
        dataset = dataset.replace(np.inf, np.nan)
        dataset = dataset.ffill(axis="rows")
        return dataset

    def convert_formats(self, dataset):
        if not self.classification:
            dataset[self.target_column] = dataset[self.target_column].astype(float)
        else:
            dataset[self.target_column] = dataset[self.target_column].astype(int)

        for name in self.tv_unknown_cat:
            dataset[name] = dataset[name].astype(str)
        return dataset

    def convert_time_to_ms(self):
        if self.dataset.shape[0] > 0:
            digit_len = len(str(int(self.dataset[self.time_column].values[0])))
            if digit_len >= 13:
                self.dataset[self.time_column] = self.dataset[self.time_column].apply(
                    lambda x: int(str(int(x))[:13])
                )
            else:
                self.dataset[self.time_column] = self.dataset[self.time_column].apply(
                    lambda x: int(int(str(int(x))[:digit_len]) * 10 ** (13 - digit_len))
                )
            self.dataset[self.time_column] = self.dataset[self.time_column].apply(
                lambda x: int(x // 1e4 * 1e4)
            )

    def add_obligatory_columns(self, dataset):
        n = dataset.shape[0]
        dataset["time_idx"] = range(n)  # TODO check time gaps
        return dataset

    def get_time_difference_current(self):
        if self.dataset.shape[0] > 0:
            last_timestamp_database = self.dataset[self.time_column].values[-1]
            current_time = int(time.time())
            print(
                f"Time difference between last timestamp and current time: {current_time - last_timestamp_database}"
            )
            logging.info(
                f"Time difference between last timestamp and current time: {current_time - last_timestamp_database}"
            )

    def check_gap(self):
        if (self.dataset.shape[0] > 0) and (self.target_column in self.dataset.columns):
            self.dataset = self.dataset.groupby(by=[self.time_column]).min()
            self.dataset[self.time_column] = self.dataset.index
            self.dataset.index = range(self.dataset.shape[0])
            self.convert_time_to_ms()
            self.dataset[self.target_column] = pd.to_numeric(
                self.dataset[self.target_column], errors="coerce"
            ).fillna(np.nan)
            self.dataset = self.dataset.replace(np.inf, np.nan)
            self.dataset = self.dataset.dropna(subset=[self.target_column])
            if self.dataset.shape[0] > 0:
                max_gap = self.dataset[self.time_column].diff().abs().max()
                logging.info(
                    f"Metric: {self.target_column} Max time gap in series {max_gap}"
                )
                print(f" Metric: {self.target_column} Max time gap in series {max_gap}")
                series_freq = (
                    (self.dataset[self.time_column])
                    .diff()
                    .fillna(0)
                    .value_counts()
                    .index.values[0]
                )

                logging.info(
                    f"Metric: {self.target_column} Detected series with {series_freq} frequency"
                )
                print(
                    f"Metric: {self.target_column} Detected series with {series_freq} frequency"
                )
                if series_freq != self.publish_rate:
                    logging.info(
                        f"Metric: {self.target_column} Detected series with {series_freq} frequency, but the frequency should be: {self.publish_rate}!"
                    )
                    print(
                        f"Metric: {self.target_column} Detected series with {series_freq} frequency, but the frequency should be: {self.publish_rate}!"
                    )

                # check series length
                series = np.split(
                    self.dataset,
                    *np.where(
                        self.dataset[self.time_column]
                        .diff()
                        .abs()
                        .fillna(0)
                        .astype(int)
                        >= np.abs(self.max_missing_values * self.publish_rate)
                    ),
                )
                logging.info(f"Metric: {self.target_column} {len(series)} series found")
                print(f"{len(series)} series found")
                preprocessed_series = []
                for i, s in enumerate(series):
                    s = self.fill_na(s)
                    s = self.cut_nan_start(s)
                    s = self.add_obligatory_columns(s)
                    s["split"] = "train"
                    s = self.convert_formats(s)
                    logging.info(
                        f"Metric: {self.target_column} Found series {i} of length: {s.shape[0]}, required data rows: {self.prediction_length * 2 + self.context_length}"
                    )
                    if s.shape[0] > self.prediction_length * 2 + self.context_length:
                        s["series"] = i
                        preprocessed_series.append(s)
                    if i == len(series) - 1:
                        logging.info(
                            f"Metric: {self.target_column} Fresh data rows: {s.shape[0]}, required fresh data rows: {self.prediction_length * 2 + self.context_length}"
                        )

                logging.info(
                    f"Metric: {self.target_column} {len(preprocessed_series)} long enough series found"
                )
                print(f"{len(preprocessed_series)} long enough series found")
                # logging.info(f"")
                if preprocessed_series:
                    self.dataset = pd.concat(preprocessed_series)
                    if self.dataset["series"].max() != len(series) - 1:
                        self.dropped_recent_series = True
                    else:
                        self.dropped_recent_series = False
                else:
                    self.dataset = pd.DataFrame()
                    self.dropped_recent_series = True
                self.dataset.index = range(self.dataset.shape[0])
        else:
            self.dataset = pd.DataFrame()
            self.dropped_recent_series = True
            logging.info(f"metric: {self.target_column} no data found")
        if self.dataset.shape[0] > 0:
            self.get_time_difference_current()

    def inherited_dataset(self, split1, split2):
        df1 = (
            self.dataset[lambda x: x.split == split1]
            .groupby("series", as_index=False)
            .apply(lambda x: x.iloc[-self.context_length :])
        )  # previous split fragment
        df2 = self.dataset[lambda x: x.split == split2]  # split part
        inh_dataset = pd.concat([df1, df2])
        inh_dataset = inh_dataset.sort_values(by=["series", "time_idx"])
        inh_dataset = TimeSeriesDataSet.from_dataset(
            self.ts_dataset, inh_dataset, min_prediction_idx=0, stop_randomization=True
        )
        return inh_dataset

    def create_time_series_dataset(self):
        if not self.classification:
            self.time_varying_unknown_reals = [
                self.target_column
            ] + self.tv_unknown_reals
            self.time_varying_unknown_categoricals = self.tv_unknown_cat
        else:
            self.time_varying_unknown_reals = self.tv_unknown_reals
            self.time_varying_unknown_categoricals = [
                self.target_column
            ] + self.tv_unknown_cat

        ts_dataset = TimeSeriesDataSet(
            self.dataset[lambda x: x.split == "train"],
            time_idx="time_idx",
            target=self.target_column,
            categorical_encoders={"series": NaNLabelEncoder().fit(self.dataset.series)},
            group_ids=["series"],
            time_varying_unknown_reals=[self.target_column],
            min_encoder_length=self.context_length,
            max_encoder_length=self.context_length,
            max_prediction_length=self.prediction_length,
            min_prediction_length=self.prediction_length,
            add_relative_time_idx=False,
            allow_missings=False,
        )
        return ts_dataset

    def get_from_dataset(self, dataset):
        return TimeSeriesDataSet.from_dataset(
            self.ts_dataset, dataset, min_prediction_idx=0, stop_randomization=True
        )
