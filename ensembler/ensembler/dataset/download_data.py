"""Script for downloading data from influx
"""

import os

import pandas as pd
from influxdb import DataFrameClient


class InfluxDataDownloader:
    def __init__(
        self,
    ) -> None:
        """class for downloading data from inlux,
        necessary are columns with predictions and
        real values
        """

        self.influx_client = DataFrameClient(
            host=os.environ.get("INFLUXDB_HOSTNAME", "localhost"),
            port=int(os.environ.get("INFLUXDB_PORT", "8086")),
            username=os.environ.get("INFLUXDB_USERNAME", "morphemic"),
            password=os.environ.get("INFLUXDB_PASSWORD", "password"),
        )
        self.influx_client.switch_database(
            os.environ.get("INFLUXDB_DBNAME", "morphemic")
        )

    @staticmethod
    def convert_timestamp(data_frame: pd.DataFrame) -> pd.DataFrame:
        """convert date index to desired format

        Args:
        -------
            data_frame (pd.DataFrame): data frame with
            time index (pandas time index)

        Returns:
        -------
            pd.DataFrame: data frame with date index
            with desired format
        """
        return pd.to_datetime(data_frame.index, unit="s").tz_convert(None)

    def download_predictions(self, metric_name: str) -> pd.DataFrame:
        """Download predicted values from influx

        Returns:
        -------
            pd.DataFrame: pandas data
             frame with predictions values
        """
        return self.influx_client.query(
            f'SELECT * FROM "{metric_name}Predictions" WHERE time > now() - {os.environ.get("MAX_PAST_DAYS", 100)}d'
        )[f"{metric_name}Predictions"]

    def download_real(self, start_time: pd.DatetimeIndex) -> pd.DataFrame:
        """Download real values from influx

        Args:
        -------
            start_time (pd.DatetimeIndex): first
            date with predictions,

        Returns:
        -------
            pd.DataFrame: pandas data
             frame with real values from PS
        """
        return self.influx_client.query(
            f'SELECT * FROM "{os.environ.get("APP_NAME", "default_application")}" WHERE time > {start_time}'
        )[os.environ.get("APP_NAME", "default_application")]

    def download_data(self, metric_name: str, predictions_freq: int) -> pd.DataFrame:
        """
        Download data from inlux
        (2 tables one with predictions, second with
        real values from PS), merge data and save them to csv

        Args:
        ------
            metric_name (str): metric name
            predictions_freq (int): predictions
            frequency (in seconds)

        Returns:
        -------
            pd.DataFrame: pandas data
             frame with real and predicted values
        """
        predictions = self.download_predictions(metric_name)
        predictions = self.convert_timestamp(predictions)
        start_time = predictions.index.values[0]

        real = self.download_data(start_time)
        real.index = real["ems_time"]
        real = self.convert_timestamp(real)

        predictions = predictions.resample(
            f"{predictions_freq}S", origin=start_time
        ).mean()
        real = (
            real.resample(f"{predictions_freq}S", origin=start_time)
            .mean()
            .rename({metric_name: "y"}, axis=1)
        )["y"]

        merged = pd.merge(
            predictions,
            real,
            how="left",
            left_index=True,
            right_index=True,
        ).dropna(subset=["y"])

        return merged
