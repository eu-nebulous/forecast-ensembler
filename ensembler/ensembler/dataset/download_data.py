import logging
import os
import pandas as pd
from influxdb_client import InfluxDBClient, QueryApi

from ensembler.dataset.data import PredictionsDF


class InfluxDataDownloader:
    def __init__(self) -> None:
        """Initialize the InfluxDB client."""
        self.client = InfluxDBClient(
            url=f"http://{os.environ.get('INFLUXDB_HOSTNAME', 'nebulous-influxdb')}:{os.environ.get('INFLUXDB_PORT', '8086')}",
            token=os.environ.get("INFLUXDB_TOKEN", "my-super-secret-auth-token"),
            org=os.environ.get("INFLUXDB_ORG", "my-org"),
            username=os.environ.get("INFLUXDB_USERNAME", "my-user"),
            password=os.environ.get("INFLUXDB_PASSWORD", "my-password"),
        )
        self.query_api: QueryApi = self.client.query_api()

    @staticmethod
    def convert_timestamp(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the index of the DataFrame to UTC (if it's a DateTimeIndex)
        and remove the timezone information.
        """
        if not isinstance(data_frame.index, pd.DatetimeIndex):
            data_frame.index = pd.to_datetime(data_frame.index)
        return data_frame.tz_localize(None)

    def download_predictions(self, app_id: str, metric_name: str) -> pd.DataFrame:
        """
        Downloads prediction data from an InfluxDB bucket and returns it as a pivoted DataFrame.
        Each forecaster becomes its own column, and the column names end with ".prediction".

        Raises:
            ValueError: If no DataFrames are returned or if the final DataFrame is empty.
        """

        bucket_name = f"nebulous_{app_id}_predicted_bucket"
        flux_query = f'''
            from(bucket: "{bucket_name}")
                |> range(start: -{os.environ.get("MAX_PAST_DAYS", "100")}d)
                |> filter(fn: (r) => r["_measurement"] == "{metric_name}")
                |> filter(fn: (r) => r["_field"] == "metricValue")
        '''

        result = self.query_api.query_data_frame(flux_query)

        # If result is a list of DataFrames, concatenate them
        if isinstance(result, list):
            if not result:
                raise ValueError(
                    f"No data returned for metric '{metric_name}' in bucket '{bucket_name}'."
                )
            df = pd.concat(result, ignore_index=False)
        else:
            df = result

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError(
                f"InfluxDB query returned an empty DataFrame for metric '{metric_name}' in "
                f"bucket '{bucket_name}'."
            )

        # Convert the _time column to datetime and set it as the index
        df["_time"] = pd.to_datetime(df["_time"], errors="coerce")
        df = df.set_index("_time").sort_index()

        # Pivot so that each forecaster is a separate column
        df = df.pivot_table(
            index=df.index,  # Use the already-set _time index
            columns="forecaster",  # Pivot on forecaster names
            values="_value",  # Numeric predictions
            aggfunc="first"  # If duplicates, just take the first (or use "mean" if preferred)
        )

        # Add ".prediction" suffix to all columns
        df = df.add_suffix(".prediction")

        # Prepend metric_name to each column, e.g. "AccumulatedSecondsPendingRequests.lstm.prediction"
        df = df.rename(
            columns=lambda col: f"{metric_name}.{col}" if col.endswith(".prediction") else col
        )

        return df

    def download_real(self, app_id: str, metric_name: str, start_time: str) -> pd.DataFrame:
        """
        Download real values for a specific app_id starting from a specific time.
        """
        bucket_name = f"nebulous_{app_id}_bucket"
        flux_query = f'''
            from(bucket: "{bucket_name}")
            |> range(start: time(v: "{start_time}"))
            |> filter(fn: (r) => r["_measurement"] == "{metric_name}")
            |> filter(fn: (r) => r["_field"] == "metricValue")
        '''
        result = self.query_api.query_data_frame(flux_query)

        if result.empty:
            raise ValueError(
                f"No real data found for app_id: {app_id} starting at {start_time}"
            )

        df = result.set_index("_time").sort_index()
        return df

    def download_data(self, app_id: str, metric_name: str) -> PredictionsDF:
        """
        Download and merge prediction and real values
        for a specific app_id and metric, aligning timestamps
        with an as-of merge (nearest match).
        """
        # 1. Download predictions
        predictions = self.download_predictions(app_id, metric_name)
        predictions.index = pd.to_datetime(predictions.index, errors="coerce")
        predictions = self.convert_timestamp(predictions)

        if predictions.index.empty:
            raise ValueError(
                f"No valid timestamps in predictions for {app_id}, {metric_name}"
            )

        # Convert earliest prediction timestamp to ISO 8601 with 'Z' for UTC
        start_time = predictions.index[0].isoformat() + "Z"

        # 2. Download real data from start_time
        real = self.download_real(app_id, metric_name, start_time)
        real.index = pd.to_datetime(real.index, errors="coerce")
        real = self.convert_timestamp(real)

        logging.info(
            f"[{app_id}:{metric_name}] Downloaded real: {real.shape[0]} rows"
        )

        # Rename the real column from _value to "y"
        real.rename(columns={"_value": "y"}, inplace=True)

        # 3. Use merge_asof() to align the real data to the nearest prediction timestamp
        # Sort both DataFrames by index (time) - merge_asof requires that both are sorted
        predictions_sorted = predictions.sort_index()
        real_sorted = real.sort_index()

        logging.info(
            f"[{app_id}:{metric_name}] Predictions shape: {predictions_sorted.shape}, "
            f"Real shape: {real_sorted.shape}"
        )

        # Merge “as of” the prediction timestamps, taking the nearest real data
        # within a 5-minute tolerance (tweak this as needed).
        merged = pd.merge_asof(
            left=predictions_sorted,
            right=real_sorted[["y"]],
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=pd.Timedelta("5m")
        )
        logging.info(
            f"[{app_id}:{metric_name}] Merged data shape before dropna: {merged.shape}"
        )

        # Drop rows where 'y' is still missing (i.e., if no real data within tolerance)
        merged.dropna(subset=["y"], inplace=True)
        logging.info(
            f"[{app_id}:{metric_name}] Merged data shape after dropna: {merged.shape}"
        )

        return PredictionsDF(merged, target_column="y")
