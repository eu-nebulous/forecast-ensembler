"""Script for data frame with predictions class
"""

import time

import pandas as pd


class PredictionsDF:
    """Predictions data frame class
    df: DataFrame with predictions and real value,
    target_column: name of the target column.
    """

    def __init__(self, data_frame, target_column="y"):
        """Init method"""
        # If data_frame is a list, convert it to a DataFrame
        if isinstance(data_frame, list):
            data_frame = pd.DataFrame(data_frame)

        self.df = data_frame
        self.last_update_time = int(time.time())
        self.target_column = target_column

    def update(self, data_frame):
        """Update method, replaces self.df and last_update_time."""
        # If the new data is a list, convert it
        if isinstance(data_frame, list):
            data_frame = pd.DataFrame(data_frame)

        self.df = data_frame
        self.last_update_time = int(time.time())
