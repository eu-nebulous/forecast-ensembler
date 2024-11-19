"""Script for data frame with predictions class
"""

import time


class PredictionsDF:
    """Predictions data frame class
    df: data frame with predictions and real value,
    target_column: target column name"""

    def __init__(self, data_frame, target_column="y"):
        """Init method"""
        self.df = data_frame
        self.last_update_time = int(time.time())
        self.target_column = target_column

    def update(self, data_frame):
        """Upade method, change df, and updates last_update_time"""
        self.data_frame = data_frame
        self.last_update_time = int(time.time())
