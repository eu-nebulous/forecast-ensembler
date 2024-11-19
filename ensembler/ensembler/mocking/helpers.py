"""Script with helpers for testing ensemblers, mocking ActiveMQ messages etc."""
import json

import numpy as np
import pandas as pd


class Msg(object):
    """Mocked ActiveMQ message"""

    def __init__(self, body, headers):
        self.body = json.dumps(body)
        self.headers = headers


start_ens_header = {
    "expires": "0",
    "correlation-id": "",
    "destination": "/topic/start_ensembler",
    "subscription": "2",
    "priority": "0",
    "type": "",
    "message-id": "ID:bulls-ThinkPad-T480-39609-1639727895279-6:1:1:1:6",
    "timestamp": "1639738796231",
}
start_ens_body = {
    "metrics": [
        {"metric": "MaxCPULoad", "level": 3, "publish_rate": 60000},
        {"metric": "MinCPULoad", "level": 3, "publish_rate": 50000},
    ],
    "models": ["tft", "nbeats", "gluon"],
}


ens_header = {
    "expires": "0",
    "correlation-id": "",
    "destination": "/topic/ensemble",
    "subscription": "2",
    "priority": "0",
    "type": "",
    "message-id": "ID:bulls-ThinkPad-T480-39609-1639727895279-6:1:1:1:6",
    "timestamp": "1639738796231",
}


ens_body = {
    "method": "Average",
    "metric": "MaxCPULoad",
    "timestamp": 123456789,
    "nbeats": 13,
    "tft": None,
    "gluon": 123,
}


def mock_predictions_df(columns_fields):
    mock_df = pd.DataFrame()
    for column in columns_fields:
        mock_df[column] = np.random.rand(1000)
    mock_df = mock_df.mask(np.random.random(mock_df.shape) < 0.1)
    mock_df["y"] = np.random.rand(1000)
    return mock_df


# start_ensemble_msg = Msg(body, header)
