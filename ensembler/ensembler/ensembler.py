"""Script for ensembler class, currenntly 
3 types of ensembling models are available:
    * Mean ensembler
    * Linnear programming
    * Top k on last n
"""

import json
import logging
import threading
import time
from typing import List

import stomp
from slugify import slugify

from ensembler.dataset.data import PredictionsDF
from ensembler.dataset.download_data import InfluxDataDownloader
from ensembler.mocking.helpers import mock_predictions_df
from ensembler.models.ensembler_models import (
    AverageEnsembler,
    BestSubsetEnsembler,
    LinnearProgrammingEnsembler,
)


class Ensembler(stomp.ConnectionListener):
    """Custom listener, parameters:
    - conn (stomp connector)"""

    def __init__(self, conn):
        self.conn = conn
        self.publish_rate = None
        self.metrics = None
        self.forecasters_names = None
        self.ensemblers = {}
        self.prediction_dfs = {}
        self.metrics_frequency = {}
        self.predictions_tables_names = None
        self.influx_data_dwonloader = InfluxDataDownloader()

    def get_data(self, metric: str, columns: List[str], mock: bool = True) -> None:
        """Download data frame with predictions and real
        values for given metric,
        currently mocked.

        Args:
        -----
            metric (str): metric name
            columns (List[str]): list of columns
        """
        if mock:
            if not self.prediction_dfs[metric]:
                self.prediction_dfs[metric] = PredictionsDF(
                    mock_predictions_df(columns)
                )
            else:
                self.prediction_dfs[metric].update(mock_predictions_df(columns))

        else:
            if not self.prediction_dfs[metric]:
                self.prediction_dfs[metric] = PredictionsDF(
                    self.influx_data_dwonloader.download_data(
                        metric, self.metrics_frequency[metric]
                    )
                )
            else:
                self.prediction_dfs[metric].update(
                    self.influx_data_dwonloader.download_data(
                        metric, self.metrics_frequency[metric]
                    )
                )

    def get_predictions_fields(self) -> None:
        """Get predictions columns names"""
        self.predictions_tables_names = {
            metric: [
                f"{metric}.{forecaster}.prediction"
                for forecaster in self.forecasters_names
            ]
            for metric in self.metrics
        }

    def on_error(self, frame):
        """On message error"""
        print(f"received an error {frame.body}")

    def on_start_ensembler(self, body):
        """Get predicted metrics, methods, frequency"""
        self.metrics = [metric["metric"] for metric in body["metrics"]]
        self.forecasters_names = body["models"]
        self.get_predictions_fields()
        self.metrics_frequency = {
            metric_dict["metric"]: metric_dict["publish_rate"]
            for metric_dict in body["metrics"]
        }
        self.ensemblers = {
            metric: {
                "Average": AverageEnsembler(True, self.forecasters_names),
                "BestSubset": BestSubsetEnsembler(False, self.forecasters_names),
                "Linnear_programming": LinnearProgrammingEnsembler(
                    False, self.forecasters_names
                ),
            }
            for metric in self.metrics
        }
        self.prediction_dfs = {metric: None for metric in self.metrics}

    def on_ensemble(self, body):
        """On ensemble message"""
        ensembler = self.ensemblers[body["metric"]][body["method"]]
        if ensembler.available:
            if body["method"] != "Average":
                if (
                    int(time.time()) - ensembler.last_update_time
                    > self.metrics_frequency[body["metric"]] // 1000
                ):
                    self.get_data(
                        body["metric"], self.predictions_tables_names[body["metric"]]
                    )
                    ensembler.train(self.prediction_dfs[body["metric"]])
            prediction = ensembler.predict(body)
        else:
            self.get_data(body["metric"], self.predictions_tables_names[body["metric"]])
            ensembler.train(self.prediction_dfs[body["metric"]])
            if ensembler.available:
                prediction = ensembler.predict(body)
            else:
                print("Ensembler not available returning average prediction insetad")
                logging.info(
                    "Ensembler not available returning average prediction insetad"
                )
                prediction = self.ensemblers[body["metric"]["Average"]].predict(body)

        ensembling_msg = self.create_msg(prediction, body["predictionTime"])
        return ensembling_msg
        # self.send_ensembled_prediction(ensembling_msg, f"ensembled.{body['metric']}")

    @staticmethod
    def is_topic(headers, event):
        """Check is topic"""
        if not hasattr(event, "_match"):
            return False
        match = getattr(event, "_match")
        return headers.get("destination").startswith(match)

    @staticmethod
    def has_topic_name(headers, string):
        """Check topic name"""
        return headers.get("destination").startswith(string)

    @staticmethod
    def get_topic_name(headers):
        """Get topic name"""
        return headers.get("destination").replace("/topic/", "")

    def on_message(self, frame):
        """On any message, runs 'on_[MESSAGE NAME]'"""
        body = json.loads(frame.body)
        headers = frame.headers
        topic_name = slugify(
            headers.get("destination").replace("/topic/", ""),
            separator="_",
        )
        func_name = f"on_{topic_name}"
        if hasattr(self, func_name):
            func = getattr(self, func_name)
            func(body)
        else:
            print(f"Unknonw topic name: {topic_name}")

    @staticmethod
    def create_msg(prediction, prediction_time):
        """Create message with ensembled prediction and prediction time"""
        msg = {
            "ensembledValue": prediction,
            "timestamp": int(time.time()),
            "predictionTime": prediction_time,
        }

        return msg

    def send_ensembled_prediction(self, msg, dest):
        """Send prediction via ActiveMQ"""
        self.conn.send_to_topic(dest, msg)

    def train_ensembling_methods(self):
        """Train esnembling methods (which require time consuming training)"""
        print(self.forecasters_names)
        print("trainig ensemblers!")

    def run_ensemblers_training(self):
        """Run ensemblers training (in sepparate thread)"""
        training_thread = threading.Thread(target=self.train_ensembling_methods)
        training_thread.start()
        training_thread.join()
