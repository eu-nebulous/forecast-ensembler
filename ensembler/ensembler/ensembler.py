"""Script for Ensembler class.
Handles three types of ensembling models:
    * Mean Ensembler
    * Best Subset Ensembler
    * Linear Programming Ensembler
"""
import json
import logging
import time
from typing import List, Any, Dict

import pandas as pd
from ensembler.dataset.data import PredictionsDF
from ensembler.dataset.download_data import InfluxDataDownloader
from ensembler.mocking.helpers import mock_predictions_df
from ensembler.models.ensembler_models import AverageEnsembler, BestSubsetEnsembler, LinnearProgrammingEnsembler

from ensembler.publisher import Publisher
from pydantic import ValidationError

from ensembler.messages_schemas import EnsembleResponse


class Ensembler:
    """Custom listener, parameters."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ensembler with configuration.

        Args:
            config: Configuration object (a dict).
        """
        self.config = config
        self.metrics = None
        self.forecasters_names = None
        self.ensemblers = {}
        self.prediction_dfs = {}
        self.predictions_tables_names = None
        self.influx_data_dwonloader = InfluxDataDownloader()
        self.publishers = {}  # Store Publisher instances for each (app_id -> topic)

    def get_or_create_publisher(self, app_id: str, topic: str) -> Publisher:
        """
        Retrieve or create a Publisher for (app_id, topic).
        Must pass 'address=topic' and 'topic=True' to fix the 'NoneType' error.
        """
        if app_id not in self.publishers:
            self.publishers[app_id] = {}

        if topic not in self.publishers[app_id]:
            self.publishers[app_id][topic] = Publisher(
                name=f"Publisher_{app_id}_{topic}",
                address=topic,   # JMS/AMQP topic name
                topic=True       # Tells EXN it's a topic
            )
            logging.info(f"Created new Publisher for app_id={app_id}, topic={topic}.")

        return self.publishers[app_id][topic]

    def get_data(self, app_id: str, metric: str, columns: List[str], mock: bool = False):
        """
        Download or mock data, then store/update in self.prediction_dfs[app_id][metric].
        """
        if app_id not in self.prediction_dfs:
            self.prediction_dfs[app_id] = {}

        if metric not in self.prediction_dfs[app_id]:
            self.prediction_dfs[app_id][metric] = None

        # Fetch new data
        new_data = mock_predictions_df(columns) if mock \
            else self.influx_data_dwonloader.download_data(app_id, metric)

        # Convert to a DataFrame if needed
        if isinstance(new_data, PredictionsDF):
            new_data_df = new_data.df
        else:
            new_data_df = pd.DataFrame(new_data) if isinstance(new_data, list) else new_data

        if self.prediction_dfs[app_id][metric] is None:
            # Wrap first usage
            self.prediction_dfs[app_id][metric] = PredictionsDF(new_data_df)
        else:
            # Update the existing PredictionsDF
            self.prediction_dfs[app_id][metric].update(new_data_df)

    def on_start_ensembler(self, body: Dict[str, Any], app_id: str):
        """
        Initialize ensemblers for a specific app_id.
        Expects body to contain "metrics" (list of dicts) and "models" (list).
        """
        self.metrics = [metric_info["metric"] for metric_info in body["metrics"]]
        self.forecasters_names = body["models"]
        self.get_predictions_fields()

        if app_id not in self.ensemblers:
            self.ensemblers[app_id] = {}

        self.ensemblers[app_id] = {
            metric: {
                "Average": AverageEnsembler(True, self.forecasters_names),
                "BestSubset": BestSubsetEnsembler(False, self.forecasters_names),
                "Linear_programming": LinnearProgrammingEnsembler(False, self.forecasters_names),
            }
            for metric in self.metrics
        }

        # Initialize an empty PredictionsDF for each metric
        self.prediction_dfs[app_id] = {
            metric: PredictionsDF([]) for metric in self.metrics
        }

    def on_ensemble(self, body: Dict[str, Any]):
        """
        Perform ensembling for a single request:
          - body must include {"metric", "method", "app_id", "predictionTime", ...}
        """
        metric = body["metric"]
        method = body["method"]
        app_id = body["app_id"]
        topic = f"ensembled.{app_id}.{metric}"

        # If no ensemblers for this app, attempt to initialize
        if app_id not in self.ensemblers:
            logging.warning(f"Ensemblers not found for app_id: {app_id}. Initializing...")
            if "metrics" in body and "models" in body:
                self.on_start_ensembler(body, app_id)
            else:
                raise KeyError(f"Missing initialization data for app_id: {app_id}")

        # Confirm we have the requested metric/method
        if metric not in self.ensemblers[app_id]:
            raise KeyError(f"Metric '{metric}' not initialized for app_id: {app_id}")
        if method not in self.ensemblers[app_id][metric]:
            raise KeyError(f"Method '{method}' not initialized for app_id: {app_id}, metric: {metric}")

        # Retrieve the specific ensembler
        ensembler = self.ensemblers[app_id][metric][method]

        # Possibly train the model if needed
        if ensembler.available:
            # "Average" typically doesn't need training
            if method != "Average":
                self.get_data(app_id, metric, self.predictions_tables_names[metric])
                ensembler.train(self.prediction_dfs[app_id][metric])
            prediction = ensembler.predict(body)
        else:
            # If not available, fetch data, train, see if it becomes available
            self.get_data(app_id, metric, self.predictions_tables_names[metric])
            ensembler.train(self.prediction_dfs[app_id][metric])
            if ensembler.available:
                prediction = ensembler.predict(body)
            else:
                # fallback to "Average"
                logging.info(
                    f"Ensembler for app_id: {app_id}, metric: {metric} not available. "
                    f"Returning average prediction instead."
                )
                prediction = self.ensemblers[app_id][metric]["Average"].predict(body)

        # Build the dictionary with the numeric fields
        # "ensembledValue" is your final prediction
        # "timestamp" is "now" in seconds
        # "predictionTime" is from the request
        ensembling_msg = {
            "ensembledValue": prediction,
            "timestamp": int(time.time()),
            "predictionTime": body["predictionTime"],
        }

        # We also set up 'status' and 'data' to match our model exactly
        # For 'data', pass any extra debugging, raw body, etc. as a dict


        try:
            # Create a final validated response
            final_response = EnsembleResponse(
                status="success",
                **ensembling_msg     # merges in ensembledValue, timestamp, predictionTime
            )
            logging.info(final_response)
        except ValidationError as exc:
            logging.error(f"Validation error creating EnsembleResponse: {exc}")
            raise

        # Publish to ActiveMQ / EXN
        # self.send_ensembled_prediction(app_id, topic, final_response.dict())

        # Return the Pydantic model instance
        logging.info(final_response)
        return final_response

    def send_ensembled_prediction(self, app_id: str, topic: str, msg: dict):
        """
        Send the final ensemble prediction to ActiveMQ.
        """
        try:
            publisher = self.get_or_create_publisher(app_id, topic)
            publisher.send_message(msg, application=topic)
        except Exception as e:
            logging.error(f"Failed to send ensembled prediction to '{topic}': {e}")

    def get_predictions_fields(self):
        """
        Build a dict of metric -> list of forecaster-based column names,
        e.g., 'metric.LSTM.prediction'.
        """
        self.predictions_tables_names = {
            metric: [
                f"{metric}.{forecaster}.prediction"
                for forecaster in self.forecasters_names
            ]
            for metric in self.metrics
        }
