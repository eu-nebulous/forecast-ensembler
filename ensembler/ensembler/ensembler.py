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
        logging.info(f"[INIT] Initializing Ensembler with config: {config}")
        self.config = config
        self.metrics = None
        self.forecasters_names = None
        self.ensemblers = {}
        self.prediction_dfs = {}
        self.predictions_tables_names = None
        self.influx_data_dwonloader = InfluxDataDownloader()
        self.publishers = {}  # Store Publisher instances for each (app_id -> topic)
        logging.info("[INIT] Ensembler initialization complete.")

    def get_or_create_publisher(self, app_id: str, topic: str) -> Publisher:
        """
        Retrieve or create a Publisher for (app_id, topic).
        Must pass 'address=topic' and 'topic=True' to fix the 'NoneType' error.
        """
        logging.debug(f"[PUBLISHER] Checking publisher for app_id={app_id}, topic={topic}...")
        if app_id not in self.publishers:
            self.publishers[app_id] = {}
            logging.debug(f"[PUBLISHER] No publishers found yet for app_id={app_id}, creating dict.")

        if topic not in self.publishers[app_id]:
            self.publishers[app_id][topic] = Publisher(
                name=f"Publisher_{app_id}_{topic}",
                address=topic,   # JMS/AMQP topic name
                topic=True       # Tells EXN it's a topic
            )
            logging.info(f"[PUBLISHER] Created new Publisher for app_id={app_id}, topic={topic}.")
        else:
            logging.debug(f"[PUBLISHER] Reusing existing Publisher for app_id={app_id}, topic={topic}.")

        return self.publishers[app_id][topic]

    def get_data(self, app_id: str, metric: str, columns: List[str], mock: bool = False):
        """
        Download or mock data, then store/update in self.prediction_dfs[app_id][metric].
        """
        logging.info(f"[DATA] Fetching data for app_id={app_id}, metric={metric}, mock={mock}.")
        if app_id not in self.prediction_dfs:
            self.prediction_dfs[app_id] = {}

        if metric not in self.prediction_dfs[app_id]:
            self.prediction_dfs[app_id][metric] = None

        if mock:
            logging.debug("[DATA] Using mock data.")
            new_data = mock_predictions_df(columns)
        else:
            logging.debug("[DATA] Downloading data from Influx.")
            new_data = self.influx_data_dwonloader.download_data(app_id, metric)

        # Convert to a DataFrame if needed
        if isinstance(new_data, PredictionsDF):
            new_data_df = new_data.df
        else:
            new_data_df = pd.DataFrame(new_data) if isinstance(new_data, list) else new_data

        logging.debug(f"[DATA] New data shape for app_id={app_id}, metric={metric}: {new_data_df.shape}")

        if self.prediction_dfs[app_id][metric] is None:
            # Wrap first usage
            self.prediction_dfs[app_id][metric] = PredictionsDF(new_data_df)
            logging.info(f"[DATA] Created new PredictionsDF for app_id={app_id}, metric={metric}.")
        else:
            # Update the existing PredictionsDF
            logging.debug(f"[DATA] Updating existing PredictionsDF for app_id={app_id}, metric={metric}.")
            self.prediction_dfs[app_id][metric].update(new_data_df)

    def on_start_ensembler(self, body: Dict[str, Any], app_id: str):
        """
        Initialize ensemblers for a specific app_id.
        Expects body to contain "metrics" (list of dicts) and "models" (list).
        """
        logging.info(f"[ENSEMBLER_INIT] Starting ensembler initialization for app_id={app_id}.")
        self.metrics = [metric_info["metric"] for metric_info in body["metrics"]]
        self.forecasters_names = body["models"]
        self.get_predictions_fields()

        logging.debug(f"[ENSEMBLER_INIT] Metrics for app_id={app_id}: {self.metrics}")
        logging.debug(f"[ENSEMBLER_INIT] Forecasters for app_id={app_id}: {self.forecasters_names}")

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
        logging.info(f"[ENSEMBLER_INIT] Created ensemblers for app_id={app_id}.")

        # Ensure self.prediction_dfs exists
        if not hasattr(self, "prediction_dfs"):
            self.prediction_dfs = {}
            logging.debug("[ENSEMBLER_INIT] Created self.prediction_dfs dict.")

        # Initialize an empty PredictionsDF for each metric only if not already created
        if app_id not in self.prediction_dfs:
            self.prediction_dfs[app_id] = {}
            logging.debug(f"[ENSEMBLER_INIT] Created prediction_dfs entry for app_id={app_id}.")

        for metric in self.metrics:
            if metric not in self.prediction_dfs[app_id]:
                logging.debug(f"[ENSEMBLER_INIT] Initializing empty PredictionsDF for metric={metric}.")
                self.prediction_dfs[app_id][metric] = PredictionsDF([])

    def on_ensemble(self, body: Dict[str, Any]):
        """
        Perform ensembling for a single request:
          - body must include {"metric", "method", "app_id", "predictionTime", ...}
        """
        logging.info(f"[ENSEMBLE] Received ensemble request: {body}")
        metric = body["metric"]
        method = body["method"]
        app_id = body["app_id"]
        topic = f"ensembled.{app_id}.{metric}"

        # If no ensemblers for this app, attempt to initialize
        if app_id not in self.ensemblers:
            logging.warning(f"[ENSEMBLE] Ensemblers not found for app_id={app_id}. Attempting initialization...")
            if "metrics" in body and "models" in body:
                self.on_start_ensembler(body, app_id)
            else:
                logging.error(f"[ENSEMBLE] Missing initialization data for app_id={app_id}.")
                raise KeyError(f"Missing initialization data for app_id: {app_id}")

        # Confirm we have the requested metric/method
        if metric not in self.ensemblers[app_id]:
            err_msg = f"Metric '{metric}' not initialized for app_id: {app_id}"
            logging.error(f"[ENSEMBLE] {err_msg}")
            raise KeyError(err_msg)

        if method not in self.ensemblers[app_id][metric]:
            err_msg = f"Method '{method}' not initialized for app_id: {app_id}, metric: {metric}"
            logging.error(f"[ENSEMBLE] {err_msg}")
            raise KeyError(err_msg)

        # Retrieve the specific ensembler
        ensembler = self.ensemblers[app_id][metric][method]

        # Possibly train the model if needed
        if ensembler.available:
            logging.debug(f"[ENSEMBLE] '{method}' ensembler is available for training/prediction.")
            # "Average" typically doesn't need training
            if method != "Average":
                logging.debug(f"[ENSEMBLE] Training method='{method}' with new data.")
                self.get_data(app_id, metric, self.predictions_tables_names[metric])
                ensembler.train(self.prediction_dfs[app_id][metric])
            prediction = ensembler.predict(body)
        else:
            logging.info(f"[ENSEMBLE] Ensembler for {method} is not yet available; attempting to train.")
            self.get_data(app_id, metric, self.predictions_tables_names[metric])
            ensembler.train(self.prediction_dfs[app_id][metric])
            if ensembler.available:
                logging.debug(f"[ENSEMBLE] '{method}' ensembler became available after training.")
                prediction = ensembler.predict(body)
            else:
                logging.info(
                    f"[ENSEMBLE] Ensembler for app_id={app_id}, metric={metric}, method={method} not available. "
                    f"Using Average as fallback."
                )
                prediction = self.ensemblers[app_id][metric]["Average"].predict(body)

        # Build the dictionary with the numeric fields
        ensembling_msg = {
            "ensembledValue": prediction,
            "timestamp": int(time.time()),
            "predictionTime": body["predictionTime"],
        }

        try:
            # Create a final validated response
            final_response = EnsembleResponse(
                status="success",
                **ensembling_msg  # merges in ensembledValue, timestamp, predictionTime
            )
            logging.info(f"[ENSEMBLE] Successfully created EnsembleResponse: {final_response}")
        except ValidationError as exc:
            logging.error(f"[ENSEMBLE] Validation error creating EnsembleResponse: {exc}")
            raise

        # Publish to ActiveMQ / EXN
        # self.send_ensembled_prediction(app_id, topic, final_response.dict())

        logging.info(f"[ENSEMBLE] Finished ensemble for app_id={app_id}, metric={metric}, method={method}.")
        return final_response

    def send_ensembled_prediction(self, app_id: str, topic: str, msg: dict):
        """
        Send the final ensemble prediction to ActiveMQ.
        """
        logging.debug(f"[PUBLISH] Attempting to publish message to topic={topic}: {msg}")
        try:
            publisher = self.get_or_create_publisher(app_id, topic)
            publisher.send_message(msg, application=topic)
            logging.info(f"[PUBLISH] Successfully published message to {topic} for app_id={app_id}.")
        except Exception as e:
            logging.error(f"[PUBLISH] Failed to send ensembled prediction to '{topic}' for app_id={app_id}: {e}")

    def get_predictions_fields(self):
        """
        Build a dict of metric -> list of forecaster-based column names,
        e.g., 'metric.LSTM.prediction'.
        """
        logging.debug("[CONFIG] Building predictions field mapping.")
        self.predictions_tables_names = {
            metric: [
                f"{metric}.{forecaster}.prediction"
                for forecaster in self.forecasters_names
            ]
            for metric in self.metrics
        }
        logging.info(f"[CONFIG] Predictions fields set: {self.predictions_tables_names}")