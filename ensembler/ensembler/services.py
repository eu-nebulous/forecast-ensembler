import json
import logging
import threading
import time

from proton import Message
from exn.connector import EXN
from exn.core.context import Context
from exn.core.consumer import Consumer
from exn.core.handler import Handler
from exn.handler.connector_handler import ConnectorHandler

# If your publisher is defined elsewhere, adjust the import accordingly.
# Example assumes a Publisher class at `ensembler.publisher`
from ensembler.publisher import Publisher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('exn.connector').setLevel(logging.DEBUG)

# Initialize your Publisher
publisher = Publisher(name="ensembled_predictions", address="ensembled.predictions", topic=True)


class StartEnsemblingHandler(Handler):
    """
    Handler for 'start_ensembler' messages.
    Invokes self.ensembler.on_start_ensembler(data).
    """
    def __init__(self, ensembler):
        super().__init__()
        self.ensembler = ensembler
        logging.info("StartEnsemblingHandler initialized.")

    def on_message(self, key, address, body, message: Message, context=None):
        logging.info(f"[StartEnsemblingHandler] Received - Key: {key}, Address: {address}, Body: {body}")
        try:
            if body:
                # Handle body as dict or JSON string
                data = body if isinstance(body, dict) else json.loads(body)
                app_id = message.subject
                self.ensembler.on_start_ensembler(data, app_id)
            else:
                logging.warning("Received an empty message body on start_ensembler topic.")
        except Exception as e:
            logging.error(f"Error in StartEnsemblingHandler: {e}")


class EnsembleHandler(Handler):
    """
    Handler for 'on_ensemble' messages.
    Invokes self.ensembler.on_ensemble(data).
    """
    def __init__(self, ensembler):
        super().__init__()
        self.ensembler = ensembler
        logging.info("EnsembleHandler initialized.")

    def on_message(self, key, address, body, message: Message, context=None):
        logging.info(f"[EnsembleHandler] Received - Key: {key}, Address: {address}, Body: {body}")
        try:
            if body:
                # Handle body as dict or JSON string
                data = body if isinstance(body, dict) else json.loads(body)
                self.ensembler.on_ensemble(data)
            else:
                logging.warning("Received an empty message body on ensemble topic.")
        except Exception as e:
            logging.error(f"Error in EnsembleHandler: {e}")


class Bootstrap(ConnectorHandler):
    """
    ConnectorHandler subclass to manage consumer registration upon readiness.
    This is where we register two separate consumers.
    """
    def __init__(self, ensembler, config):
        super().__init__()
        self.ensembler = ensembler
        self.config = config

    def ready(self, context: Context):
        logging.info("Connector is ready. Registering consumers.")

        # Create two distinct handlers
        start_ensembling_handler = StartEnsemblingHandler(self.ensembler)
        ensemble_handler = EnsembleHandler(self.ensembler)

        # Register two consumers, each with a different topic
        context.register_consumers(
            Consumer(
                address=self.config['START_TOPIC'],   # e.g., "start.ensembler"
                handler=start_ensembling_handler,
                topic=True,
                fqdn=True,
                key="start_ensembling_consumer"
            ))

        context.register_consumers(
                Consumer(
                address=self.config['ENSEMBLE_TOPIC'],
                handler=ensemble_handler,
                topic=True,
                fqdn=True,
                key="ensembling_consumer"
            )
        )

        logging.info("Consumers registered successfully.")


class EXNService:
    """
    Class for starting the EXN service in a separate thread, connecting to your AMQ/EXN broker,
    and subscribing to the two separate topics for ensembling.
    """
    def __init__(self, config, ensembler):
        """
        :param config: Dictionary containing configuration parameters, e.g.:
                       {
                         'AMQ_HOST': 'amqp://mybroker',
                         'AMQ_PORT_BROKER': 5672,
                         'AMQ_USER': 'user',
                         'AMQ_PASSWORD': 'pass',
                         'START_TOPIC': 'start.ensembler',
                         'ENSEMBLE_TOPIC': 'ensemble.ensembler',
                         ...
                       }
        :param ensembler: Instance of a class that implements
                          on_start_ensembler(body) and on_ensemble(body)
        """
        self.config = config
        self.ensembler = ensembler
        self.connector = None
        self.thread = None
        self.stop_event = threading.Event()

    def start_exn_service(self):
        """
        Connect to the EXN broker and initialize the Connector with
        two consumers and one publisher.
        """
        logging.info(f"Initializing EXN with component='ensembler_service', url={self.config['AMQ_HOST']}")
        logging.info(f"Handler: {Bootstrap(self.ensembler, self.config)}, Publishers: {[publisher]}")

        # Initialize the EXN connector
        self.connector = EXN(
            component="ensembler_service",
            handler=Bootstrap(self.ensembler, self.config),
            publishers=[publisher],
            consumers=[],   # Consumers are registered inside Bootstrap.ready()
            enable_health=True,
            enable_state=True,
            url=self.config["AMQ_HOST"],
            port=self.config["AMQ_PORT_BROKER"],
            username=self.config["AMQ_USER"],
            password=self.config["AMQ_PASSWORD"]
        )

        logging.info("Starting EXN connector...")
        self.connector.start()
        logging.info("EXN connector started successfully.")

    def run(self):
        """Run the EXN service in a separate thread."""
        logging.info("EXNService thread starting.")
        self.start_exn_service()
        while not self.stop_event.is_set():
            time.sleep(1)  # Keep thread alive
        logging.info("EXNService thread stopping.")
        self.connector.stop()  # Stop the connector gracefully

    def start(self):
        """Start the EXN service in a new daemon thread."""
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logging.info("EXNService thread started.")

    def stop(self):
        """Signal the EXN service to stop and wait for the thread to finish."""
        logging.info("Stopping EXNService...")
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        logging.info("EXNService stopped successfully.")