"""Script for serivices"""
from amq_message_python_library import *  # python amq-message-python-library

from ensembler.ensembler import Ensembler


class AMQService:
    """Class for starting amq service"""

    def __init__(self, config):
        """init function"""
        self.start_amq_service(config)
        # self.start_fastapi_service()

    def start_amq_service(self, config):
        """Connect to amq, subscribe to start ensembling topic"""
        start_conn = morphemic.Connection(
            config["AMQ_USER"],
            config["AMQ_PASSWORD"],
            host=config["AMQ_HOST"],
            port=config["AMQ_PORT_BROKER"],
            debug=True,
        )
        start_conn.connect()
        start_conn.conn.subscribe(f"/topic/{config['START_TOPIC']}", "1", ack="auto")
        self.ens = Ensembler(start_conn)
        start_conn.conn.set_listener("ensemble", self.ens)
