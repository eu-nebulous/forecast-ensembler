"""Script for reading enviromental variables"""

import os


def create_env_config():
    """return dict with enviromental variables,"""
    config = {}
    config["APP_NAME"] = os.environ.get("APP_NAME", "demo")
    config["AMQ_USER"] = os.environ.get("AMQ_USER", "admin")
    config["AMQ_PASSWORD"] = os.environ.get("AMQ_PASSWORD", "admin")
    config["AMQ_HOST"] = os.environ.get("AMQ_HOST", "localhost")
    config["AMQ_PORT_BROKER"] = os.environ.get("AMQ_PORT_BROKER", "61613")
    config["START_TOPIC"] = "start_ensembler"
    config["TZ"] = os.environ.get("TIME_ZONE", "Europe/Vienna")
    config["LOGING_FILE_NAME"] = os.environ.get("LOGING_FILE_NAME", "ensembler")
    return config
