"""Main script for Ensembler module"""

import logging
from datetime import datetime
import setproctitle
from fastapi import FastAPI, HTTPException
from grpc import services
from pytz import timezone
import asyncio
import os
import uvicorn

from ensembler.ensembler import Ensembler
from ensembler.services import EXNService, publisher
from ensembler.env_config import create_env_config
from ensembler.messages_schemas import EnsembleResponse, Prediction

# Initialize environment configuration
env_config = create_env_config()

# Set the process title for easier identification
setproctitle.setproctitle("Ensembler")

# Ensure the logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, f"{env_config['LOGING_FILE_NAME']}.out"),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
)

# Adjust the logging time converter to use the specified timezone
logging.Formatter.converter = lambda *args: datetime.now(
    tz=timezone(env_config["TZ"])
).timetuple()

# Create a logger instance
logger = logging.getLogger(__name__)
logger.info(f"Ensembler service starting for application: {env_config['APP_NAME']}")

app = FastAPI(title="Ensembler Service", debug=True)

# Initialize the Ensembler and EXNService instances
def initialize_services():
    """
    Initialize and return the Ensembler and EXNService instances.
    """

    # Initialize the Ensembler
    ensembler = Ensembler(config=env_config)

    # Initialize EXNService with the configuration and ensembler
    exn_service = EXNService(config=env_config, ensembler=ensembler)

    return ensembler, exn_service

# Store global instances
ensembler_instance, exn_service_instance = initialize_services()

@app.on_event("startup")
def startup_event():
    """Event handler for FastAPI startup."""
    logger.info("Starting EXNService...")
    exn_service_instance.start()
    logger.info("EXNService started successfully.")

@app.on_event("shutdown")
def shutdown_event():
    """Event handler for FastAPI shutdown."""
    logger.info("Shutting down EXNService...")
    exn_service_instance.stop()
    logger.info("EXNService shut down successfully.")


@app.post("/ensemble", response_model=EnsembleResponse)
async def ensemble(prediction: Prediction):
    """
    Endpoint to process an ensemble prediction.

    :param prediction: Prediction data received from the client.
    :return: EnsembleResponse containing the processed result.
    """
    try:
        # Process the prediction using Ensembler's on_ensemble method
        loop = asyncio.get_event_loop()
        # on_ensemble(...) already returns an EnsembleResponse
        result: EnsembleResponse = await loop.run_in_executor(
            None,
            ensembler_instance.on_ensemble,
            {**prediction.dict()}
        )
        # Simply return the same result object
        return result

    except Exception as e:
        logging.error(f"Error processing ensemble: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    # Log that the service is starting
    logger.info("Ensembler service is starting...")

    # Run the FastAPI app with Uvicorn
    uvicorn.run(
        "app:app",  # Ensure this matches the filename if different
        host=env_config.get("HOST", "0.0.0.0"),
        port=int(env_config.get("PORT", 8000)),
        log_level="info",
        reload=False  # Set to True for development with auto-reload
    )