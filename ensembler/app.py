"""main script for ensembler module"""

import logging
from datetime import datetime

import setproctitle
from fastapi import FastAPI
from pytz import timezone

from ensembler.env_config import create_env_config
from ensembler.messages_schemas import EnsembleResponse, Prediction
from ensembler.services import AMQService

env_config = create_env_config()
setproctitle.setproctitle("Ensembler")
logging.basicConfig(
    filename=f"logs/{env_config['LOGING_FILE_NAME']}.out",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
)
logging.Formatter.converter = lambda *args: datetime.now(
    tz=timezone(env_config["TZ"])
).timetuple()

amq_service = AMQService(env_config)

app = FastAPI()


@app.post("/ensemble", response_model=EnsembleResponse)
async def add_country(prediction: Prediction):
    """Function for returning ensembled value on request"""
    return amq_service.ens.on_ensemble(prediction.dict())


log = logging.getLogger()
log.info(f"Ensebler service started for application: {env_config['APP_NAME']}")
