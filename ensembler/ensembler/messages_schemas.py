"""Script for defining objects in pydantic. Untrusted data can
be passed to a model, and after parsing and validation pydantic
guarantees that the fields of the resultant model instance will
conform to the field types defined on the model. """
from typing import Dict, Optional

# pylint: disable=no-name-in-module
from pydantic import BaseModel

# pylint: enable=no-name-in-module

# pylint: disable=too-few-public-methods
class Prediction(BaseModel):
    """Ensemble request message body
    method: string, requested ensembling method e.g linnear_programming
    metric: string, predicted metric
    predictiontime: int, prediction time in seconds
    values: map with fields : forecaster (string): prediction (float value or None)"""

    method: str
    metric: str
    predictionTime: int
    predictionsToEnsemble: Dict[
        str,
        Optional[float],
    ]


class EnsembleResponse(BaseModel):
    """Ensemble response message body
    metricValue: float, ensembled prediction
    timestamp: int, ensembled prediction creation time in seconds
    predictiontime: int, prediction time in seconds"""

    ensembledValue: float
    timestamp: int
    predictionTime: int


# pylint: enable=too-few-public-methods
