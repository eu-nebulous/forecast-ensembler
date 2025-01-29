"""Script for defining objects in pydantic. Untrusted data can
be passed to a model, and after parsing and validation pydantic
guarantees that the fields of the resultant model instance will
conform to the field types defined on the model. """
from typing import Dict, Optional, Any

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
    ] #method_name exmp lstm and value
    app_id: str



class EnsembleResponse(BaseModel):
    """
    Pydantic response model with these required fields:
        - status: e.g. "success"
        - data:   any dict with additional info
        - ensembledValue: final numeric prediction
        - timestamp: creation time in seconds
        - predictionTime: original prediction time (from request)"""
    status: str
    data: Optional[Dict[str, Any]] = None
    ensembledValue: float
    timestamp: int
    predictionTime: int

# pylint: enable=too-few-public-methods
