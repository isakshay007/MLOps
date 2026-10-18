from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field

from data import CLASS_NAMES, FEATURE_NAMES
from predict import load_metrics, load_model, predict_data, MODEL_PATH


app = FastAPI(
    title="Wine Cultivar Classifier API",
    description="Predicts which of 3 cultivars a wine comes from, using 13 chemical measurements "
                "and a Random Forest trained on the scikit-learn Wine dataset.",
    version="1.0.0",
)

# A real sample from the dataset (cultivar_1), used as the example in /docs
EXAMPLE_WINE = {
    "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
    "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
    "hue": 1.04, "od280_od315_of_diluted_wines": 3.92, "proline": 1065.0,
}


class WineData(BaseModel):
    """
    Pydantic BaseModel representing the 13 chemical measurements of a wine sample.

    Every value must be positive. Upper limits are about 1.5x the largest value in the
    dataset, so typos like 1400 instead of 14.0 are rejected with a 422 error.
    """
    alcohol: float = Field(..., gt=0, le=20, description="Alcohol (% vol)")
    malic_acid: float = Field(..., gt=0, le=10, description="Malic acid (g/L)")
    ash: float = Field(..., gt=0, le=5, description="Ash (g/L)")
    alcalinity_of_ash: float = Field(..., gt=0, le=45, description="Alcalinity of ash")
    magnesium: float = Field(..., gt=0, le=250, description="Magnesium (mg/L)")
    total_phenols: float = Field(..., gt=0, le=6, description="Total phenols")
    flavanoids: float = Field(..., gt=0, le=8, description="Flavanoids")
    nonflavanoid_phenols: float = Field(..., gt=0, le=1, description="Nonflavanoid phenols")
    proanthocyanins: float = Field(..., gt=0, le=6, description="Proanthocyanins")
    color_intensity: float = Field(..., gt=0, le=20, description="Color intensity")
    hue: float = Field(..., gt=0, le=3, description="Hue")
    od280_od315_of_diluted_wines: float = Field(..., gt=0, le=6, description="OD280/OD315 of diluted wines")
    proline: float = Field(..., gt=0, le=2500, description="Proline (mg/L)")

    model_config = {"json_schema_extra": {"examples": [EXAMPLE_WINE]}}

    def to_row(self):
        """Return the features in the same column order the model was trained on."""
        return [getattr(self, name) for name in FEATURE_NAMES]


class WineBatch(BaseModel):
    samples: list[WineData] = Field(..., min_length=1, max_length=100)


class WineResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]


class WineBatchResponse(BaseModel):
    count: int
    predictions: list[WineResponse]


def run_prediction(rows):
    """Run the model and turn failures into clear HTTP errors."""
    try:
        return predict_data(rows)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Health check that also reports whether the trained model file is present."""
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
async def model_info():
    """
    Describe the deployed model: type, hyperparameters, expected features, classes,
    and the accuracy / F1 / confusion matrix measured on the held-out test set.
    """
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    return {
        "model_type": type(model).__name__,
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
        "metrics": load_metrics(),
    }


@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine: WineData):
    """
    Predict the cultivar of one wine sample.
    Returns the predicted class id and name, the model's confidence,
    and the probability of each of the 3 cultivars.
    """
    return run_prediction([wine.to_row()])[0]


@app.post("/predict/batch", response_model=WineBatchResponse)
async def predict_wine_batch(batch: WineBatch):
    """Predict the cultivar of 1 to 100 wine samples in a single request."""
    predictions = run_prediction([wine.to_row() for wine in batch.samples])
    return {"count": len(predictions), "predictions": predictions}
