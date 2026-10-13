import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError
from scipy import stats

from src.config import settings
from src.schemas import (
    BatchNetworkPackets,
    BatchPredictionResponse,
    BinaryPrediction,
    DriftCheckRequest,
    DriftCheckResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    MulticlassPrediction,
    NetworkPacket,
    SinglePredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "models" / "artifacts"


class ModelRegistry:
    def __init__(self):
        self.binary_model = None
        self.multiclass_model = None
        self.preprocessing_pipeline = None
        self.unsupervised_models = None
        self.reference_data = None
        self.model_metadata = {}
        self.start_time = time.time()
        self.loaded = False

    def load_models(self) -> None:
        try:
            binary_path = MODEL_ARTIFACTS_DIR / "best_binary_model.joblib"
            multiclass_path = MODEL_ARTIFACTS_DIR / "best_multiclass_model.joblib"
            preprocess_path = MODEL_ARTIFACTS_DIR / "preprocessing_pipeline.joblib"
            unsupervised_path = MODEL_ARTIFACTS_DIR / "unsupervised_models.joblib"
            metadata_path = MODEL_ARTIFACTS_DIR / "model_metadata.json"

            if binary_path.exists():
                self.binary_model = joblib.load(binary_path)
                logger.info("Binary model loaded")
            else:
                logger.warning("Binary model not found")

            if multiclass_path.exists():
                self.multiclass_model = joblib.load(multiclass_path)
                logger.info("Multiclass model loaded")
            else:
                logger.warning("Multiclass model not found")

            if preprocess_path.exists():
                self.preprocessing_pipeline = joblib.load(preprocess_path)
                logger.info("Preprocessing pipeline loaded")
            else:
                logger.warning("Preprocessing pipeline not found")

            if unsupervised_path.exists():
                self.unsupervised_models = joblib.load(unsupervised_path)
                logger.info("Unsupervised models loaded")
            else:
                logger.warning("Unsupervised models not found")

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.model_metadata = json.load(f)
                logger.info("Model metadata loaded")

            self.loaded = (
                self.binary_model is not None and self.multiclass_model is not None
            )

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def predict_single(self, packet: NetworkPacket) -> SinglePredictionResponse:
        if not self.loaded or self.binary_model is None or self.multiclass_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")

        df = pd.DataFrame([packet.model_dump()])

        # If model expects preprocessed features directly or if it's a full pipeline
        try:
            if hasattr(self.binary_model, "predict_proba"):
                binary_proba = self.binary_model.predict_proba(df)[0]
                binary_pred = self.binary_model.predict(df)[0]
            else:
                X_proc = self.preprocessing_pipeline.transform(df)
                binary_proba = self.binary_model.predict_proba(X_proc)[0]
                binary_pred = self.binary_model.predict(X_proc)[0]
        except Exception:
            X_proc = self.preprocessing_pipeline.transform(df)
            binary_proba = self.binary_model.predict_proba(X_proc)[0]
            binary_pred = self.binary_model.predict(X_proc)[0]

        binary_label = "Attack" if binary_pred == 1 else "Normal"
        binary_prob = float(binary_proba[1]) if len(binary_proba) > 1 else float(binary_proba[0])

        try:
            if hasattr(self.multiclass_model, "predict_proba"):
                multi_proba = self.multiclass_model.predict_proba(df)[0]
                multi_pred = self.multiclass_model.predict(df)[0]
            else:
                X_proc = self.preprocessing_pipeline.transform(df)
                multi_proba = self.multiclass_model.predict_proba(X_proc)[0]
                multi_pred = self.multiclass_model.predict(X_proc)[0]
        except Exception:
            X_proc = self.preprocessing_pipeline.transform(df)
            multi_proba = self.multiclass_model.predict_proba(X_proc)[0]
            multi_pred = self.multiclass_model.predict(X_proc)[0]

        multi_classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        multi_label = multi_classes[int(multi_pred)] if int(multi_pred) < len(multi_classes) else "Normal"
        multi_probs = {cls: float(prob) for cls, prob in zip(multi_classes, multi_proba)}

        return SinglePredictionResponse(
            binary=BinaryPrediction(
                label=binary_label,
                probability=binary_prob,
                confidence=max(binary_prob, 1.0 - binary_prob),
            ),
            multiclass=MulticlassPrediction(
                label=multi_label,
                probabilities=multi_probs,
                confidence=max(multi_probs.values()) if multi_probs else 0.0,
            ),
            cluster_id=None,
            anomaly_score=None,
        )

    def predict_batch(self, packets: List[NetworkPacket]) -> BatchPredictionResponse:
        predictions = [self.predict_single(p) for p in packets]

        summary = {"Normal": 0, "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0, "Attack": 0, "Benign": 0}
        for pred in predictions:
            summary[pred.multiclass.label] += 1
            if pred.binary.label == "Attack":
                summary["Attack"] += 1
            else:
                summary["Benign"] += 1

        return BatchPredictionResponse(predictions=predictions, summary=summary)

    def check_drift(self, request: DriftCheckRequest) -> DriftCheckResponse:
        current_df = pd.DataFrame(request.current_data)
        reference_df = pd.DataFrame(request.reference_data)
        features = request.features or list(current_df.columns)

        p_values = {}
        drifted_features = []

        for feature in features:
            if feature in current_df.columns and feature in reference_df.columns:
                try:
                    _, p_val = stats.ks_2samp(
                        reference_df[feature].dropna(),
                        current_df[feature].dropna(),
                    )
                    p_values[feature] = float(p_val)
                    if p_val < request.alpha:
                        drifted_features.append(feature)
                except Exception:
                    p_values[feature] = 1.0

        return DriftCheckResponse(
            drift_detected=len(drifted_features) > 0,
            p_values=p_values,
            drifted_features=drifted_features,
            alpha=request.alpha,
        )

    def get_health(self) -> HealthResponse:
        return HealthResponse(
            status="healthy" if self.loaded else "unhealthy",
            model_loaded=self.loaded,
            model_version=str(self.model_metadata.get("training_date", "1.0.0")),
            uptime_seconds=float(time.time() - self.start_time),
        )

    def get_metrics(self) -> MetricsResponse:
        eval_binary_path = MODEL_ARTIFACTS_DIR / "evaluation_results_binary.json"
        eval_multi_path = MODEL_ARTIFACTS_DIR / "evaluation_results_multiclass.json"

        binary_metrics = {}
        multiclass_metrics = {}
        cm_binary = [[0, 0], [0, 0]]
        cm_multi = [[0] * 5 for _ in range(5)]

        if eval_binary_path.exists():
            with open(eval_binary_path, "r") as f:
                data = json.load(f)
                best_model = self.model_metadata.get("best_binary_model", "").lower().replace(" ", "_")
                if best_model in data:
                    binary_metrics = data[best_model].get("test", {})
                    cm_binary = data[best_model].get("test", {}).get("confusion_matrix", [[0, 0], [0, 0]])

        if eval_multi_path.exists():
            with open(eval_multi_path, "r") as f:
                data = json.load(f)
                best_model = self.model_metadata.get("best_multiclass_model", "").lower().replace(" ", "_")
                if best_model in data:
                    multiclass_metrics = data[best_model].get("test", {})
                    cm_multi = data[best_model].get("test", {}).get("confusion_matrix", [[0] * 5 for _ in range(5)])

        return MetricsResponse(
            binary_metrics=binary_metrics,
            multiclass_metrics=multiclass_metrics,
            confusion_matrix_binary=cm_binary,
            confusion_matrix_multiclass=cm_multi,
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_type="Ensemble (Random Forest + XGBoost)",
            version=str(self.model_metadata.get("training_date", "1.0.0")),
            training_date=str(self.model_metadata.get("training_date", "unknown")),
            hyperparameters={},
            feature_count=len(settings.categorical_features) + len(settings.numerical_features),
            classes_binary=["Normal", "Attack"],
            classes_multiclass=["Normal", "DoS", "Probe", "R2L", "U2R"],
        )


model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    try:
        model_registry.load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="CyberGuard-IDS API",
    description="AI-Based Network Intrusion Detection System",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            status_code=422,
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return model_registry.get_health()


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    return model_registry.get_model_info()


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    return model_registry.get_metrics()


@app.post("/predict", response_model=SinglePredictionResponse)
async def predict(packet: NetworkPacket):
    return model_registry.predict_single(packet)


@app.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchNetworkPackets):
    return model_registry.predict_batch(batch.packets)


@app.post("/drift-check", response_model=DriftCheckResponse)
async def drift_check(request: DriftCheckRequest):
    return model_registry.check_drift(request)


@app.post("/reload")
async def reload_models():
    model_registry.load_models()
    return {"status": "reloaded", "loaded": model_registry.loaded}