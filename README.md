# CyberGuard-IDS: AI-Based Network Intrusion Detection System

A production-ready Network Intrusion Detection System (NIDS) built with Python, scikit-learn, XGBoost, and FastAPI. Implements the complete M1-M6 syllabus for ML-based cybersecurity detection.

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Ingestion │────▶│ Feature Pipeline │────▶│  Model Training │
│  (NSL-KDD)      │     │ (Preprocessing)  │     │  (M4-M5)        │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                    ┌──────────────────┐                  │
                    │ Unsupervised     │◀─────────────────┘
                    │ (PCA/K-Means/    │
                    │  DBSCAN)         │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Model Serving    │
                    │ (FastAPI)        │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌────────────┐
        │ /predict │  │ /batch   │  │ /drift-check│
        └──────────┘  └──────────┘  └────────────┘
```

## M1-M6 Syllabus Mapping

| Module | Topic | Implementation |
|--------|-------|----------------|
| **M1** | Data Ingestion & EDA | `src/data_ingestion.py` - NSL-KDD download, cleaning, labeling, train/val/test split |
| **M2** | Feature Engineering | `src/feature_pipeline.py` - ColumnTransformer, OneHotEncoder, StandardScaler, SelectKBest, DriftDetector |
| **M3** | Unsupervised Learning | `src/unsupervised.py` - PCA, K-Means clustering, DBSCAN anomaly detection |
| **M4** | Linear Models | `src/models/linear.py` - Logistic Regression (L1/L2), GridSearchCV, feature importance |
| **M5** | Tree-Based Models | `src/models/tree_based.py` - Random Forest, XGBoost (binary/multiclass), permutation importance |
| **M6** | Evaluation & Serving | `src/evaluation.py` - Binary/multiclass metrics, McNemar's test; `src/serving/app.py` - FastAPI REST API |

## Quick Start

### Prerequisites
- Python 3.11+
- 4GB+ RAM recommended
- Internet access (for NSL-KDD download on first run)

### Installation

```bash
# Clone repository
cd CyberGuard-IDS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Training the Models

```bash
# Full training pipeline (downloads NSL-KDD, trains all models, evaluates)
python src/train.py

# With hyperparameter tuning (slower but better performance)
python -c "from src.train import run_full_training; run_full_training(tune_hyperparameters=True)"
```

Training artifacts saved to `models/artifacts/`:
- `best_binary_model.joblib` - Best binary classifier
- `best_multiclass_model.joblib` - Best multi-class classifier
- `preprocessing_pipeline.joblib` - Fitted feature pipeline
- `unsupervised_models.joblib` - PCA, K-Means, DBSCAN models
- `model_metadata.json` - Training metadata
- `evaluation_results_*.json` - Detailed metrics
- `mcnemar_test.json` - Statistical significance test

### Starting the API Server

```bash
# Development mode with auto-reload
uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

# Production mode (4 workers)
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 4
```

API available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health**: http://localhost:8000/health

### Docker Deployment

```bash
# Build image
docker build -t cyberguard-ids .

# Run container
docker run -d -p 8000:8000 --name cyberguard cyberguard-ids

# Check logs
docker logs cyberguard
```

## API Reference

### Health Check
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2024-01-15T10:30:00",
  "uptime_seconds": 123.45
}
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 10,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 1000,
    "dst_bytes": 2000,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 5,
    "srv_count": 3,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 10,
    "dst_host_srv_count": 5,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
  }'
```
**Response:**
```json
{
  "binary": {
    "label": "Normal",
    "probability": 0.0234,
    "confidence": 0.9766
  },
  "multiclass": {
    "label": "Normal",
    "probabilities": {
      "Normal": 0.95,
      "DoS": 0.02,
      "Probe": 0.01,
      "R2L": 0.01,
      "U2R": 0.01
    },
    "confidence": 0.95
  },
  "cluster_id": 3,
  "anomaly_score": 0.0
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "packets": [
      {"duration": 10, "protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 1000, "dst_bytes": 2000, "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0, "count": 5, "srv_count": 3, "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 10, "dst_host_srv_count": 5, "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0},
      {"duration": 0, "protocol_type": "tcp", "service": "private", "flag": "REJ", "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 1, "num_failed_logins": 4, "logged_in": 0, "num_compromised": 1, "root_shell": 0, "su_attempted": 1, "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0, "count": 10, "srv_count": 10, "serror_rate": 1.0, "srv_serror_rate": 1.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 0.0, "diff_srv_rate": 1.0, "srv_diff_host_rate": 1.0, "dst_host_count": 255, "dst_host_srv_count": 255, "dst_host_same_srv_rate": 0.0, "dst_host_diff_srv_rate": 1.0, "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 1.0, "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0}
    ]
  }'
```

### Drift Detection
```bash
curl -X POST http://localhost:8000/drift-check \
  -H "Content-Type: application/json" \
  -d '{
    "reference_data": [
      {"duration": 10, "src_bytes": 1000, "dst_bytes": 2000, "count": 5},
      {"duration": 20, "src_bytes": 1500, "dst_bytes": 2500, "count": 8},
      {"duration": 15, "src_bytes": 1200, "dst_bytes": 2200, "count": 6}
    ],
    "current_data": [
      {"duration": 500, "src_bytes": 50000, "dst_bytes": 10000, "count": 50},
      {"duration": 600, "src_bytes": 60000, "dst_bytes": 12000, "count": 60}
    ],
    "features": ["duration", "src_bytes", "dst_bytes", "count"],
    "alpha": 0.05
  }'
```
**Response:**
```json
{
  "drift_detected": true,
  "p_values": {
    "duration": 0.0012,
    "src_bytes": 0.0003,
    "dst_bytes": 0.0456,
    "count": 0.0001
  },
  "drifted_features": ["duration", "src_bytes", "dst_bytes", "count"],
  "alpha": 0.05
}
```

### Model Metrics
```bash
curl http://localhost:8000/metrics
```

### Model Info
```bash
curl http://localhost:8000/model/info
```

### Reload Models (Hot Reload)
```bash
curl -X POST http://localhost:8000/reload
```

## Project Structure

```
CyberGuard-IDS/
├── data/
│   ├── raw/           # Downloaded NSL-KDD files
│   └── processed/     # Parquet train/val/test splits
├── models/
│   └── artifacts/     # Trained model artifacts
├── src/
│   ├── config.py              # Settings & label mappings
│   ├── schemas.py             # Pydantic request/response models
│   ├── data_ingestion.py      # NSL-KDD download, clean, split
│   ├── feature_pipeline.py    # Preprocessing + feature selection
│   ├── unsupervised.py        # PCA, K-Means, DBSCAN
│   ├── evaluation.py          # Metrics, McNemar's test
│   ├── train.py               # Orchestrated training pipeline
│   ├── models/
│   │   ├── linear.py          # Logistic Regression (L1/L2)
│   │   └── tree_based.py      # Random Forest, XGBoost
│   └── serving/
│       └── app.py             # FastAPI application
├── tests/
│   ├── test_pipeline.py       # Unit tests for ML pipeline
│   └── test_api.py            # API integration tests
├── requirements.txt
├── Dockerfile
└── README.md
```

## Configuration

Key settings in `src/config.py`:
- `pca_components`: PCA dimensions (default: 30)
- `kmeans_clusters`: K-Means clusters (default: 10)
- `dbscan_eps` / `dbscan_min_samples`: DBSCAN params
- Model hyperparameters: `rf_params`, `xgb_params_binary`, `xgb_params_multiclass`
- Data splits: `train_split=0.7`, `val_split=0.15`, `test_split=0.15`

Environment variables (`.env` file):
```bash
MODEL_ARTIFACTS_DIR=/custom/path/models
DATA_PROCESSED_DIR=/custom/path/data
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_pipeline.py -v
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance Benchmarks (NSL-KDD)

| Model | Binary F1 | Multiclass F1 (Weighted) |
|-------|-----------|-------------------------|
| Logistic L1 | ~0.92 | ~0.78 |
| Logistic L2 | ~0.93 | ~0.79 |
| Random Forest | ~0.96 | ~0.85 |
| XGBoost | ~0.97 | ~0.87 |

*Results may vary based on data split and hyperparameters.*

## Security Considerations

- Input validation via Pydantic schemas (all 41 features validated)
- Rate limiting recommended for production (add `slowapi` middleware)
- HTTPS termination via reverse proxy (nginx/Traefik)
- Model artifacts should be integrity-checked in production
- Drift detection endpoint monitors feature distribution shifts

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Citation

If you use this system in research, please cite:
```bibtex
@software{cyberguard-ids,
  title = {CyberGuard-IDS: AI-Based Network Intrusion Detection System},
  author = {Harivikash katta},
  year = {2024},
  url = {https://github.com/your-repo/CyberGuard-IDS}
}
```