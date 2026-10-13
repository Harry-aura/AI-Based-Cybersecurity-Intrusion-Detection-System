import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from src.serving.app import app, model_registry
from src.schemas import (
    NetworkPacket, BatchNetworkPackets, DriftCheckRequest
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_packet():
    return {
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
    }


@pytest.fixture
def mock_models():
    binary_model = MagicMock()
    binary_model.predict.return_value = np.array([0])
    binary_model.predict_proba.return_value = np.array([[0.9, 0.1]])

    multiclass_model = MagicMock()
    multiclass_model.predict.return_value = np.array([0])
    multiclass_model.predict_proba.return_value = np.array([[0.8, 0.05, 0.05, 0.05, 0.05]])

    preprocessing = MagicMock()
    preprocessing.named_steps = {
        "preprocessing": MagicMock()
    }
    preprocessing.named_steps["preprocessing"].transform.return_value = np.array([[1.0] * 30])

    unsupervised = {
        "pca": MagicMock(),
        "kmeans": MagicMock(),
        "dbscan": MagicMock()
    }
    unsupervised["pca"].transform.return_value = np.array([[0.5] * 10])
    unsupervised["kmeans"].predict.return_value = np.array([3])
    unsupervised["dbscan"].predict.return_value = np.array([0])

    return binary_model, multiclass_model, preprocessing, unsupervised


class TestHealthEndpoint:
    def test_health_check_not_loaded(self, client):
        model_registry.loaded = False
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False

    def test_health_check_loaded(self, client, mock_models):
        binary_model, multiclass_model, preprocessing, unsupervised = mock_models
        model_registry.loaded = True
        model_registry.binary_model = binary_model
        model_registry.multiclass_model = multiclass_model
        model_registry.preprocessing_pipeline = preprocessing
        model_registry.unsupervised_models = unsupervised
        model_registry.model_metadata = {"training_date": "2024-01-01"}

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_predict_not_loaded(self, client, sample_packet):
        model_registry.loaded = False
        response = client.post("/predict", json=sample_packet)
        assert response.status_code == 503

    def test_predict_success(self, client, sample_packet, mock_models):
        binary_model, multiclass_model, preprocessing, unsupervised = mock_models
        model_registry.loaded = True
        model_registry.binary_model = binary_model
        model_registry.multiclass_model = multiclass_model
        model_registry.preprocessing_pipeline = preprocessing
        model_registry.unsupervised_models = unsupervised

        response = client.post("/predict", json=sample_packet)
        assert response.status_code == 200
        data = response.json()
        assert "binary" in data
        assert "multiclass" in data
        assert data["binary"]["label"] in ["Normal", "Attack"]
        assert data["multiclass"]["label"] in ["Normal", "DoS", "Probe", "R2L", "U2R"]
        assert 0 <= data["binary"]["probability"] <= 1
        assert 0 <= data["binary"]["confidence"] <= 1


class TestBatchEndpoint:
    def test_batch_predict(self, client, sample_packet, mock_models):
        binary_model, multiclass_model, preprocessing, unsupervised = mock_models
        model_registry.loaded = True
        model_registry.binary_model = binary_model
        model_registry.multiclass_model = multiclass_model
        model_registry.preprocessing_pipeline = preprocessing
        model_registry.unsupervised_models = unsupervised

        batch_request = {"packets": [sample_packet, sample_packet]}
        response = client.post("/batch", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 2


class TestDriftCheckEndpoint:
    def test_drift_check_no_reference(self, client):
        model_registry.loaded = True
        model_registry.reference_data = None

        request = {
            "reference_data": [{"duration": 10, "src_bytes": 1000}],
            "current_data": [{"duration": 20, "src_bytes": 2000}],
            "features": ["duration", "src_bytes"],
            "alpha": 0.05
        }
        response = client.post("/drift-check", json=request)
        assert response.status_code == 503

    def test_drift_check_success(self, client):
        model_registry.loaded = True
        model_registry.reference_data = pd.DataFrame({
            "duration": [10, 20, 30, 40, 50],
            "src_bytes": [1000, 2000, 3000, 4000, 5000]
        })

        request = {
            "reference_data": [{"duration": 10, "src_bytes": 1000}],
            "current_data": [{"duration": 20, "src_bytes": 2000}],
            "features": ["duration", "src_bytes"],
            "alpha": 0.05
        }
        response = client.post("/drift-check", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "p_values" in data
        assert "drifted_features" in data


class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        model_registry.loaded = True
        model_registry.model_metadata = {
            "best_binary_model": "random_forest",
            "best_multiclass_model": "xgboost"
        }

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "binary_metrics" in data
        assert "multiclass_metrics" in data
        assert "confusion_matrix_binary" in data
        assert "confusion_matrix_multiclass" in data


class TestModelInfoEndpoint:
    def test_model_info(self, client):
        model_registry.loaded = True
        model_registry.model_metadata = {"training_date": "2024-01-01T00:00:00"}

        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "version" in data
        assert "classes_binary" in data
        assert "classes_multiclass" in data
        assert data["classes_binary"] == ["Normal", "Attack"]
        assert data["classes_multiclass"] == ["Normal", "DoS", "Probe", "R2L", "U2R"]


class TestSchemaValidation:
    def test_network_packet_validation(self):
        packet = NetworkPacket(
            duration=10,
            protocol_type="tcp",
            service="http",
            flag="SF",
            src_bytes=1000,
            dst_bytes=2000,
            land=0,
            wrong_fragment=0,
            urgent=0,
            hot=0,
            num_failed_logins=0,
            logged_in=1,
            num_compromised=0,
            root_shell=0,
            su_attempted=0,
            num_root=0,
            num_file_creations=0,
            num_shells=0,
            num_access_files=0,
            num_outbound_cmds=0,
            is_host_login=0,
            is_guest_login=0,
            count=5,
            srv_count=3,
            serror_rate=0.0,
            srv_serror_rate=0.0,
            rerror_rate=0.0,
            srv_rerror_rate=0.0,
            same_srv_rate=1.0,
            diff_srv_rate=0.0,
            srv_diff_host_rate=0.0,
            dst_host_count=10,
            dst_host_srv_count=5,
            dst_host_same_srv_rate=1.0,
            dst_host_diff_srv_rate=0.0,
            dst_host_same_src_port_rate=0.0,
            dst_host_srv_diff_host_rate=0.0,
            dst_host_serror_rate=0.0,
            dst_host_srv_serror_rate=0.0,
            dst_host_rerror_rate=0.0,
            dst_host_srv_rerror_rate=0.0
        )
        assert packet.duration == 10
        assert packet.protocol_type.value == "tcp"

    def test_network_packet_invalid_protocol(self):
        with pytest.raises(Exception):
            NetworkPacket(
                duration=10,
                protocol_type="invalid",
                service="http",
                flag="SF",
                src_bytes=1000,
                dst_bytes=2000,
                land=0,
                wrong_fragment=0,
                urgent=0,
                hot=0,
                num_failed_logins=0,
                logged_in=1,
                num_compromised=0,
                root_shell=0,
                su_attempted=0,
                num_root=0,
                num_file_creations=0,
                num_shells=0,
                num_access_files=0,
                num_outbound_cmds=0,
                is_host_login=0,
                is_guest_login=0,
                count=5,
                srv_count=3,
                serror_rate=0.0,
                srv_serror_rate=0.0,
                rerror_rate=0.0,
                srv_rerror_rate=0.0,
                same_srv_rate=1.0,
                diff_srv_rate=0.0,
                srv_diff_host_rate=0.0,
                dst_host_count=10,
                dst_host_srv_count=5,
                dst_host_same_srv_rate=1.0,
                dst_host_diff_srv_rate=0.0,
                dst_host_same_src_port_rate=0.0,
                dst_host_srv_diff_host_rate=0.0,
                dst_host_serror_rate=0.0,
                dst_host_srv_serror_rate=0.0,
                dst_host_rerror_rate=0.0,
                dst_host_srv_rerror_rate=0.0
            )

    def test_batch_network_packets_validation(self, sample_packet):
        batch = BatchNetworkPackets(packets=[sample_packet, sample_packet])
        assert len(batch.packets) == 2

    def test_batch_network_packets_empty(self):
        with pytest.raises(Exception):
            BatchNetworkPackets(packets=[])

    def test_batch_network_packets_too_many(self, sample_packet):
        with pytest.raises(Exception):
            BatchNetworkPackets(packets=[sample_packet] * 10001)


class TestDriftCheckSchema:
    def test_drift_check_request(self):
        request = DriftCheckRequest(
            reference_data=[{"duration": 10, "src_bytes": 1000}],
            current_data=[{"duration": 20, "src_bytes": 2000}],
            features=["duration", "src_bytes"],
            alpha=0.05
        )
        assert request.alpha == 0.05
        assert request.features == ["duration", "src_bytes"]

    def test_drift_check_request_invalid_alpha(self):
        with pytest.raises(Exception):
            DriftCheckRequest(
                reference_data=[{"duration": 10}],
                current_data=[{"duration": 20}],
                alpha=1.5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])