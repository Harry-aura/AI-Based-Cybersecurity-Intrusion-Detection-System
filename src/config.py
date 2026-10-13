from pathlib import Path
from typing import Dict, List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=()
    )

    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    data_processed_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    model_artifacts_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models" / "artifacts")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    nsl_kdd_train_url: str = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    nsl_kdd_test_url: str = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
    nsl_kdd_names_url: str = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDD%20Names.txt"

    column_names: List[str] = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]

    categorical_features: List[str] = ["protocol_type", "service", "flag"]
    numerical_features: List[str] = [
        "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]

    target_column: str = "label"
    difficulty_column: str = "difficulty"

    attack_categories: Dict[str, List[str]] = {
        "Normal": ["normal"],
        "DoS": [
            "back", "land", "neptune", "pod", "smurf", "teardrop",
            "apache2", "udpstorm", "processtable", "worm", "mailbomb"
        ],
        "Probe": [
            "satan", "ipsweep", "nmap", "portsweep",
            "mscan", "saint"
        ],
        "R2L": [
            "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
            "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
            "httptunnel", "sendmail", "named"
        ],
        "U2R": [
            "buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
            "xterm", "ps", "httptunnel"
        ]
    }

    label_map_binary: Dict[str, int] = {
        "Normal": 0, "Attack": 1,
        "DoS": 1, "Probe": 1, "R2L": 1, "U2R": 1,
        "normal": 0, "attack": 1,
        "dos": 1, "probe": 1, "r2l": 1, "u2r": 1
    }
    label_map_multiclass: Dict[str, int] = {
        "Normal": 0, "DoS": 1, "Probe": 2, "R2L": 3, "U2R": 4,
        "normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4
    }

    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42
    cv_folds: int = 5

    lr_l1_params: Dict = {"C": 1.0, "penalty": "l1", "solver": "saga", "max_iter": 1000, "random_state": 42}
    lr_l2_params: Dict = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "random_state": 42}

    rf_params: Dict = {
        "n_estimators": 200, "max_depth": 20, "min_samples_split": 5,
        "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42, "n_jobs": -1
    }
    xgb_params_binary: Dict = {
        "n_estimators": 200, "max_depth": 8, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
        "n_jobs": -1, "eval_metric": "logloss", "objective": "binary:logistic"
    }
    xgb_params_multiclass: Dict = {
        "n_estimators": 300, "max_depth": 10, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
        "n_jobs": -1, "eval_metric": "mlogloss", "objective": "multi:softprob",
        "num_class": 5
    }

    pca_components: int = 30
    kmeans_clusters: int = 10
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5

    drift_ks_alpha: float = 0.05
    drift_features: List[str] = ["duration", "src_bytes", "dst_bytes", "count"]

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4


settings = Settings()

ATTACK_TO_CATEGORY: Dict[str, str] = {}
for category, attacks in settings.attack_categories.items():
    for attack in attacks:
        ATTACK_TO_CATEGORY[attack.lower()] = category


def get_category_from_label(label: str) -> str:
    cleaned = str(label).strip().lower()
    return ATTACK_TO_CATEGORY.get(cleaned, "Normal")


def map_label_to_binary(label: str) -> int:
    category = get_category_from_label(label)
    return 0 if category == "Normal" else 1


def map_label_to_multiclass(label: str) -> int:
    category = get_category_from_label(label)
    return settings.label_map_multiclass.get(category, 0)