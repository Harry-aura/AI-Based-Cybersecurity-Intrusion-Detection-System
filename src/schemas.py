from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProtocolType(str, Enum):
    tcp = "tcp"
    udp = "udp"
    icmp = "icmp"


class FlagType(str, Enum):
    SF = "SF"
    S0 = "S0"
    REJ = "REJ"
    RSTR = "RSTR"
    RSTO = "RSTO"
    SH = "SH"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    OTH = "OTH"


class ServiceType(str, Enum):
    http = "http"
    smtp = "smtp"
    finger = "finger"
    domain_u = "domain_u"
    auth = "auth"
    telnet = "telnet"
    ftp = "ftp"
    eco_i = "eco_i"
    ntp_u = "ntp_u"
    ecp = "ecp"
    tim_i = "tim_i"
    klogin = "klogin"
    kshell = "kshell"
    shell = "shell"
    login = "login"
    ftp_data = "ftp_data"
    urh_i = "urh_i"
    urp_i = "urp_i"
    nntp = "nntp"
    vmnet = "vmnet"
    syslog = "syslog"
    domain = "domain"
    other = "other"
    private = "private"
    IRC = "IRC"
    X11 = "X11"
    Z39_50 = "Z39_50"
    aol = "aol"
    http_443 = "http_443"
    exec = "exec"
    printer = "printer"
    efm = "efm"
    couplet = "couplet"
    ctf = "ctf"
    hostnames = "hostnames"
    link = "link"
    supdup = "supdup"
    iso_tsap = "iso_tsap"
    csnet_ns = "csnet_ns"
    uucp = "uucp"
    uucp_path = "uucp_path"
    netbios_ns = "netbios_ns"
    netbios_dgm = "netbios_dgm"
    netbios_ssn = "netbios_ssn"
    sql_net = "sql_net"
    gopher = "gopher"
    nnsp = "nnsp"
    pop_2 = "pop_2"
    pop_3 = "pop_3"
    sunrpc = "sunrpc"
    remote_job = "remote_job"
    rje = "rje"
    name = "name"
    whois = "whois"
    netstat = "netstat"
    time = "time"


class NetworkPacket(BaseModel):
    duration: int = Field(ge=0, description="Connection duration in seconds")
    protocol_type: ProtocolType = Field(description="Protocol type")
    service: ServiceType = Field(description="Network service")
    flag: FlagType = Field(description="Connection state flag")
    src_bytes: int = Field(ge=0, description="Source to destination bytes")
    dst_bytes: int = Field(ge=0, description="Destination to source bytes")
    land: Literal[0, 1] = Field(default=0, description="Land attack indicator")
    wrong_fragment: int = Field(default=0, ge=0, description="Wrong fragments count")
    urgent: int = Field(default=0, ge=0, description="Urgent packets count")
    hot: int = Field(default=0, ge=0, description="Hot indicator count")
    num_failed_logins: int = Field(default=0, ge=0, description="Failed login attempts")
    logged_in: Literal[0, 1] = Field(default=0, description="Successfully logged in")
    num_compromised: int = Field(default=0, ge=0, description="Compromised conditions count")
    root_shell: Literal[0, 1] = Field(default=0, description="Root shell obtained")
    su_attempted: Literal[0, 1] = Field(default=0, description="SU root attempted")
    num_root: int = Field(default=0, ge=0, description="Root access count")
    num_file_creations: int = Field(default=0, ge=0, description="File creations count")
    num_shells: int = Field(default=0, ge=0, description="Shell prompts count")
    num_access_files: int = Field(default=0, ge=0, description="Access files count")
    num_outbound_cmds: int = Field(default=0, ge=0, description="Outbound commands count")
    is_host_login: Literal[0, 1] = Field(default=0, description="Host login indicator")
    is_guest_login: Literal[0, 1] = Field(default=0, description="Guest login indicator")
    count: int = Field(ge=0, default=1, description="Connections to same host in past 2s")
    srv_count: int = Field(ge=0, default=1, description="Connections to same service in past 2s")
    serror_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="SYN error rate")
    srv_serror_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Service SYN error rate")
    rerror_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="REJ error rate")
    srv_rerror_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Service REJ error rate")
    same_srv_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Same service rate")
    diff_srv_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Different service rate")
    srv_diff_host_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Different host rate for same service")
    dst_host_count: int = Field(default=255, ge=0, description="Destination host count")
    dst_host_srv_count: int = Field(default=255, ge=0, description="Destination host service count")
    dst_host_same_srv_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Destination host same service rate")
    dst_host_diff_srv_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Destination host different service rate")
    dst_host_same_src_port_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Destination host same src port rate")
    dst_host_srv_diff_host_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Destination host service different host rate")
    dst_host_serror_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Destination host SYN error rate")
    dst_host_srv_serror_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Destination host service SYN error rate")
    dst_host_rerror_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Destination host REJ error rate")
    dst_host_srv_rerror_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Destination host service REJ error rate")

    @field_validator("protocol_type", mode="before")
    @classmethod
    def normalize_protocol(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @field_validator("service", mode="before")
    @classmethod
    def normalize_service(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) and v.lower() in [s.value for s in ServiceType] else v

    @field_validator("flag", mode="before")
    @classmethod
    def normalize_flag(cls, v: Any) -> Any:
        return v.upper() if isinstance(v, str) else v


class BatchNetworkPackets(BaseModel):
    packets: List[NetworkPacket] = Field(min_length=1, max_length=10000)


class BinaryPrediction(BaseModel):
    label: Literal["Normal", "Attack"] = Field(description="Binary classification result")
    probability: float = Field(ge=0.0, le=1.0, description="Probability of attack class")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence")


class MulticlassPrediction(BaseModel):
    label: Literal["Normal", "DoS", "Probe", "R2L", "U2R"] = Field(description="Multi-class classification result")
    probabilities: Dict[str, float] = Field(description="Class probabilities")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence")


class SinglePredictionResponse(BaseModel):
    binary: BinaryPrediction
    multiclass: MulticlassPrediction
    cluster_id: Optional[int] = Field(default=None, description="K-Means cluster assignment")
    anomaly_score: Optional[float] = Field(default=None, description="DBSCAN anomaly score")


class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePredictionResponse]
    summary: Dict[str, int] = Field(description="Summary counts by class")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class DriftCheckRequest(BaseModel):
    reference_data: List[Dict[str, float]] = Field(min_length=1, description="Reference feature distributions")
    current_data: List[Dict[str, float]] = Field(min_length=1, description="Current feature distributions")
    features: Optional[List[str]] = Field(default=None, description="Features to check for drift")
    alpha: float = Field(default=0.05, ge=0.0, le=1.0, description="Significance level")


class DriftCheckResponse(BaseModel):
    drift_detected: bool
    p_values: Dict[str, float]
    drifted_features: List[str]
    alpha: float


class MetricsResponse(BaseModel):
    binary_metrics: Dict[str, float]
    multiclass_metrics: Dict[str, Dict[str, float]]
    confusion_matrix_binary: List[List[int]]
    confusion_matrix_multiclass: List[List[int]]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    version: str
    training_date: str
    hyperparameters: Dict[str, Any]
    feature_count: int
    classes_binary: List[str]
    classes_multiclass: List[str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int