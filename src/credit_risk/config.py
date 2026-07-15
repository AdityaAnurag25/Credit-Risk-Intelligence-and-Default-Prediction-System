from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CREDIT_RISK_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Path(__file__).resolve().parents[2]

    raw_data_path: Path = project_root / "data" / "raw" / "loan.csv"
    processed_data_dir: Path = project_root / "data" / "processed"
    models_dir: Path = project_root / "models"
    outputs_dir: Path = project_root / "outputs"
    dashboard_export_dir: Path = project_root / "dashboard" / "data_exports"

    random_seed: int = 42

    target_column: str = "target_default"
    issue_date_column: str = "issue_d"

    mlflow_tracking_uri: str = f"sqlite:///{project_root / 'mlflow.db'}"
    mlflow_experiment_name: str = "credit_risk_pd"
    mlflow_registered_model_name: str = "credit_risk_pd_model"

    category_vocab_path: Path = project_root / "outputs" / "models" / "category_vocab.json"
    feature_defaults_path: Path = project_root / "outputs" / "models" / "feature_defaults.json"

    decision_approve_below: float = 0.20
    decision_reject_above: float = 0.50

    reference_data_path: Path = project_root / "outputs" / "reference_data.parquet"
    current_data_path: Path = project_root / "outputs" / "current_data.parquet"
    drift_report_path: Path = project_root / "outputs" / "monitoring" / "drift_report.html"


settings = Settings()
