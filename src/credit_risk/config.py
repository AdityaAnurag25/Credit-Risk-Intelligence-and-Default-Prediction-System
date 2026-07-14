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

    random_seed: int = 42

    target_column: str = "target_default"
    issue_date_column: str = "issue_d"


settings = Settings()
