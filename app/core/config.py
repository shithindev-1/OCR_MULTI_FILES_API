"""Application settings, loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The project root, so `.env` is found no matter which directory the process starts in - a Windows
# service (NSSM) does not necessarily start in the project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    # Server - read by `python -m app`, the entry point the Windows service runs.
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    # Reverse proxies whose X-Forwarded-* headers are trusted (comma-separated IPs, or "*").
    forwarded_allow_ips: str = "127.0.0.1"
    # Swagger UI at /docs and the schema at /openapi.json. Turn off on an internet-facing host.
    docs_enabled: bool = True
    # Comma-separated keys accepted in the X-API-Key header on /api/v1 routes. Empty = auth off.
    api_keys: str = ""
    # Where the rotating log files go. Empty = the project's own logs/ folder.
    log_dir: str = ""
    # Comma-separated browser origins allowed to call the API. Empty = no CORS headers.
    cors_origins: str = ""

    gemini_api_key: str
    gemini_model: str
    gemini_max_tokens: int = 2048
    # Model for the two crop/rotate calls (box detection, orientation). Empty = GEMINI_MODEL.
    # A Flash-Lite model is faster and cheaper for these; extraction and verification never use it.
    gemini_vision_model: str = ""
    # Thinking tokens allowed to self- and cross-verification. Most of a check's time is spent
    # writing output, so a smaller budget answers sooner; too small can miss subtler problems.
    gemini_verification_thinking_budget: int = 256
    # Per Gemini call, not per request. A call that runs longer fails like any other provider
    # error, so a hung call can no longer hold a request open indefinitely.
    gemini_timeout_seconds: int = 105
    # Gemini scales oversized images down and bills them in 768px tiles anyway, so a larger
    # original adds no detail - only upload bandwidth and inline-payload size.
    gemini_max_image_edge: int = 1568
    require_https: bool = True
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    max_file_size_mb: int = 15
    # The whole box-detection/crop/rotate pipeline. Off means one model call per request instead
    # of three, and nothing written to detection_output_dir. A request may override this either
    # way with the X-Detection header.
    detection_enabled: bool = True
    detection_output_dir: str = "detections"
    detection_pdf_dpi: int = 200
    detection_crop_padding_px: int = 10
    # Equal on every side: orientation is only known after cropping, so there is no way to tell
    # in advance which edge will end up as the document's top and deserve the extra allowance.
    # Small, because edge refinement puts the corners on the document's real border — the old
    # generous margin only existed to absorb inaccurate corners, and showed up as background.
    detection_crop_padding_ratio: float = 0.015
    detection_crop_top_padding_ratio: float = 0.015
    detection_rotation_min_confidence: float = 0.7
    detection_crop_target_long_edge: int = 1600
    detection_crop_sharpen_amount: float = 0.6
    detection_crop_contrast_clip: float = 1.5
    # A fourth model call per document (on top of extraction, and crop+rotation when detection is
    # on): Gemini looks at the document again alongside its own extracted fields and judges what a
    # field-by-field rule can't - MRZ-to-printed-data consistency, front/back agreement, and other
    # internal contradictions. Same override pattern as detection_enabled: the X-Validation header
    # overrides this per request either way.
    validation_enabled: bool = True
    # How long a byte-identical file's extraction and passed self-verification are reused instead of
    # calling Gemini again, matched by the file's SHA-256. Held in process memory only - never on
    # disk - and cleared on restart. A finding is never reused, so a rejected file is always
    # re-checked. 0 = off.
    result_cache_seconds: int = 10800
    # Files under detection_output_dir and leasing_output_dir (originals, crops, merged PDFs) older
    # than this are deleted by a background sweep that runs at startup and then hourly. 0 = keep forever.
    output_retention_hours: int = 24
    log_retention_days: int = 30
    # Writes every extraction's parsed fields - National ID, DOB, passport/visa numbers - to
    # logs/results.log. Development only: never enable where real customer documents are processed.
    log_extracted_results: bool = False
    # Hosts a /ocr/leasing `source` URL may point at, besides the configured blob container
    # (comma-separated, exact hostnames). Empty = only the blob container. Stops the endpoint being
    # used to make the server fetch arbitrary or internal addresses.
    allowed_source_hosts: str = ""
    # Where the /ocr/leasing endpoints write each request's originals, crops and merged PDFs,
    # one folder per document_refnumber.
    leasing_output_dir: str = "postman_output"
    # A country value in form_data that app/data does not recognise is returned as extracted and
    # saved to a new-names list (new_country_codes.json / new_nationalities.json in this folder);
    # Gemini fills in its exact country name in the background, so the next request with the same
    # value is converted. Kept outside app/data, and out of git, because the server writes to it.
    # A relative path is under the project root. Off = unrecognised values are only returned as is.
    country_learning_enabled: bool = True
    country_learning_dir: str = "learned_countries"
    # Azure Blob Storage for /ocr/leasing: sources are fetched from this container and each
    # document's cropped/rotated result is uploaded back beside its original as <name>_ocr.<ext>.
    # Both unset = blob storage off (sources must then be plain downloadable URLs).
    azure_storage_connection_string: str | None = None
    azure_storage_container: str | None = None

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def api_key_list(self) -> list[str]:
        return _csv(self.api_keys)

    @property
    def cors_origin_list(self) -> list[str]:
        return _csv(self.cors_origins)

    @property
    def allowed_source_host_list(self) -> list[str]:
        return [host.lower() for host in _csv(self.allowed_source_hosts)]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()