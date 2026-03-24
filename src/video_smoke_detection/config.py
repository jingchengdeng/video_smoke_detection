from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    max_size: int = 500
    mhi_duration: int = 20
    default_threshold: int = 30
    max_time_delta: int = 3
    min_time_delta: int = 2
    color_threshold: int = 85
