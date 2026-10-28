import json
import logging
from pathlib import Path
from typing import List, Optional
from config.settings import settings
from src.core.models import TelemetrySignal

logger = logging.getLogger(__name__)


class TelemetryLogger:
    """Buffers, records, and queries runtime query telemetry signals."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or settings.TELEMETRY_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.in_memory_signals: List[TelemetrySignal] = []

    def log_signal(self, signal: TelemetrySignal) -> None:
        """Appends a TelemetrySignal to disk and memory buffer."""
        self.in_memory_signals.append(signal)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(signal.model_dump_json() + "\n")
        except Exception as e:
            logger.error("Failed to write telemetry signal to disk: %s", e)

    def record_user_feedback(self, signal_id: str, rating: float) -> bool:
        """Updates user rating for a specific telemetry trace (-1.0 to 1.0)."""
        found = False
        for s in self.in_memory_signals:
            if s.signal_id == signal_id:
                s.user_feedback = rating
                found = True
                break
        return found

    def get_degraded_signals(self, threshold: float = 0.50) -> List[TelemetrySignal]:
        """Identifies signals indicating failure: low confidence, retries, hallucinations, or negative feedback."""
        degraded = []
        for s in self.in_memory_signals:
            is_low_conf = s.avg_relevance < threshold
            is_hallucinated = s.hallucination_detected
            is_high_retry = s.retry_count > 0
            is_negative_feedback = s.user_feedback is not None and s.user_feedback < 0.0

            if is_low_conf or is_hallucinated or is_high_retry or is_negative_feedback:
                degraded.append(s)

        return degraded
