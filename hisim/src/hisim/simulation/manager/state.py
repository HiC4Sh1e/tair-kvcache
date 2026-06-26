import logging
from hisim.utils import get_logger

logger = get_logger("hisim.debug")

class StateManager:
    _iteration: int = 0
    _global_clock: float = 0
    _last_inference_dur: float = 0
    _current_inference_dur: float = 0
    _hicache_l2_load_dur: float = 0
    _hicache_l2_backup_dur: float = 0

    @classmethod
    def reset(cls):
        cls._iteration = 0
        cls._global_clock = 0
        cls._last_inference_dur = 0
        cls._current_inference_dur = 0
        cls._hicache_l2_backup_dur = 0
        cls._hicache_l2_load_dur = 0

    @classmethod
    def inc_iteration(cls) -> None:
        cls._iteration += 1

    @classmethod
    def get_iteration(cls) -> int:
        return cls._iteration

    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        cls._hicache_l2_load_dur += dur

    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        cls._hicache_l2_backup_dur += dur

    @classmethod
    def pop_hicache_l2_load_dur(cls) -> float:
        dur = cls._hicache_l2_load_dur
        cls._hicache_l2_load_dur = 0

        # Debug: Log hicache l2 load duration (negative values affect token usage)
        if dur < 0:
            logger.warning(
                f"NEGATIVE pop_hicache_l2_load_dur detected: "
                f"dur={dur:.4f}s (this could cause negative token usage statistics)"
            )

        return dur

    @classmethod
    def pop_hicache_l2_backup_dur(cls) -> float:
        dur = cls._hicache_l2_backup_dur
        cls._hicache_l2_backup_dur = 0

        # Debug: Log hicache l2 backup duration
        if dur < 0:
            logger.warning(
                f"NEGATIVE pop_hicache_l2_backup_dur detected: "
                f"dur={dur:.4f}s"
            )

        return dur

    @classmethod
    def get_global_clock(cls) -> float:
        return cls._global_clock

    @classmethod
    def step_global_clock(cls, dur: float) -> None:
        old_clock = cls._global_clock
        cls._global_clock += dur

        # Debug: Check for unusual clock changes
        if dur < 0:
            logger.warning(
                f"NEGATIVE step_global_clock detected: "
                f"old_clock={old_clock:.4f}, dur={dur:.4f}, new_clock={cls._global_clock:.4f}"
            )

    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        old_clock = cls._global_clock
        cls._global_clock = clock

        # Debug: Check for unusual clock changes
        if clock < old_clock:
            logger.warning(
                f"NEGATIVE set_global_clock detected: "
                f"old_clock={old_clock:.4f}, new_clock={clock:.4f}, delta={clock - old_clock:.4f}"
            )

    @classmethod
    def set_current_inference_dur(cls, dur: float) -> None:
        cls._last_inference_dur = cls._current_inference_dur
        cls._current_inference_dur = dur

        # Debug: Check for unusual inference duration
        if dur < 0:
            logger.warning(
                f"NEGATIVE set_current_inference_dur detected: "
                f"last_dur={cls._last_inference_dur:.4f}, new_dur={dur:.4f}"
            )

    @classmethod
    def get_last_inference_dur(cls) -> float:
        return cls._last_inference_dur

    @classmethod
    def get_current_inference_dur(cls) -> float:
        return cls._current_inference_dur
