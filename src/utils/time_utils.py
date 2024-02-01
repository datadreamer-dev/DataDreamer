from time import time

TIME_DURATION_UNITS = (
    ("week", 60 * 60 * 24 * 7),
    ("day", 60 * 60 * 24),
    ("hour", 60 * 60),
    ("min", 60),
    ("sec", 1),
)


def human_time_duration(seconds: float) -> str:  # pragma: no cover
    parts = []
    for unit, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(int(seconds), div)
        if amount > 0:
            parts.append("{} {}{}".format(amount, unit, "" if amount == 1 else "s"))
    if len(parts) == 0:
        return "0 secs"
    return ", ".join(parts)


def progress_eta(progress: float, start_time: float) -> str:
    elapsed_time = time() - start_time
    eta = (
        human_time_duration((elapsed_time / progress) - elapsed_time)
        if progress > 0
        else "calculating..."
    )
    return f"(Estimated time left: {eta})"
