import time


def get_current_timestamp_in_milliseconds() -> int:
    """Get the current timestamp in milliseconds."""

    return int(time.time() * 1000)
