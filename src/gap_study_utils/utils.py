

def _fmt_rutime(t: float):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    fmt = ""
    if hours:
        fmt += f"{int(hours)}h"
    if minutes:
        fmt += f"{int(minutes)}m"
    if seconds:
        fmt += f"{int(seconds)}s"
    return fmt
