"""Stop only process trees started by services.sh."""
import sys
import psutil

owned = []
for pid in sys.argv[1:]:
    try:
        process = psutil.Process(int(pid))
        owned.extend(process.children(recursive=True))
        owned.append(process)
    except psutil.NoSuchProcess:
        pass
for process in owned:
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        pass
_, alive = psutil.wait_procs(owned, timeout=10)
for process in alive:
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
