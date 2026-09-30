"""Console + file logging without creating the model directory before atomic save."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
import time
import yaml

from .errors import ExecutionError


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return False


def resolved_config(config):
    """Save effective local inputs/settings beside the checkpoint for reuse."""
    continual = isinstance(config.method, str)
    result = {
        'schema_version': 1, 'workflow': 'continual' if continual else 'fusion',
        'method': config.method if continual else config.method.name,
        'parameters': config.parameters if continual else {
            key: value for key, value in config.method_parameters.items()
            if key in config.method.user_parameters
        },
        'base': str(config.base),
        'experts': [{'id': item.id, 'path': str(item.path),
                     **({'data': str(item.data)} if item.data and not continual else {})}
                    for item in config.experts],
        'runtime': config.runtime,
        'output': str(config.run_root if continual else config.output),
        'log': str(config.log) if config.log else None,
    }
    if continual:
        result.update(model_profile=str(config.model_profile_path), initial=config.initial_id, order=list(config.order))
        if config.method == 'regmean':
            result['data'] = {item.id: str(item.data) for item in config.experts}
    else:
        if config.model_profile: result['model_profile'] = str(config.model_profile.path)
        if config.initial_model: result['initial_model'] = str(config.initial_model)
        if config.calibration_data: result['calibration_data'] = str(config.calibration_data)
    return result


@contextmanager
def run_log(output: Path, log: Path = None, *, resume=False, snapshot=None, device="cpu"):
    if output.exists() and not resume:
        raise ExecutionError(f'Output already exists: {output}. Choose a new output or set output: null; existing files were not changed.')
    destination = log or output / 'run.log'
    if destination.exists() and not destination.is_file():
        raise ExecutionError(f'log must name a file, not a directory: {destination}')
    if destination.suffix.lower() != '.log':
        raise ExecutionError(f'log must end in .log to avoid overwriting model metadata or weights: {destination}')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Staging the log preserves the engine's atomic checkpoint commit.
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8', dir=output.parent) as stream:
        status, error = 'FAILED', None
        metrics = {'algorithm_runtime_seconds': None, 'peak_gpu_memory_gib': None}
        cuda = None
        started = time.perf_counter()
        try:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                print(f'Start: {datetime.now(timezone.utc).isoformat()}\nOutput: {output}\nLog: {destination}', flush=True)
                try:
                    if str(device).startswith('cuda'):
                        import torch
                        cuda = torch.cuda
                        cuda.synchronize(device)
                        cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    yield metrics
                    status = 'SUCCESS'
                    print('Status: SUCCESS', flush=True)
                except BaseException as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    traceback.print_exc()
                    raise
                finally:
                    if cuda is not None:
                        try:
                            cuda.synchronize(device)
                            metrics['peak_gpu_memory_gib'] = cuda.max_memory_allocated(device) / 2**30
                        except RuntimeError as exc:
                            # A failed CUDA context must not hide the workload's error.
                            print(f'GPU metrics unavailable: {exc}', flush=True)
                    metrics['wall_time_seconds'] = time.perf_counter() - started
                    algorithm = metrics['algorithm_runtime_seconds']
                    peak = metrics['peak_gpu_memory_gib']
                    print('Runtime summary | status={} | algorithm={} | wall={:.2f}s | peak_gpu_memory={} (PyTorch allocated)'.format(
                        status, 'N/A' if algorithm is None else f'{algorithm:.2f}s',
                        metrics['wall_time_seconds'], 'N/A' if peak is None else f'{peak:.3f} GiB'), flush=True)
        finally:
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream.seek(0)
            with destination.open('a', encoding='utf-8') as target:
                shutil.copyfileobj(stream, target)
            # Failure diagnostics are not a completed model; retain their cause.
            output.mkdir(parents=True, exist_ok=True)
            if snapshot is not None:
                (output / 'run_config.yaml').write_text(yaml.safe_dump(snapshot, sort_keys=False))
            (output / 'run_status.json').write_text(json.dumps({
                'status': status, 'error': error, 'output': str(output), 'metrics': metrics,
                'log': str(destination), 'finished_at': datetime.now(timezone.utc).isoformat(),
            }, indent=2) + '\n')
