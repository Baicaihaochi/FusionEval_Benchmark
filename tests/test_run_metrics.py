"""Runtime summaries without requiring CUDA hardware or model downloads."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fusioneval.run_logging import run_log


class RunMetricsTests(unittest.TestCase):
    def test_cuda_summary_and_resume_reset(self):
        cuda = Mock()
        cuda.max_memory_allocated.side_effect = [3 * 2**30, 2**30]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'model'
            external = Path(directory) / 'external.log'
            with patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}), contextlib.redirect_stdout(io.StringIO()):
                for resume, seconds, peak in [(False, 12.5, 3), (True, 4.0, 1)]:
                    with run_log(output, external, resume=resume, device='cuda:1') as metrics:
                        print('method output')
                        metrics['algorithm_runtime_seconds'] = seconds
                    state = json.loads((output / 'run_status.json').read_text())
                    self.assertEqual(state['metrics']['peak_gpu_memory_gib'], peak)
                    self.assertEqual(state['metrics']['algorithm_runtime_seconds'], seconds)
                    last = external.read_text().splitlines()[-1]
                    self.assertIn(f'algorithm={seconds:.2f}s', last)
                    self.assertIn(f'peak_gpu_memory={peak:.3f} GiB', last)
            self.assertEqual(cuda.reset_peak_memory_stats.call_count, 2)
            cuda.reset_peak_memory_stats.assert_called_with('cuda:1')
            self.assertEqual(cuda.synchronize.call_count, 4)

    def test_failure_keeps_summary_and_original_error(self):
        cuda = Mock()
        cuda.synchronize.side_effect = [None, RuntimeError('CUDA context failed')]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed'
            with patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'original failure'):
                    with run_log(output, device='cuda'):
                        raise ValueError('original failure')
            state = json.loads((output / 'run_status.json').read_text())
            self.assertEqual(state['error'], 'ValueError: original failure')
            self.assertIsNone(state['metrics']['peak_gpu_memory_gib'])
            self.assertGreaterEqual(state['metrics']['wall_time_seconds'], 0)
            last = (output / 'run.log').read_text().splitlines()[-1]
            self.assertIn('status=FAILED', last)
            self.assertIn('peak_gpu_memory=N/A', last)

    def test_cpu_summary(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / 'cpu'
            with run_log(output):
                print('CPU run')
            self.assertIn('peak_gpu_memory=N/A', (output / 'run.log').read_text().splitlines()[-1])
