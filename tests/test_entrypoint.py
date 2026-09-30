"""Config/dispatch regression tests: no checkpoints, GPU, downloads or Ray needed."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from fusioneval.cli import main
from fusioneval.config import load_config
from fusioneval.continual import load_continual_config, compile_continual_plan
from fusioneval.errors import ConfigError
from fusioneval.plan import compile_plan

ROOT = Path(__file__).resolve().parents[1]


class EntryTests(unittest.TestCase):
    def test_builtin_method_contracts(self):
        # Developer-owned metadata is checked here, not on every CLI import.
        from fusioneval.registry import SPECS, resolve_method
        for name, plugin in SPECS.items():
            with self.subTest(method=name):
                self.assertIn(plugin.compute_precision, {'runtime', 'float32'})
                self.assertIn(plugin.execution, {'tensor', 'data'})
                self.assertIn(plugin.data_mode, {'none', 'per_expert', 'shared'})
                self.assertEqual(plugin.requires_data, plugin.data_mode != 'none')
                if plugin.requires_data or plugin.requires_initial_model:
                    self.assertEqual(plugin.execution, 'data')
                self.assertGreaterEqual(plugin.minimum_experts, 1)
                self.assertIn(plugin.input_mode, {'deltas', 'experts'})
                for key in (name, plugin.op_code, *plugin.aliases):
                    self.assertIs(resolve_method(key), plugin)
        with self.assertRaisesRegex(ConfigError, 'unknown method'):
            resolve_method('not_a_method')

    def test_launch_and_direct_cli_missing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing.yaml')
            for command in ('launch', 'plan'):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main([command, '--config', missing]), 2)
                self.assertIn(missing, stderr.getvalue())
                self.assertIn('fusioneval:', stderr.getvalue())

    def test_explicit_precision_options(self):
        from fusioneval.config import normalize_runtime
        self.assertEqual(normalize_runtime({})['precision'], 'bfloat16')
        for precision in ('bfloat16', 'float32'):
            self.assertEqual(normalize_runtime({'precision': precision})['precision'], precision)
        with self.assertRaisesRegex(ConfigError, 'bfloat16 or float32'):
            normalize_runtime({'precision': 'unsupported'})

    def test_default_paths_and_explicit_precedence(self):
        from fusioneval.assets import ROOT as ASSETS_ROOT, missing_hint
        c = load_config(ROOT / 'configs/qwen3-4b/fisher.yaml')
        self.assertEqual(c.base, ASSETS_ROOT / 'models/qwen3-4b/base')
        self.assertEqual(c.experts[0].data, ASSETS_ROOT / 'data/calibration/data/math.jsonl')
        self.assertIn('scripts/download/models.py --model qwen3-4b', missing_hint(c.base))
        self.assertIn('scripts/download/data.py --dataset calibration', missing_hint(c.experts[0].data))
        raw = yaml.safe_load((ROOT / 'configs/qwen3-4b/ta.yaml').read_text())
        raw['model_profile'] = str(ROOT / 'configs/models/qwen3-4b.yaml')
        with tempfile.TemporaryDirectory() as d:
            raw['base'] = 'custom-base'
            p = Path(d) / 'run.yaml'
            p.write_text(yaml.safe_dump(raw))
            c = load_config(p)
            self.assertEqual(c.base, (Path(d) / 'custom-base').resolve())
            self.assertEqual(c.experts[0].path, ASSETS_ROOT / 'models/qwen3-4b/math')

    def test_automatic_output_names_and_explicit_override(self):
        from fusioneval.outputs import output_path
        with patch('fusioneval.outputs.uuid4') as uid:
            uid.return_value.hex = 'a7c2e9f1' + '0' * 24
            p = output_path(None, None, ROOT, 'qwen3-4b', 'task_arithmetic', 3, {'scale': 0.3})
            self.assertEqual(p, ROOT / 'outputs/ta/qwen3-4b_m3_s0.3_a7c2e9f1')
            self.assertEqual(output_path('custom', 'fixed', ROOT, 'qwen3-4b', 'ties', 3, {}), ROOT / 'fixed')
            self.assertEqual(output_path('ignored', str(ROOT / 'results/exact name'), ROOT, 'qwen3-4b', 'ties', 3, {}), ROOT / 'results/exact name')
            self.assertIn('/outputs/continual/ta/', str(output_path(None, None, ROOT, 'qwen3-4b', 'task_arithmetic', 3, {'scale': 0.3}, continual=True)))
        first = load_config(ROOT / 'configs/qwen3-4b/ta.yaml')
        second = load_config(ROOT / 'configs/qwen3-4b/ta.yaml')
        self.assertNotEqual(first.output, second.output)
        self.assertFalse(first.output.exists())
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(['launch', '--config', str(ROOT / 'configs/continual/ta.yaml'), '--resume']), 2)
        self.assertIn('existing run', error.getvalue())

    def test_logs_default_external_and_failure(self):
        from fusioneval.run_logging import run_log
        from fusioneval.errors import ExecutionError
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(d)
            output = root / 'model'
            with run_log(output, snapshot={'runtime': {'seed': 42}}):
                self.assertFalse(output.exists())
                output.mkdir()
                print('fusion done')
            self.assertIn('fusion done', (output / 'run.log').read_text())
            self.assertEqual(yaml.safe_load((output / 'run_config.yaml').read_text())['runtime']['seed'], 42)
            before = (output / 'run.log').read_bytes()
            with self.assertRaises(ExecutionError):
                with run_log(output): pass
            self.assertEqual(before, (output / 'run.log').read_bytes())
            directory_log = root / 'logdir'
            directory_log.mkdir()
            with self.assertRaisesRegex(ExecutionError, 'not a directory'):
                with run_log(root / 'bad-directory', directory_log): self.fail('must reject before executing')
            with self.assertRaisesRegex(ExecutionError, 'end in .log'):
                with run_log(root / 'bad-name', root / 'bad-name/config.json'): self.fail('must reject before executing')
            external = root / 'external.log'
            with run_log(root / 'second', external): print('external log')
            self.assertTrue(external.exists())
            self.assertFalse((root / 'second/run.log').exists())
            with self.assertRaisesRegex(RuntimeError, 'example failure'):
                with run_log(root / 'failed'):
                    raise RuntimeError('example failure')
            self.assertEqual(json.loads((root / 'failed/run_status.json').read_text())['status'], 'FAILED')
            self.assertIn('example failure', (root / 'failed/run.log').read_text())

    def test_resolved_snapshot_preserves_seed_inputs_and_output(self):
        from fusioneval.run_logging import resolved_config
        config = load_config(ROOT / 'configs/qwen3-4b/dare.yaml')
        snapshot = resolved_config(config)
        self.assertEqual(snapshot['runtime']['seed'], 42)
        self.assertEqual(snapshot['output'], str(config.output))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'run_config.yaml'
            path.write_text(yaml.safe_dump(snapshot))
            reloaded = load_config(path)
            self.assertEqual(reloaded.output, config.output)
            self.assertEqual(reloaded.experts, config.experts)
            self.assertEqual(reloaded.method_parameters, config.method_parameters)

    def test_all_templates_plan(self):
        for folder in ['qwen3-4b', 'qwen3-8b', 'continual']:
            for path in (ROOT / 'configs' / folder).glob('*.yaml'):
                raw = yaml.safe_load(path.read_text())
                if 'method' not in raw:
                    continue
                with self.subTest(path=path):
                    self.assertNotIn('common', raw)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(['launch', '--config', str(path), '--plan']), 0)

    def test_variable_experts_method_data_and_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'run.yaml'
            raw = {'schema_version': 1, 'workflow': 'fusion', 'base': 'base',
                   'experts': [{'id': x, 'path': x, 'data': x+'.jsonl'} for x in ['a', 'b', 'c']],
                   'method': 'task_arithmetic', 'parameters': {'scale': 0.7},
                   'output_root': 'out', 'output': 'custom'}
            for count in [2, 3]:
                current = {**raw, 'experts': raw['experts'][:count]}
                p.write_text(yaml.safe_dump(current))
                c = load_config(p)
                self.assertEqual(len(c.experts), count)
                self.assertEqual(c.method_parameters['scale'], 0.7)
                self.assertEqual(c.experts[0].data, (Path(d)/'a.jsonl').resolve())
            raw.update(method='fisher', parameters={'coefficients': [0.2, 0.3, 0.5]})
            p.write_text(yaml.safe_dump(raw))
            inline = compile_plan(load_config(p))
            common = {k:v for k,v in raw.items() if k not in ['workflow','method','parameters','output']}
            (Path(d)/'common.yaml').write_text(yaml.safe_dump(common))
            p.write_text(yaml.safe_dump({k:raw[k] for k in ['schema_version','method','parameters','output']} | {'common':'common.yaml'}))
            legacy = compile_plan(load_config(p))
            self.assertEqual(inline['sources'], legacy['sources'])
            self.assertEqual(inline['method'], legacy['method'])
            self.assertEqual(inline['output'], legacy['output'])
            p.write_text(yaml.safe_dump({**raw, 'common':'common.yaml'}))
            with self.assertRaises(ConfigError): load_config(p)

    def test_continual_custom_count(self):
        raw=yaml.safe_load((ROOT/'configs/continual/ta.yaml').read_text())
        raw['model_profile']=str(ROOT/'configs/models/qwen3-4b.yaml')
        raw['experts'] = [x for x in raw['experts'] if x['id'] == raw['initial']] + [x for x in raw['experts'] if x['id'] != raw['initial']][:2]
        raw['order']=[x['id'] for x in raw['experts'] if x['id'] != raw['initial']]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sequence.yaml'; p.write_text(yaml.safe_dump(raw))
            plan=compile_continual_plan(load_continual_config(p))
            self.assertEqual(plan['updates'],2)
            from fusioneval.continual import _generated_stage_configs
            config = load_continual_config(p)
            stage = compile_continual_plan(config)['stages'][0]
            common, run = _generated_stage_configs(config, stage)
            (Path(d) / run['common']).write_text(yaml.safe_dump(common))
            stage_path = Path(d) / 'stage.yaml'
            stage_path.write_text(yaml.safe_dump(run))
            self.assertEqual(load_config(stage_path).output, Path(stage['output']))

    def test_della_stage_seeds_need_no_previous_rng(self):
        from fusioneval.continual import _generated_stage_configs, _same_fixed_contract
        config = load_continual_config(ROOT / 'examples/qwen3-4b/continual/della.yaml')
        plan = compile_continual_plan(config)
        self.assertEqual(plan['updates'], 4)
        self.assertEqual([s['parameters']['seed'] for s in plan['stages']], [42, 43, 44, 45])
        self.assertFalse(plan['rng_contract']['historical_rng_state'])
        stage = plan['stages'][1]
        common, raw = _generated_stage_configs(config, stage, {'seed': 43, 'rng_state': None})
        self.assertEqual(common['runtime']['seed'], raw['parameters']['seed'])
        self.assertIsNone(common['runtime']['rng_state'])
        self.assertEqual({e['id'] for e in common['experts']}, {'previous', 'incoming'})
        self.assertFalse(_same_fixed_contract(plan, {**plan, 'protocol': 'old-stream-protocol'}))

    def test_append_to_existing_two_expert_model(self):
        for method in ('soup', 'ta', 'ties', 'della', 'regmean'):
            raw = yaml.safe_load((ROOT / f'examples/qwen3-4b/continual/{method}.yaml').read_text())
            raw.update(initial='current', initial_experts=2, order=['agent'],
                       model_profile=str(ROOT / 'configs/models/qwen3-4b.yaml'),
                       experts=[{'id':'current','path':'current-model'}, {'id':'agent','path':'agent-model'}])
            if 'data' in raw: raw['data'] = {'agent': 'agent.jsonl'}
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'append.yaml'; p.write_text(yaml.safe_dump(raw))
                c = load_continual_config(p)
                plan = compile_continual_plan(c)
                self.assertEqual(plan['updates'], 1)
                self.assertEqual(plan['total_experts'], 3)
                self.assertIn('_m3_', str(c.run_root))
                self.assertFalse(plan['initial']['executes'])
                self.assertEqual(plan['initial']['experts_included'], 2)
                self.assertEqual(plan['stages'][0]['source_roles']['previous'], str(c.initial.path))
                self.assertEqual(plan['stages'][0]['experts_included'], 3)
                if method == 'della': self.assertEqual(plan['stages'][0]['parameters']['seed'], 43)
                if method == 'regmean': self.assertEqual(set(plan['expert_data']), {'agent'})

    def test_distillation_dispatch_without_submit(self):
        with patch('fusioneval.cli.subprocess.call', return_value=0) as call:
            self.assertEqual(main(['launch','--config',str(ROOT/'examples/distillation/seqkd_4b.sh'),'--plan']),0)
            self.assertIn('--plan',call.call_args[0][0])

    def test_distillation_uses_bundled_slime(self):
        spec = importlib.util.spec_from_file_location('distill', ROOT / 'scripts/distillation/run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            env = {'SLIME_ROOT': '', 'MEGATRON_ROOT': directory, 'STUDENT_HF': directory,
                   'STUDENT_CHECKPOINT': directory, 'STUDENT_STEP': '49',
                   'PROMPT_DATA': directory + '/train.jsonl', 'OUTPUT_DIR': directory + '/out'}
            with patch.dict(os.environ, env), patch('sys.argv', ['run.py', '--', '--num-rollout', '50']), patch.object(module.subprocess, 'run') as submit:
                module.main()
                cmd = submit.call_args[0][0]
                runtime = json.loads(cmd[cmd.index('--runtime-env-json') + 1])['env_vars']
                self.assertEqual(runtime['SLIME_ROOT'], str(ROOT / 'slime'))
                self.assertIn(str(ROOT / 'slime'), runtime['PYTHONPATH'].split(os.pathsep))
                self.assertEqual(cmd[-2:], ['--num-rollout', '50'])

    def test_distillation_shell_arguments_and_resume(self):
        import subprocess
        import shlex
        env = {**os.environ, 'STUDENT_HF': '/models/student with spaces',
               'STUDENT_CHECKPOINT': '/models/checkpoint', 'STUDENT_STEP': '49',
               'PROMPT_DATA': '/data/train.jsonl', 'OUTPUT_DIR': '/results/run'}
        for path in (ROOT / 'examples/distillation').glob('*.sh'):
            result = subprocess.run(['bash', str(path), '--plan'], env=env, capture_output=True, text=True, check=True)
            cmd = shlex.split(result.stdout.splitlines()[-1])
            self.assertEqual(cmd[cmd.index('--hf-checkpoint') + 1], env['STUDENT_HF'])
            self.assertEqual(cmd[cmd.index('--prompt-data') + 1], env['PROMPT_DATA'])
            self.assertEqual(cmd[cmd.index('--ref-ckpt-step') + 1], '49')
            result = subprocess.run(['bash', str(path), '--resume', '/results/previous', '--plan'], env=env, capture_output=True, text=True, check=True)
            cmd = shlex.split(result.stdout.splitlines()[-1])
            self.assertEqual(cmd[cmd.index('--load') + 1], '/results/previous')
            for flag in ['--ref-load', '--ref-ckpt-step', '--start-rollout-id', '--finetune', '--no-load-optim', '--no-load-rng']:
                self.assertNotIn(flag, cmd)


if __name__ == '__main__':
    unittest.main()
