"""Offline recipe expansion: no Ray, SSH, SGLang or GPU processes are started."""
import os
import ast
import socket
from unittest.mock import patch
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TopologyTests(unittest.TestCase):
    def test_one_and_two_node_recipes(self):
        for model in ('4b', '8b'):
            for nodes in ('1', '2'):
                env = {**os.environ, 'NODE_COUNT': nodes, 'LOCAL_SERVICES': '1',
                       'HEAD_IP': '192.0.2.10', 'WORKER_IP': '192.0.2.11'}
                path = ROOT / f'examples/distillation/mopd_{model}.sh'
                # --plan after --resume must remain side-effect free.
                output = subprocess.check_output(['bash', str(path), '--resume', '/example/checkpoint', '--plan'], env=env, text=True)
                cmd = shlex.split(output.splitlines()[-1])
                value = lambda flag: cmd[cmd.index(flag) + 1]
                self.assertEqual(value('--actor-num-gpus-per-node'), '3' if nodes == '1' else '8')
                self.assertEqual(value('--rollout-num-gpus'), '3')
                self.assertEqual(value('--global-batch-size'), '510' if nodes == '1' else '512')
                self.assertEqual(value('--rollout-batch-size'), value('--global-batch-size'))
                self.assertEqual('--colocate' in cmd, nodes == '1')
                self.assertEqual('--sequence-parallel' in cmd, nodes == '2')
                self.assertEqual(value('--tensor-model-parallel-size'), '1' if nodes == '1' else ('2' if model == '4b' else '4'))
                teacher = '127.0.0.1' if nodes == '1' else env['WORKER_IP']
                self.assertIn(f'math=http://{teacher}:28000/generate', cmd)
                self.assertIn(f'agent=http://{teacher}:28004/generate', cmd)
                self.assertEqual(value('--load'), '/example/checkpoint')

    def test_worker_preview_and_shell_syntax(self):
        for path in list((ROOT / 'examples/distillation').glob('*.sh')) + [ROOT / 'scripts/distillation/services.sh']:
            subprocess.run(['bash', '-n', str(path)], check=True)
        env = {**os.environ, 'NODE_COUNT': '2', 'LOCAL_SERVICES': '1'}
        subprocess.run(['bash', str(ROOT / 'examples/distillation/mopd_4b.sh'), '--worker', '--plan'], env=env, check=True, capture_output=True)

    def test_training_node_precedes_ip_sort_order(self):
        source = (ROOT / 'slime/slime/ray/placement_group.py').read_text()
        tree = ast.parse(source)
        sort_key = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sort_key')
        namespace = {'socket': socket}
        exec(compile(ast.Module(body=[sort_key], type_ignores=[]), '<sort_key>', 'exec'), namespace)
        # A worker with a lower IP must not take the first eight actor slots.
        placement = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_create_placement_group')
        statements = [n for n in placement.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {'train_ip', 'sorted_bundle_infos'}]
        bundles = [(i, '192.0.2.20', i) for i in range(8)] + [(8+i, '192.0.2.10', i) for i in range(3)]
        namespace.update(os=os, bundle_infos=bundles)
        with patch.dict(os.environ, {'FUSIONEVAL_TRAIN_NODE_IP': '192.0.2.20'}):
            exec(compile(ast.Module(body=statements, type_ignores=[]), '<ordering>', 'exec'), namespace)
        self.assertEqual([item[1] for item in namespace['sorted_bundle_infos'][:8]], ['192.0.2.20'] * 8)
