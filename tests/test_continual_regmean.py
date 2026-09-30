"""Input-access regressions; optional tiny tensor checks never load real models."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml
from fusioneval.config import load_config
from fusioneval.continual import (
    load_continual_config, compile_continual_plan, _generated_stage_configs,
    _same_fixed_contract,
)
from fusioneval.methods.continual_regmean import runner, normalize
from fusioneval.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]


class ContinualRegMeanAccessTests(unittest.TestCase):
    def test_five_models_only_previous_incoming_and_new_data(self):
        config = load_continual_config(ROOT/'examples/qwen3-4b/continual/regmean.yaml')
        plan = compile_continual_plan(config)
        self.assertEqual(plan['updates'], 4)
        self.assertFalse(plan['initial']['executes'])
        self.assertNotIn(config.initial_id, plan['expert_data'])
        self.assertFalse(plan['access_contract']['previous_cumulative_state'])
        self.assertIsNone(plan['common_anchor'])
        for stage in plan['stages']:
            self.assertEqual(set(stage['source_roles']), {'previous', 'incoming', 'incoming_data'})
            common, raw = _generated_stage_configs(config, stage)
            self.assertEqual(common['base'], stage['source_roles']['previous'])
            self.assertEqual(len(common['experts']), 1)
            self.assertEqual(common['experts'][0]['data'], stage['source_roles']['incoming_data'])
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)
                (p/raw['common']).write_text(yaml.safe_dump(common))
                (p/'run.yaml').write_text(yaml.safe_dump(raw))
                load_config(p/'run.yaml')
        self.assertEqual(plan['stages'][1]['source_roles']['previous'], plan['stages'][0]['output'])
        old = {**plan, 'protocol': 'old-protocol'}
        self.assertFalse(_same_fixed_contract(old, plan))

    def test_stage_zero_does_not_accept_statistics_initialisation(self):
        with self.assertRaises(ConfigError):
            normalize({'alpha': .7, 'experts_included': 1})

    def test_both_models_recomputed_on_incoming_data(self):
        incoming=SimpleNamespace(path=Path('/new-model'), data=Path('/new-domain.jsonl'))
        config=SimpleNamespace(base=Path('/current-model'), experts=[incoming],
            runtime={'device':'cpu','max_shard_size_gib':5}, method_parameters={'examples':128})
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {'torch':fake_torch}), \
                patch.object(runner,'precision',return_value='float32'), \
                patch.object(runner,'load_tokenizer',return_value='tokenizer') as tokenizer, \
                patch.object(runner,'load_model',side_effect=['current','new']) as load, \
                patch.object(runner,'collect_grams',return_value=({'gram':object()},{'examples':128},{'linear':'gram'})) as collect, \
                patch.object(runner,'write_stats') as write:
            runner.collect_pair_stats(config,Path(d))
            self.assertEqual([c.args[0] for c in load.call_args_list],[config.base,incoming.path])
            tokenizer.assert_called_once_with(config.base)
            self.assertEqual([c.args[:3] for c in collect.call_args_list],
                             [('current','tokenizer',incoming.data),('new','tokenizer',incoming.data)])
            self.assertEqual([c.args[1] for c in write.call_args_list],[Path(d)/'previous',Path(d)/'incoming'])

    def test_runner_reads_only_fresh_workspace_stats_and_publishes_no_state(self):
        config=SimpleNamespace(base=Path('/current-model'),experts=[SimpleNamespace(path=Path('/new-model'))],
            method_parameters={'alpha':.7,'experts_included':3,'examples':128,'leftover_edge':'mean','leftover_1d':'mean'})
        with tempfile.TemporaryDirectory() as d, \
                patch.object(runner,'collect_pair_stats',return_value=({},{})), \
                patch.object(runner,'read_map',return_value={}) as read, \
                patch.object(runner,'checkpoint_tensors',return_value={}), \
                patch.object(runner,'open_model'), \
                patch.object(runner,'stat_readers',return_value=([{},{}],[None,None])) as stats:
            result=runner.execute(config,Path(d))
            self.assertEqual([c.args[0] for c in read.call_args_list],[config.base,config.experts[0].path])
            self.assertEqual(stats.call_args.args[1],[Path(d)/'previous',Path(d)/'incoming'])
            self.assertEqual(result.artifacts,{})
            self.assertFalse(result.diagnostics['historical_statistics_loaded'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required for tiny tensor checks')
class ContinualRegMeanNumericTests(unittest.TestCase):
    def test_three_way_pairwise_differs_from_batch(self):
        import torch
        g=torch.eye(2)
        weights=[torch.full((1,2),v) for v in (1.,2.,3.)]
        first=runner.pairwise_regmean_weight(weights[0],weights[1],g,g,.7)
        final=runner.pairwise_regmean_weight(first,weights[2],g,g,.7)
        torch.testing.assert_close(final,torch.full((1,2),2.25))
        self.assertFalse(torch.allclose(final,sum(weights)/3))

    def test_leftovers_are_pairwise_not_expert_count_weighted(self):
        import torch
        value=runner.leftover_update(torch.tensor([2.]),torch.tensor([6.]),'norm.weight','mean','mean')
        torch.testing.assert_close(value,torch.tensor([4.]))
        preserved=runner.leftover_update(torch.tensor([2.]),torch.tensor([6.]),'norm.weight','mean','base')
        torch.testing.assert_close(preserved,torch.tensor([2.]))
