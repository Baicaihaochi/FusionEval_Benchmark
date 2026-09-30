"""Validate release templates without model files, GPUs or downloads."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import yaml
from fusioneval.config import load_config
from fusioneval.continual import load_continual_config, compile_continual_plan

ROOT=Path(__file__).resolve().parents[1]
DOMAINS={'math','code','agent','if','science'}

class PortableConfigTests(unittest.TestCase):
    def test_all_release_templates_have_five_experts_and_local_defaults(self):
        paths=list((ROOT/'configs').rglob('*.yaml'))+list((ROOT/'examples').rglob('*.yaml'))
        count=0
        for p in paths:
            raw=yaml.safe_load(p.read_text())
            if 'method' not in raw:continue
            count+=1
            with self.subTest(config=p):
                self.assertEqual({x['id'] for x in raw['experts']},DOMAINS)
                self.assertEqual(len(raw['experts']),5)
                self.assertIsNone(raw['base'])
                self.assertIsNone(raw['output'])
                self.assertTrue(all(x['path'] is None for x in raw['experts']))
                self.assertFalse(Path(raw['model_profile']).is_absolute())
                if 'initial_model' in raw:self.assertFalse(Path(raw['initial_model']).is_absolute())
                if raw['workflow']=='continual':
                    c=load_continual_config(p);plan=compile_continual_plan(c)
                    self.assertEqual(plan['updates'],4)
                    self.assertEqual(plan['total_experts'],5)
                else:
                    c=load_config(p)
                    if raw['method']=='fisher':self.assertEqual(raw['parameters']['coefficients'],[.2]*5)
        self.assertEqual(count,80)
        for model in ('qwen3-4b','qwen3-8b','llama3.2-3b'):
            self.assertEqual(len(list((ROOT/f'examples/{model}').glob('*.yaml'))),11)
            self.assertEqual(len(list((ROOT/f'examples/{model}/continual').glob('*.yaml'))),5)

    def test_repository_relocation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('src','configs','examples'):
                shutil.copytree(ROOT/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
            code='''from pathlib import Path
from fusioneval.config import load_config
from fusioneval.continual import load_continual_config
root=Path.cwd()
for model in ('qwen3-4b','qwen3-8b','llama3.2-3b'):
 c=load_config(root/f'examples/{model}/ta.yaml')
 assert c.base==root/f'models/{model}/base'
 assert c.output.is_relative_to(root/'outputs')
 c=load_config(root/f'examples/{model}/featcal.yaml')
 assert c.initial_model==root/f'models/{model}/initial'
 c=load_continual_config(root/f'examples/{model}/continual/regmean.yaml')
 assert all(x.data.is_relative_to(root/'data') for x in c.experts if x.data)
'''
            result=subprocess.run([sys.executable,'-c',code],cwd=root,env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
