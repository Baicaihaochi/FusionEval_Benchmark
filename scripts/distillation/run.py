import argparse
import json
import os
from pathlib import Path
from string import Template
import subprocess
import sys


def command(config, env, resume=None):
    env = dict(env)
    for key in ['STUDENT_HF', 'STUDENT_CHECKPOINT', 'PROMPT_DATA', 'OUTPUT_DIR']:
        if env.get(key):
            env[key] = str(Path(env[key]).resolve())
    arguments = [Template(value).substitute(env) for value in config['arguments']]
    if resume:
        for flag in ['--ref-load', '--ref-ckpt-step', '--start-rollout-id']:
            if flag in arguments:
                index = arguments.index(flag)
                del arguments[index:index + 2]
        arguments = [x for x in arguments if x not in {'--no-load-optim', '--no-load-rng', '--finetune'}]
        arguments += ['--load', str(Path(resume).resolve())]
    return [sys.executable, str(Path(__file__).with_name('train.py')), '--method', config['method'], *arguments]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--ray-address')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    env = os.environ.copy()
    cmd = command(config, env, args.resume)
    if args.dry_run:
        print(json.dumps(cmd, indent=2))
        return
    if not args.ray_address:
        parser.error('--ray-address is required to submit training')
    save = Path(env['OUTPUT_DIR']) / 'checkpoints'
    if save.exists() and not args.resume:
        raise FileExistsError('Existing checkpoints require --resume')
    runtime = {k: str(Path(env[k]).resolve()) for k in ['SLIME_ROOT', 'MEGATRON_ROOT']}
    runtime['PYTHONPATH'] = os.pathsep.join([runtime['MEGATRON_ROOT'], runtime['SLIME_ROOT'], str(Path(__file__).parent.resolve()), env.get('PYTHONPATH', '')])
    runtime['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    subprocess.run(['ray', 'job', 'submit', '--address', args.ray_address,
                    '--runtime-env-json', json.dumps({'env_vars': runtime}), '--', *cmd], check=True)


if __name__ == '__main__':
    main()
