"""Submit a shell recipe's Slime arguments to an existing Ray cluster."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def command(arguments, resume=None):
    arguments = list(arguments)
    if resume:
        for flag in ['--ref-load', '--ref-ckpt-step', '--start-rollout-id']:
            if flag in arguments:
                index = arguments.index(flag)
                del arguments[index:index + 2]
        arguments = [x for x in arguments if x not in {'--no-load-optim', '--no-load-rng', '--finetune'}]
        arguments += ['--load', str(Path(resume).expanduser().resolve())]
    return [sys.executable, str(Path(__file__).with_name('train.py').resolve()), *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', '--plan', action='store_true', dest='dry_run')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--ray-address')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    cmd = command(arguments, args.resume)
    if args.dry_run:
        print(shlex.join(cmd))
        return
    # Required paths stay blank in the recipes; avoid interpreting them as cwd.
    os.environ['SLIME_ROOT'] = os.environ.get('SLIME_ROOT') or str(Path(__file__).resolve().parents[2] / 'slime')
    required = ['MEGATRON_ROOT', 'STUDENT_HF', 'PROMPT_DATA', 'OUTPUT_DIR']
    if not args.resume:
        required += ['STUDENT_CHECKPOINT', 'STUDENT_STEP']
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        parser.error('Fill in the .sh recipe: ' + ', '.join(missing))
    if (Path(os.environ['OUTPUT_DIR']).expanduser() / 'checkpoints').exists() and not args.resume:
        raise FileExistsError('Existing checkpoints require --resume or a new OUTPUT_DIR')
    runtime = {key: str(Path(os.environ[key]).expanduser().resolve()) for key in ['SLIME_ROOT', 'MEGATRON_ROOT']}
    runtime['PYTHONPATH'] = os.pathsep.join([runtime['MEGATRON_ROOT'], runtime['SLIME_ROOT'], str(Path(__file__).parent.resolve()), os.environ.get('PYTHONPATH', '')])
    runtime['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    if os.environ.get('LOCAL_SERVICES') == '1' and os.environ.get('NODE_COUNT', '1') == '2':
        runtime['FUSIONEVAL_TRAIN_NODE_IP'] = os.environ['HEAD_IP']
    for key in ('NO_PROXY', 'no_proxy'):
        if key in os.environ:
            runtime[key] = os.environ[key]
    cmd = ['ray', 'job', 'submit', '--runtime-env-json', json.dumps({'env_vars': runtime}), '--', *cmd]
    address = args.ray_address or os.environ.get('RAY_JOB_ADDRESS')
    if address:
        cmd[3:3] = ['--address', address]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
