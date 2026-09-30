"""Download published base/experts into models/<model>/<role>. Run explicitly."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES = {'base': None, 'math': 'Math', 'code': 'Code', 'agent': 'Agent', 'if': 'IF', 'science': 'Science'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen3-4b', 'qwen3-8b', 'llama3.2-3b'], default='qwen3-4b')
    parser.add_argument('--include', nargs='+', choices=ROLES, default=list(ROLES), help='default: base and all five experts')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models', help='models root')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--plan', action='store_true', help='print destinations without network access')
    args = parser.parse_args()
    llama = args.model == 'llama3.2-3b'
    model = 'Llama-3.2-3B' if llama else 'Qwen3-' + args.model.split('-')[-1].upper()
    for role in args.include:
        suffix = ('base' if llama else 'Base-SFT-R-64') if role == 'base' else 'expert-' + ROLES[role]
        repo = f'David6995/{model}-{suffix}'
        destination = args.output_dir.expanduser().resolve() / args.model / role
        print(f'{repo} -> {destination}', flush=True)
        if not args.plan:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, revision=args.revision, local_dir=destination)


if __name__ == '__main__':
    main()
