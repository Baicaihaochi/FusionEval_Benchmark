"""Download one released dataset into data/<dataset>, preserving Hub paths."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Frozen releases used in the paper; --revision can explicitly select another version.
DATASETS = {
    'calibration': ('FusionEval-Calibration', 'ca923e68d723ae94acd9049cb808cf3214e404e3'),
    'sft': ('FusionEval-SFT', 'bb581c9fb5333abfc6539666f45f2cf342434847'),
    'mopd': ('FusionEval-MOPD', 'e352676472a613d546f33d67f614bd66e67cccb0'),
    'student': ('FusionEval-Student', 'b3ec4038c40b433a05fcb24d19473821f6037ee0'),
}
DOMAINS = ['math', 'code', 'agent', 'if', 'science']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=[*DATASETS, 'all'], default='calibration')
    parser.add_argument('--domains', nargs='+', choices=DOMAINS, default=DOMAINS, help='domain subset for calibration/SFT')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data', help='data root')
    parser.add_argument('--revision', help='override the pinned release revision')
    parser.add_argument('--plan', action='store_true', help='print destinations without network access')
    args = parser.parse_args()
    for name in DATASETS if args.dataset == 'all' else [args.dataset]:
        repo, revision = DATASETS[name]
        destination = args.output_dir.expanduser().resolve() / name
        files = [f'data/{domain}.jsonl' for domain in args.domains] if name in {'calibration', 'sft'} else ['data/train.jsonl']
        print(f'David6995/{repo} -> {destination} ({", ".join(files)})', flush=True)
        if not args.plan:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id='David6995/' + repo, repo_type='dataset',
                              revision=args.revision or revision, local_dir=destination,
                              allow_patterns=['README.md', *files])


if __name__ == '__main__':
    main()
