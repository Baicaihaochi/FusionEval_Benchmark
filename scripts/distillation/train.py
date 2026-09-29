import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--method', choices=['mopd', 'seqkd'], required=True)
    selected, remaining = parser.parse_known_args()
    root = Path(os.environ['SLIME_ROOT']).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    if selected.method == 'mopd':
        from slime.ray import placement_group
        from slime.utils import arguments

        create_placement = placement_group._create_placement_group
        parse_args = arguments.parse_args

        def placement(num_gpus):
            return create_placement(11)

        def parse(*args, **kwargs):
            result = parse_args(*args, **kwargs)
            result.rollout_num_gpus = 11
            return result

        placement_group._create_placement_group = placement
        arguments.parse_args = parse

    sys.argv = [str(root / 'train.py'), *remaining]
    runpy.run_path(str(root / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
