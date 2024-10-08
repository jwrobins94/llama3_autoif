import argparse
from core.data_utils import merge_outputs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to merge shared input files into a single output file')
    parser.add_argument(f'--prefix', type=str, required=True, help='Prefix to sharded input files')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    merge_outputs(args.prefix)
