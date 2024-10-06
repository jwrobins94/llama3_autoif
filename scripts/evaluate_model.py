import lm_eval.api # TODO: for unknown reasons, this import must happen before other modules are loaded

from core.model import load_state_dict
from core.tokenizer import load_tokenizer
from core.evaluation import run_eval
import argparse
import json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to run IFEval on a trained model')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')

    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--limit', type=int, default=None, help='Optional limit on the number of evaluation rows')
    parser.add_argument(f'--batch-size', type=int, default=32, help='Batch size for evaluation')

    parser.add_argument(f'--benchmark', type=str, required=True, help='Benchmark to run. Supports hellaswag and ifeval')
    
    parser.add_argument(f'--output', type=str, default=None, help='Path to write sample results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.benchmark not in ['hellaswag', 'ifeval']:
        raise ValueError('Only "hellaswag" and "ifeval" benchmarks are supported.')
    print('Consider running with "accelerate launch scripts/evaluate_model.py ..." if you have multiple GPUs.')
    print('The lm_eval package does not appear to accept a HF API token. Run "huggingface-cli login" to cache your token locally before running this script.')

    tokenizer = load_tokenizer(args.hf_api_token)

    # Note: we delegate model loading to run_ifeval(...) to maintain compatibility with lm_eval's data parallelization
    scores, samples = run_eval(
        args.benchmark,
        args.model,
        tokenizer,
        args.batch_size,
        args.context_length,
        args.hf_api_token,
        args.limit or None,
        load_state_dict(args.ckpt) if args.ckpt else None
    )
    if not scores:
        exit() # rank > 0
    print(scores)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json.dumps(samples, indent=2))