import lm_eval.api # TODO: for unknown reasons, this import must happen before other modules are loaded

from core.model import load_state_dict
from core.tokenizer import load_tokenizer
from core.evaluation import run_ifeval
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
    
    parser.add_argument(f'--output', type=str, default=None, help='Path to write sample results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('Consider running with "accelerate launch scripts/evaluate_model.py ..." if you have multiple GPUs.')

    tokenizer = load_tokenizer(args.hf_api_token)

    # We delegate model loading to run_ifeval(...) to maintain compatibility with lm_eval's data parallelization
    scores, samples = run_ifeval(
        args.model,
        tokenizer,
        args.batch_size,
        args.context_length,
        args.hf_api_token,
        args.limit or None,
        load_state_dict(args.ckpt) if args.ckpt else None)
    print(scores)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json.dumps(samples, indent=2))