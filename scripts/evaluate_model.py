import lm_eval.api # for unknown reasons, this import must happen before other modules are loaded. TODO

from core.model import load_model
from core.tokenizer import load_tokenizer
from evaluation.ifeval import run_ifeval
import argparse
import json
import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to run IFEval on a trained model')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.2-1B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--limit', type=int, default=None, help='Optional limit on the number of evaluation rows')
    parser.add_argument(f'--output', type=str, default=None, help='Path to write sample results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if torch.cuda.is_available():
        model.to('cuda:0')

    scores, samples = run_ifeval(model, tokenizer, args.limit or None)
    print(scores)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json.dumps(samples, indent=2))