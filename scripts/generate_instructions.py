from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import json
import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to run IFEval on a trained model')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.2-1B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--limit', type=int, default=None, help='Number of new instructions to generate')

    parser.add_argument(f'--input', type=str, required=True, help='Path to a file containing a newline-delimited list of seed instructions')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write generated instructions')
    return parser.parse_args()

def construct_prompt(seed_instructions: list[str]) -> str:
    seed_instructions_str = '\n'.join(seed_instructions)
    return f'''Your task is to generate a newline-delimited list of "verifiable instructions" that will be used to train a large language model.
    At the end of this message is a list of example instructions.

    Please provide 10 new instructions like this, with one instruction per line.
    Each instruction should be designed such that a competant Python programmer could write a function to verify whether a response satisfies the instruction.

    Examples:

    {seed_instructions_str}'''

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        seed_instructions = f.readlines()

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if torch.cuda.is_available():
        model.to('cuda:0')

    batch = tokenizer.apply_chat_template([
        {
            'role': 'user',
            'content': construct_prompt(seed_instructions)
        }
    ], add_generation_prompt=True, return_tensors='pt')

    max_new_tokens = 128
    outputs = model.generate(**batch, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, use_cache=True, do_sample=True, temperature=1.0)
    outputs = outputs[:, batch['input_ids'].shape[-1]:]
    decoded = tokenizer.decode(outputs[0])
    print(decoded)


    # Generate new instructions
    # Add new instructions to the list
    # Repeat

    #if args.output:
    #    with open(args.output, 'w') as f:
    #        f.write(json.dumps(samples, indent=2))