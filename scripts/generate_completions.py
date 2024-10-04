from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch
import json
from core.inference_utils import generate_completions, wrap_with_deepspeed_inference

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate completions for each instruction')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--max-tokens', type=int, default=1024, help='Max tokens per generation')

    parser.add_argument(f'--num-completions', type=int, required=True, help='Number of completions per instruction')
    parser.add_argument(f'--deepspeed', default=False, action='store_true', help='Enables DeepSpeed Inference')

    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of filter_instructions.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write the final (instruction, verifiers, completions) tuples')
    return parser.parse_args()


def construction_generation_prompt(query: str, instruction: str) -> str:
    # this prompt is derived from the one in the paper
    # TODO!!!!!!!!!!!!!!!!!! Replace this with actual queries.
    return f'''How many legs does a typical dog have?
{instruction}
'''

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        lines = f.read().splitlines()
        instructions = list(map(json.loads, lines))

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if args.deepspeed:
        model = wrap_with_deepspeed_inference(model)

    if torch.cuda.is_available():
        model.to('cuda:0')

    with open(args.output, 'w') as output_file:
        for instruction_idx, instruction_w_verifiers in enumerate(instructions):
            print(f'Processing instruction {instruction_idx + 1} of {len(instructions)}.')

            instruction = instruction_w_verifiers['instruction']

            # TODO construction_generation_prompt(QUERY, ...)
            messages_mat = [[{'role': 'user', 'content': construction_generation_prompt('', instruction)}] for _ in range(args.num_completions)]
            prompts = [
                    tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                ) for messages in messages_mat
            ]

            completions = generate_completions(model, tokenizer, prompts, '```', args.max_tokens)
            
            res = dict(instruction_w_verifiers) # make a copy
            res['completions'] = completions

            output_file.write(json.dumps(res))
            output_file.write('\n')
            output_file.flush()

