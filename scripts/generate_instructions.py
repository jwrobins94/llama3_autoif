from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to run IFEval on a trained model')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.2-1B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--limit', type=int, required=True, help='Number of new instructions to generate')
    parser.add_argument(f'--tokens-per-completion', type=int, default=128, help='Number of new instructions to generate')
    parser.add_argument(f'--input', type=str, required=True, help='Path to a file containing a newline-delimited list of seed instructions')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write generated instructions')
    return parser.parse_args()

def construct_prompt(seed_instructions: list[str]) -> str:
    seed_instructions_str = '\n'.join(seed_instructions)
    return f'''Below is a list of "verifiable instructions" that will be used to train a large language model.
Each instruction has the following properties:
1. A competant Python programmer could write a function to verify whether a response satisfies the instruction.
2. The instruction can be followed without knowledge of external data sources.
3. The instruction is broadly applicable to a variety of user queries.

Below are a set of bad instructions that are hard to verify:
Use only words that are odd numbers (e.g., one, three, five)
Use only the first half of the sentence
Answer with a statement that ends in a rhetorical question

Below are a set of good instructions:
{seed_instructions_str}
'''

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        seed_instructions = f.read().splitlines()

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if torch.cuda.is_available():
        model.to('cuda:0')

    prompt = construct_prompt(seed_instructions)
    print(prompt)

    batch = tokenizer([prompt], return_tensors='pt')
    batch.to(model.device)

    generated_instructions = []
    while len(generated_instructions) < args.limit:
        outputs = model.generate(
            **batch,
            max_new_tokens=args.tokens_per_completion,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            do_sample=True,
            temperature=1.0
        )
        outputs = outputs[:, batch['input_ids'].shape[-1]:]
        decoded = tokenizer.decode(outputs[0])
        print(decoded)
        new_instructions = decoded.splitlines()
        generated_instructions.extend(new_instructions)
        print(f'Generated {len(generated_instructions)} out of {args.limit} instructions.')

    with open(args.output, 'w') as f:
        # we write out all of the generated instructions here
        # quality filtering happens later
        f.write('\n'.join(generated_instructions))
