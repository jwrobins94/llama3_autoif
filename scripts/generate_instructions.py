from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate instructions from a set of seed instructions via view-shot prompting')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--limit', type=int, required=True, help='Number of new instructions to generate')
    parser.add_argument(f'--tokens-per-completion', type=int, default=512, help='Number of new instructions to generate')
    parser.add_argument(f'--batch-size', type=int, default=8, help='Batch size for generations')
    parser.add_argument(f'--input', type=str, required=True, help='Path to a file containing a newline-delimited list of seed instructions')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write generated instructions')
    return parser.parse_args()

def construct_prompt(seed_instructions: list[str]) -> str:
    seed_instructions_str = '\n'.join(seed_instructions)
    # This prompt is largely copied from the source paper: https://arxiv.org/pdf/2406.13542v3
    return f'''You are an expert at writing instructions. Please provide instructions that meet
the following requirements:
- Instructions constrain the format but not style of the response
- Whether instructions are followed can be easily evaluated by a Python function
Here are some examples of instructions we need:
{seed_instructions_str}

Do not generate instructions about writing style, using metaphor, or translation. Here are
some examples of instructions we do not need:
- Incorporate a famous historical quote seamlessly into your answer
- Translate your answer into Pig Latin
- Use only words that are also a type of food
- Respond with a metaphor in every sentence
- Write the response as if you are a character from a Shakespearean play

Please generate one instruction per line in your response.
Each line should contain a single instruction and nothing else. Do not prefix your instructions with '-', numbers, or bullet points.
'''

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        seed_instructions = f.read().splitlines()

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if torch.cuda.is_available():
        model.to('cuda:0')

    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': construct_prompt(seed_instructions)}],
        add_generation_prompt=True,
        tokenize=False
    )
    print(prompt)

    batch = tokenizer([prompt]*args.batch_size, return_tensors='pt')
    batch.to(model.device)

    generated_instructions = []
    start_ts = time.time()
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
        decoded = tokenizer.batch_decode(outputs)
        for completion in decoded:
            generated_instructions.extend(completion.splitlines())
        print(f'Generated {len(generated_instructions)} out of {args.limit} instructions.')
    end_ts = time.time()
    print(f'Generated instructions in {end_ts - start_ts} seconds.')

    with open(args.output, 'w') as f:
        # we write out all of the generated instructions here
        # quality filtering happens later
        f.write('\n'.join(generated_instructions))
