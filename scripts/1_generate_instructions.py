from core.tokenizer import load_tokenizer
import argparse
import torch
import time
from core.data_utils import load_sharegpt_queries, load_hh_queries
import random
import json
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate instructions from a set of seed instructions via view-shot prompting')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--limit', type=int, required=True, help='Number of new instructions to generate')
    parser.add_argument(f'--tokens-per-completion', type=int, default=512, help='Max completion length')
    parser.add_argument(f'--batch-size', type=int, default=8, help='Batch size for generations')
    parser.add_argument(f'--query-source', type=str, default='sharegpt', help='Query source; either sharegpt or hh')

    parser.add_argument(f'--input', type=str, required=True, help='Path to a newline-delimited list of Query-Instruction pairs; see data/seed_instruction_pairs.txt for an example')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write generated instructions')

    return parser.parse_args()

def construct_prompt(seed_instructions_str: str) -> str:
    # This prompt is based heavily on the one provided in the source paper: https://arxiv.org/pdf/2406.13542v3
    return f'''You are an expert at writing instructions. Please provide instructions that meet
the following requirements:
- Instructions constrain the format but not style of the response
- Whether instructions are followed can be easily evaluated by a Python function
Here are some examples of instructions we need:
{seed_instructions_str}

Provide a list of unique (Query, Instruction) pairs that follow the same style of the examples.
The instruction should not respond to the query. It should extend the query with one or more unambiguous constraints.
For each instruction, it should be possible to write a small Python function to check whether a response adheres to the instruction exactly.
Each line should alternate between Query and Instruction.'''

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        seed_instructions = f.read() # read as a whole

    tokenizer = load_tokenizer(args.hf_api_token)
    
    if args.ckpt:
        raise ValueError('ckpt is not implemented for vllm')
    model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), max_model_len=args.context_length)

    base_prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': construct_prompt(seed_instructions)}],
        add_generation_prompt=True,
        tokenize=False
    )
    print(base_prompt)

    if args.query_source == 'sharegpt':
        print('Using sharegpt queries.')
        queries = load_sharegpt_queries()
    elif args.query_source == 'hh':
        print('Using helpful-base queries.')
        queries = load_hh_queries()
    else:
        raise ValueError(f'Unknown query source: {args.query_source}')

    # set up sampling
    all_prompts = []
    random.seed(42)
    for _ in range(args.limit):
        query = queries[random.randint(0, len(queries) - 1)]
        all_prompts.append({
            'query': query,
            'prompt': f'{base_prompt}\nQuery: {query}\nInstruction:'
        })
    dataloader = DataLoader(all_prompts, batch_size=args.batch_size)

    start_ts = time.time()
    with open(f'{args.output}.jsonl', 'w') as f:
        num_generated = 0
        for batch in dataloader:
            prompts = batch['prompt']
            num_generated += len(prompts)
            sampled_queries = batch['query']

            # completions = generate_completions(model, tokenizer, prompts, ['\n', tokenizer.eos_token, '<|eom_id|>'], args.tokens_per_completion)

            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=args.tokens_per_completion,
                stop=['\n', tokenizer.eos_token, '<|eom_id|>'],
            )
            outputs = model.generate(prompts, sampling_params)

            completions = []
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                completions.append(generated_text)

            for query, completion in zip(sampled_queries, completions):
                f.write(json.dumps({
                    'query': query,
                    'instruction': completion.strip()
                }))
                f.write('\n')
            print(f'Generated {num_generated} instructions.')
    end_ts = time.time()
    print(f'Generated instructions in {end_ts - start_ts} seconds.')
