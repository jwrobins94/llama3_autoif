import torch.distributed
from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch
import time
from core.inference_utils import generate_completions, wrap_with_deepspeed_inference
from data.data_utils import load_sharegpt_queries
import random
import json
import glob
from torch.utils.data import DataLoader, DistributedSampler

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate instructions from a set of seed instructions via view-shot prompting')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--limit', type=int, required=True, help='Number of new instructions to generate')
    parser.add_argument(f'--tokens-per-completion', type=int, default=512, help='Max completion length')
    parser.add_argument(f'--batch-size', type=int, default=8, help='Batch size for generations')
    parser.add_argument(f'--input', type=str, required=True, help='Path to a file containing a newline-delimited list of seed instructions')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write generated instructions')
    parser.add_argument(f'--deepspeed', default=False, action='store_true', help='Enables DeepSpeed Inference')
    parser.add_argument(f'--local_rank', type=int, required=False, default=0, help='GPU index')

    return parser.parse_args()

def construct_prompt(seed_instructions_str: str) -> str:
    # This prompt is largely copied from the source paper: https://arxiv.org/pdf/2406.13542v3
    return f'''You are an expert at writing instructions. Please provide instructions that meet
the following requirements:
- Instructions constrain the format but not style of the response
- Whether instructions are followed can be easily evaluated by a Python function
Here are some examples of instructions we need:
{seed_instructions_str}

Provide a list of unique (Query, Instruction) pairs that follow the same style of the examples.
Each line should alternate between Query and Instruction.'''

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        seed_instructions = f.read() # read as a whole

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if torch.cuda.is_available():
        model.to(f'cuda:{args.local_rank}')

    torch.distributed.init_process_group('nccl')
    if args.deepspeed:
        model = wrap_with_deepspeed_inference(model)

    base_prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': construct_prompt(seed_instructions)}],
        add_generation_prompt=True,
        tokenize=False
    )
    print(base_prompt)

    queries = load_sharegpt_queries()

    # set up sampling
    all_prompts = []
    random.seed(42)
    for _ in range(args.limit):
        query = queries[random.randint(0, len(queries) - 1)]
        all_prompts.append({
            'query': query,
            'prompt': f'{base_prompt}\nQuery: {query}\nInstruction:'
        })
    sampler = DistributedSampler(all_prompts, shuffle=False)
    dataloader = DataLoader(all_prompts, batch_size=args.batch_size, sampler=sampler)

    start_ts = time.time()
    with open(f'{args.output}-{args.local_rank}.jsonl', 'w') as f:
        num_generated = 0
        for batch in dataloader:
            num_generated += len(batch)
            prompts = [row['prompt'] for row in batch]
            sampled_queries = [row['query'] for row in batch]
            completions = generate_completions(model, tokenizer, prompts, '\n', args.tokens_per_completion)
            for query, completion in zip(sampled_queries, completions):
                f.write(json.dumps({
                    'query': query,
                    'instruction': completion.strip()
                }))
                f.write('\n')
            print(f'[{args.local_rank}] Generated {num_generated} instructions.')
    end_ts = time.time()
    print(f'[{args.local_rank}] Generated instructions in {end_ts - start_ts} seconds.')

    torch.distributed.barrier()

    if args.local_rank == 0:
        # merge results
        all_results = []
        for path in glob.glob(f'{args.output}-*.jsonl'):
            with open(path) as f:
                local_data = f.read()
                all_results.append(local_data)
        
        with open(f'{args.output}.jsonl', 'w') as f:
            f.write(''.join(all_results))

        
