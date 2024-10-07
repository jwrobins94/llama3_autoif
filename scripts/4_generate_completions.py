from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch
import json
from core.inference_utils import generate_completions
from core.data_utils import merge_outputs
from torch.utils.data import DataLoader, DistributedSampler

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate completions for each instruction')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')

    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--max-tokens', type=int, default=512, help='Max tokens per generation')
    parser.add_argument(f'--num-completions', type=int, required=True, help='Number of completions per instruction')
    parser.add_argument(f'--batch-size', type=int, default=8, help='Batch size for generations')

    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of 3_filter_instructions.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write the final (instruction, verifiers, completions) tuples')

    parser.add_argument(f'--local_rank', type=int, required=True, help='GPU index (set automatically by deepspeed)')
    return parser.parse_args()
    
def construction_generation_prompt(query: str, instruction: str) -> str:
    res = f'''Answer the user's query while strictly following the instruction.
Query: {query}
Instruction: {instruction}'''
    for i, verifier in enumerate(verifiers):
        res += f'\nVerifier {i + 1}:\n```\n{verifier}\n```'
    return res

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        lines = f.read().splitlines()
        all_instructions = list(map(json.loads, lines))

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token, args.ckpt)

    if torch.cuda.is_available():
        model.to(f'cuda:{args.local_rank}')

    torch.distributed.init_process_group('nccl', rank=args.local_rank)

    sampler = DistributedSampler(all_instructions, shuffle=False)
    dataloader = DataLoader(all_instructions, batch_size=args.batch_size, sampler=sampler, collate_fn=list)

    with open(f'{args.output}-{args.local_rank}.jsonl', 'w') as output_file:
        num_processed = 0
        for batch in dataloader:
            num_processed += len(batch)
            print(f'[{args.local_rank}] Generating completions: {num_processed}')
            all_prompts = []
            for elem in batch:
                query = elem['query']
                instruction = elem['instruction']

                prompts = [
                        tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': construction_generation_prompt(query, instruction)}],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                ] * args.num_completions
                all_prompts.extend(prompts)
            completions = generate_completions(model, tokenizer, all_prompts, [tokenizer.eos_token, '<|eom_id|>'], args.max_tokens)

            completions_per_query = args.num_completions
            
            for i, elem in enumerate(batch):
                res = dict(elem)
                res['completions'] = completions[i * completions_per_query: (i+1) * completions_per_query]
                output_file.write(json.dumps(res))
                output_file.write('\n')
            output_file.flush()
    
    torch.distributed.barrier()

    if args.local_rank == 0:
        merge_outputs(args.output)
