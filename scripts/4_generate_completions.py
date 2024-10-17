from core.tokenizer import load_tokenizer
import argparse
import torch
import json
from torch.utils.data import DataLoader
import random
from vllm import LLM, SamplingParams

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate completions for each instruction')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')

    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--max-tokens', type=int, default=1024, help='Max tokens per generation')
    parser.add_argument(f'--num-completions', type=int, required=True, help='Number of completions per instruction')
    parser.add_argument(f'--batch-size', type=int, default=8, help='Batch size for generations')

    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of 3_filter_instructions.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write the final (instruction, verifiers, completions) tuples')
    return parser.parse_args()
    
def construction_generation_prompt(query: str, instruction: str) -> str:
    prompt = f'''You will be given a query and an instruction.
Your task is to respond to the query such that your response strictly adheres to the instruction.
After you respond, an automated system will take your response verbatim and run a Python function to check whether it follows the instruction.

Query: {query}
Instruction: {instruction}'''
    return prompt

if __name__ == '__main__':
    args = parse_args()
    random.seed(42)

    with open(args.input) as f:
        lines = f.read().splitlines()
        all_instructions = list(map(json.loads, lines))

    tokenizer = load_tokenizer(args.hf_api_token)
    
    if args.ckpt:
        raise ValueError('ckpt is not implemented for vllm')
    model = LLM(model=args.model, dtype=torch.bfloat16, tensor_parallel_size=torch.cuda.device_count(), max_model_len=args.context_length,
               gpu_memory_utilization=0.95)

    dataloader = DataLoader(all_instructions, batch_size=args.batch_size, collate_fn=list)

    with open(f'{args.output}.jsonl', 'w') as output_file:
        num_processed = 0
        for batch in dataloader:
            num_processed += len(batch)
            print(f'Generating completions: {num_processed}')
            all_prompts = []
            for elem in batch:
                query = elem['query']
                instruction = elem['instruction']
                verifiers = elem['verifiers']

                prompt = tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': construction_generation_prompt(query, instruction)}],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                all_prompts.append(prompt)

            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=args.max_tokens,
                n=args.num_completions,
                stop=[tokenizer.eos_token, '<|eom_id|>'],
            )
            outputs = model.generate(all_prompts, sampling_params)

            completions = []
            for output in outputs:
                prompt = output.prompt
                for o in output.outputs:
                    generated_text = o.text
                completions.append(generated_text)

            completions_per_query = args.num_completions
            
            for i, elem in enumerate(batch):
                res = dict(elem)
                res['completions'] = completions[i * completions_per_query: (i+1) * completions_per_query]
                output_file.write(json.dumps(res))
                output_file.write('\n')
            output_file.flush()
