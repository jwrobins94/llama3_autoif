from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch
import json
from core.inference_utils import generate_completions
import glob
from torch.utils.data import DataLoader, DistributedSampler

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate test cases and verification functions')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--max-tokens', type=int, default=1024, help='Max tokens per generation')

    parser.add_argument(f'--num-verifications', type=int, required=True, help='Number of verifiers per instruction')

    parser.add_argument(f'--input', type=str, required=True, help='Path to a file containing a newline-delimited list of instructions')
    parser.add_argument(f'--output', type=str, required=True, help='Path to the test cases and verification functions (JSON format)')
    parser.add_argument(f'--local_rank', type=int, required=False, default=0, help='GPU index')
    return parser.parse_args()


def construct_verifier_prompt(query: str, instruction: str) -> str:
    # this prompt is derived from the one in the paper
    # however, with the LLama 8B models I had more success when breaking up the task into instructions, then verifications
    return f'''
You are an expert at writing evaluation functions in Python to evaluate whether a response
strictly follows an instruction.

You will be provided with a user query and a corresponding instruction.
Your task is to write a Python function named 'evaluate' to evaluate whether a response to this query 'response'
follows this instruction. If it follows, simply return True, otherwise return False.

Here is an example of a good output for the instruction: use the letter B between 2 and 5 times
```
def evaluate(response: str) -> bool:
    count = 0
    for char in response:
        if char.upper() == 'B':
            count += 1
    return 2 <= count <= 5
```

Here is an example of a good output for the instruction: answer in at most 27 characters
```
def evaluate(response: str) -> bool:
    return len(response) <= 27
```

The user will issue the following query: {query}
Write the 'evaluate' function that checks whether a response to this query follows this instruction: {instruction}'''


def construct_test_case_prompt(instruction: str) -> str:
    # A common failure mode I've noticed is that the test cases will be overly simple.
    # As a result, we will have poor screening of verification functions and our actual model generations
    # will have a 100% pass or fail rate.
    return f'''Now write 3 test cases for this verification function.
Write one test case per line in JSON format:
{{"response": "some response", "result": true or false}}

Here are 3 example test cases for the instruction: use the letter B between 2 and 5 times
{{"response": "That's a great idea! You can buy a bar of soap at the local pharmacy.", "result": true}}
{{"response": "Babbel is a popular app used to learn languages and is suitable for beginners. However, it does not yet support the Bemba language.", "result": false}}
{{"response": "I recommend that you bring at least $200 in cash for your trip to Bulgaria, as many companies will not accept credit cards.", "result": true}}

Now, write 3 test cases (one per line) for your instruction: {instruction}'''


if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        instructions_w_queries = [json.loads(line) for line in f.read().splitlines()]

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if torch.cuda.is_available():
        model.to(f'cuda:{args.local_rank}')

    torch.distributed.init_process_group('nccl', rank=args.local_rank)

    sampler = DistributedSampler(instructions_w_queries, shuffle=False)
    dataloader = DataLoader(instructions_w_queries, batch_size=1, sampler=sampler)

    with open(f'{args.output}-{args.local_rank}.jsonl', 'w') as output_file:
        instruction_idx = 0
        for instruction_w_query in dataloader:
            instruction_idx += 1

            print(f'[{args.local_rank}] Processing instruction {instruction_idx}.')
            messages_mat = [[{'role': 'user', 'content': construct_verifier_prompt(**instruction_w_query)}] for _ in range(args.num_verifications)]
            prompts = [
                    tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                ) for messages in messages_mat
            ]

            fn_prefix = 'def evaluate(response: str) -> bool:' # start the function spec to help the model get started
            for i, prompt in enumerate(prompts):
                prompts[i] = prompt + f'```{fn_prefix}'

            completions = generate_completions(model, tokenizer, prompts, '```', args.max_tokens)
            verified_completions = [fn_prefix + completion for completion in completions]

            for i, completion in enumerate(completions):
                messages_mat[i].append({'role': 'assistant', 'content': f'```\n{fn_prefix}{completion}```'})
                messages_mat[i].append({'role': 'user', 'content': construct_test_case_prompt(instruction_w_query['instruction'])})

            prompts_2 = [
                tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                for messages in messages_mat
            ]
            testcase_prefix = '{"response": "'
            for i, prompt in enumerate(prompts_2):
                prompts_2[i] = prompt + f'{testcase_prefix}'
            completions = generate_completions(model, tokenizer, prompts_2, tokenizer.eos_token, args.max_tokens)
            testcase_completions = [testcase_prefix + completion for completion in completions]
        
            obj = {
                'query': instruction_w_query['query'],
                'instruction': instruction_w_query['instruction'],
                'verifiers': verified_completions,
                'testcases': testcase_completions
            }
            output_file.write(json.dumps(obj))
            output_file.write('\n')
            output_file.flush()

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