from core.tokenizer import load_tokenizer
import argparse
import torch
import json
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate test cases and verification functions')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')

    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--max-tokens', type=int, default=1024, help='Max tokens per generation')
    parser.add_argument(f'--batch-size', type=int, default=8, help='Batch size for generations')

    parser.add_argument(f'--num-verifications', type=int, required=True, help='Number of verifiers per instruction')

    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of 1_generate_instructions.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to the test cases and verification functions (JSON format)')

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

The user will issue the following query: {query}
Write the 'evaluate' function that checks whether a response to this query follows this instruction: {instruction}
If the instruction can be interpreted in multiple valid ways, write your code to accept any of those interpretations as valid.
Stop once you have finished writing the 'evaluate' function (omit test cases).
Think step-by-step and write your thought process as comments in the code.'''


def construct_test_case_prompt(instruction: str) -> str:
    # A common failure mode I've noticed is that the test cases will be overly simple.
    # As a result, we will have poor screening of verification functions and our actual model generations
    # will have a 100% pass or fail rate.
    return f'''Here are 3 example test cases for the instruction: use the letter B between 2 and 5 times
{{"response": "That's a great idea! You can buy a bar of soap at the local pharmacy.", "result": true}}
{{"response": "Babbel is a popular app used to learn languages and is suitable for beginners. However, it does not yet support the Bemba language.", "result": false}}

Now, write 2 test cases for your instruction: "{instruction}"
Write one test case per line in JSON format:
{{"response": "some response", "result": true or false}}'''


if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        instructions_w_queries = [json.loads(line) for line in f.read().splitlines()]
    
    # filter out empty instructions
    instructions_w_queries = [x for x in instructions_w_queries if x['instruction']]

    tokenizer = load_tokenizer(args.hf_api_token)
    if args.ckpt:
        raise ValueError('ckpt is not implemented for vllm')
    model = LLM(model=args.model, dtype=torch.bfloat16, tensor_parallel_size=torch.cuda.device_count())

    dataloader = DataLoader(instructions_w_queries, batch_size=args.batch_size)

    with open(f'{args.output}.jsonl', 'w') as output_file:
        instruction_idx = 0
        for instruction_w_query_batch in dataloader:
            all_prompts = []
            fn_prefix = 'def evaluate(response: str) -> bool:' # start the function spec to help the model get started
            for query, instruction in zip(instruction_w_query_batch['query'], instruction_w_query_batch['instruction']):
                instruction_idx += 1

                print(f'Processing instruction {instruction_idx}.')
                messages_mat = [[{'role': 'user', 'content': construct_verifier_prompt(query, instruction)}] for _ in range(args.num_verifications)]
                prompts = [
                        tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    ) for messages in messages_mat
                ]

                for i, prompt in enumerate(prompts):
                    prompts[i] = prompt + f'```{fn_prefix}'
                
                all_prompts.extend(prompts)
            
            #completions = generate_completions(model, tokenizer, all_prompts, ['```', tokenizer.eos_token, '<|eom_id|>'], args.max_tokens)
            
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=args.max_tokens,
                stop=['```', tokenizer.eos_token, '<|eom_id|>'],
            )
            outputs = model.generate(all_prompts, sampling_params)

            completions = []
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                completions.append(generated_text)
            
            verified_completions = [fn_prefix + completion for completion in completions]

            testcase_prefix = '{"response": "'
            all_prompts_2 = []
            for group_idx, (query, instruction) in enumerate(zip(instruction_w_query_batch['query'], instruction_w_query_batch['instruction'])):
                group_completions = completions[group_idx * args.num_verifications: (group_idx + 1) * args.num_verifications]
                for i, completion in enumerate(group_completions):
                    messages_mat[i].append({'role': 'assistant', 'content': f'```\n{fn_prefix}{completion}```'})
                    messages_mat[i].append({'role': 'user', 'content': construct_test_case_prompt(instruction)})

                prompts_2 = [
                    tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    for messages in messages_mat
                ]
                
                for i, prompt in enumerate(prompts_2):
                    prompts_2[i] = prompt + f'{testcase_prefix}'
                all_prompts_2.extend(prompts_2)

            #completions = generate_completions(model, tokenizer, all_prompts_2, [tokenizer.eos_token, '<|eom_id|>'], args.max_tokens)
            
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=args.max_tokens,
                stop=[tokenizer.eos_token, '<|eom_id|>'],
            )
            outputs = model.generate(all_prompts_2, sampling_params)

            completions = []
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                completions.append(generated_text)
            
            testcase_completions = [testcase_prefix + completion for completion in completions]
            
            for group_idx, (query, instruction) in enumerate(zip(instruction_w_query_batch['query'], instruction_w_query_batch['instruction'])):
                obj = {
                    'query': query,
                    'instruction': instruction,
                    'verifiers': verified_completions[group_idx * args.num_verifications: (group_idx + 1) * args.num_verifications],
                    'testcases': testcase_completions[group_idx * args.num_verifications: (group_idx + 1) * args.num_verifications]
                }
                output_file.write(json.dumps(obj))
                output_file.write('\n')
            output_file.flush()
