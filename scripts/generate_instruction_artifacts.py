from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch
import json
from transformers import StopStringCriteria, PreTrainedTokenizerFast

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate test cases and verification functions')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--max-tokens', type=int, default=1024, help='Context length')

    parser.add_argument(f'--num-verifications', type=int, required=True, help='Number of verifiers per instruction')

    parser.add_argument(f'--input', type=str, required=True, help='Path to a file containing a newline-delimited list of instructions')
    parser.add_argument(f'--output', type=str, required=True, help='Path to the test cases and verification functions (JSON format)')
    return parser.parse_args()


def construct_verifier_prompt(instruction: str) -> str:
    # this prompt is derived from the one in the paper
    # however, with the LLama 8B models I had more success when breaking up the task into instructions, then verifications
    return f'''
You are an expert at writing evaluation functions in Python to evaluate whether a response
strictly follows an instruction.

You will be provided with a single instruction.
Your task is to write a Python function named 'evaluate' to evaluate whether an input string 'response'
follows this instruction. If it follows, simply return True, otherwise return False.

Here is an example of a good output for the instruction: use the letter B at least once
```
def evaluate(response: str) -> bool:
    return 'B' in response
```

Here is an example of a good output for the instruction: answer in at most 27 characters
```
def evaluate(response: str) -> bool:
    return len(response) <= 27
```

Now, please write the evaluate function for the following instruction: {instruction}'''


def construct_test_case_prompt(instruction: str) -> str:
    return f'''Now write 3 test cases for this verification function.
Write one test case per line in JSON format:
{{"response": "some response", "result": "true" or "false"}}

Here are 3 example test cases for the instruction: use the letter B at least once
{{"response": "Bar", "result": "true"}}
{{"response": "Foo", "result": "false"}}
{{"response": "CAB", "result": "true"}}

Now, write 3 test cases (one per line) for the following instruction: {instruction}'''


def generate_completion(model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, prompt: str, stop_str: str) -> str:
    batch = tokenizer([prompt], return_tensors='pt')
    batch.to(model.device)

    outputs = model.generate(
        **batch,
        max_new_tokens=args.max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=True,
        temperature=1.0,
        stopping_criteria=[StopStringCriteria(tokenizer, [stop_str])]
    )
    outputs = outputs[:, batch['input_ids'].shape[-1]:]
    decoded = tokenizer.batch_decode(outputs)[0]
    if stop_str in decoded:
        decoded = decoded[:decoded.index(stop_str)]
    return decoded

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        instructions = f.read().splitlines()

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if torch.cuda.is_available():
        model.to('cuda:0')

    for instruction in instructions[:2]:
        for _ in range(args.num_verifications):
            print(instruction)
            messages = [{'role': 'user', 'content': construct_verifier_prompt(instruction)}]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            # TODO: leverage structured decoding to avoid generations with basic syntactic errors.
            fn_prefix = 'def evaluate(response: str) -> bool:' # start the function spec to help the model get started
            prompt += f'```\n{fn_prefix}'
            completion = generate_completion(model, tokenizer, prompt, '```')
            verified_completion = fn_prefix + completion
            print(verified_completion)

            messages.append({'role': 'assistant', 'content': f'```\n{fn_prefix}{completion}```'})
            messages.append({'role': 'user', 'content': construct_test_case_prompt(instruction)})

            prompt_2 = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            testcase_prefix = '{"response": "'
            prompt_2 += f'```\n{testcase_prefix}'
            completion = generate_completion(model, tokenizer, prompt_2, tokenizer.eos_token)
            testcase_completion = testcase_prefix + completion
            print(testcase_completion)
        print('------------')

