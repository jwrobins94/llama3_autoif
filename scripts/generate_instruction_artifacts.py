from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch
import json
from transformers import StopStringCriteria

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



def construct_test_and_verifier_prompt(instruction: str) -> str:
    example_evaluate = json.dumps('''def evaluate(response: str) -> bool:
    return 'B' in response''')

    return f'''
You are an expert for writing evaluation functions in Python to evaluate whether a response
strictly follows an instruction.

You will be provided with a single instruction.
Please write a Python function named 'evaluate' to evaluate whether an input string 'response'
follows this instruction. If it follows, simply return True, otherwise return False.
Please respond with a single JSON that includes the evaluation function in the key 'func',
and a list of three test cases in the key 'cases', which includes an input in the key 'input' and
an expected output in the key 'output' (True or False).

Here is the expected JSON format:
```json
{{
"func": "JSON Str“,
"cases": [ {{ "input": "str", "output": "True" }}, {{ "input": "str", "output": "False" }} ]
}}
```

Here is an example of a good output for the instruction: use the letter B at least once
```json
{{
"func": {example_evaluate},
"cases": [ {{ "input": "foo", "output": "False" }}, {{ "input": "Bar", "output": "True" }}, {{ "input": "CAB", "output": "True" }} ]
}}```

Place your answer in a code block, as in the example above.
Here is your instruction: {instruction}'''

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        instructions = f.read().splitlines()

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if torch.cuda.is_available():
        model.to('cuda:0')

    for instruction in instructions[:2]:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': construct_test_and_verifier_prompt(instruction)}],
            add_generation_prompt=True,
            tokenize=False
        )

        # TODO: leverage structured decoding to avoid generations with basic syntactic errors.
        prompt += '```json'

        batch = tokenizer([prompt] * args.num_verifications, return_tensors='pt')
        batch.to(model.device)

        outputs = model.generate(
            **batch,
            max_new_tokens=args.max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=3,
            stopping_criteria=[StopStringCriteria(tokenizer, ['```'])]
        )
        outputs = outputs[:, batch['input_ids'].shape[-1]:]
        decoded = tokenizer.batch_decode(outputs)
        print(prompt)
        for completion in decoded:
            if '```' in completion:
                completion = completion[:completion.index('```')]
        
            print(completion)
        print('------------')

