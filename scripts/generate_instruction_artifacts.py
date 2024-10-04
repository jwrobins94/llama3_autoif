from core.model import load_model
from core.tokenizer import load_tokenizer
import argparse
import torch
import json
from transformers import StopStringCriteria, PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate test cases and verification functions')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--max-tokens', type=int, default=1024, help='Context length')

    parser.add_argument(f'--num-verifications', type=int, required=True, help='Number of verifiers per instruction')
    parser.add_argument(f'--deepspeed', default=False, action='store_true', help='Enables DeepSpeed Inference')

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
{{"response": "some response", "result": true or false}}

Here are 3 example test cases for the instruction: use the letter B at least once
{{"response": "Bar", "result": true}}
{{"response": "Foo", "result": false}}
{{"response": "CAB", "result": true}}

Now, write 3 test cases (one per line) for the following instruction: {instruction}'''


def generate_completions(model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, prompts: list[str], stop_str: str) -> list[str]:
    batch = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left') # left padding so that completions are all at the end
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
    decoded = tokenizer.batch_decode(outputs)
    for i in range(len(decoded)):
        if stop_str in decoded[i]:
            decoded[i] = decoded[i][:decoded[i].index(stop_str)]
    return decoded

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        instructions = f.read().splitlines()

    tokenizer = load_tokenizer(args.hf_api_token)
    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    if args.deepspeed:
        import deepspeed
        #deepspeed.init_distributed()
        ds_engine = deepspeed.init_inference(model,
                                 dtype=torch.bfloat16,
                                 #injection_policy={LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')},
                                 checkpoint=None, # TODO load checkpoint from args
                                 )
        model = ds_engine.module

    if torch.cuda.is_available():
        model.to('cuda:0')

    with open(args.output, 'w') as output_file:
        for instruction_idx, instruction in enumerate(instructions):
            print(f'Processing instruction {instruction_idx + 1} of {len(instructions)}.')
            messages_mat = [[{'role': 'user', 'content': construct_verifier_prompt(instruction)}] for _ in range(args.num_verifications)]
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

            completions = generate_completions(model, tokenizer, prompts, '```')
            verified_completions = [fn_prefix + completion for completion in completions]

            for i, completion in enumerate(completions):
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
            testcase_prefix = '{"response": "'
            for i, prompt in enumerate(prompts_2):
                prompts_2[i] = prompt + f'{testcase_prefix}'
            completions = generate_completions(model, tokenizer, prompts_2, tokenizer.eos_token)
            testcase_completions = [testcase_prefix + completion for completion in completions]
        
            obj = {
                'instruction': instruction,
                'verifiers': verified_completions,
                'testcases': testcase_completions
            }
            output_file.write(json.dumps(obj))
            output_file.write('\n')
            output_file.flush()

