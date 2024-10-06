import argparse
import json
from typing import Callable

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to filter (instruction, verifiers, test_cases) tuples for self-consistency')
    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of 4_generate_completions.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write filtered (instruction, verifiers, test_cases) tuples')
    return parser.parse_args()

def passes(verifier_fn: Callable[[str], bool], input_str: str, result: bool) -> bool:
    try:
        return verifier_fn(input_str) == result
    except Exception as e:
        return False
        
if __name__ == '__main__':
    args = parse_args()
    proceed = input('This script executes model-generated Python code. Please confirm that you are running this script in a sandboxed environment. (y/N)')
    if proceed != 'y':
        print('Aborting.')
        exit()

    with open(args.input) as f:
        instances = [json.loads(line) for line in f.read().splitlines()]

    sorted_instances = []
    num_pairs = 0
    for instance in instances:
        # load completions
        completions = instance['completions']
        num_passes = [0] * len(completions)

        # process each verifier
        verifier_functions = []
        for function_str in instance['verifiers']:
            exec(function_str, globals())
            for completion_idx, completion in enumerate(completions):
                try:
                    ok = evaluate(completion) # evaluate is loaded dynamically via exec
                except:
                    ok = False
                if ok:
                    num_passes[completion_idx] += 1        
        chosen = []
        rejected = []
        num_verifiers = len(instance['verifiers'])
        for completion_idx, completion in enumerate(completions):
            pass_rate = num_passes[completion_idx] / num_verifiers
            print(f'Pass rate: {pass_rate}')
            if pass_rate >= 0.5:
                chosen.append(completion)
            elif pass_rate == 0:
                rejected.append(completion)
        num_pairs += min(len(chosen), len(rejected))
        
        print(f'Writing out instance with {len(chosen)} chosen and {len(rejected)} rejected completions.')
        sorted_instances.append({
            'query': instance['query'],
            'instruction': instance['instruction'],
            'chosen': chosen,
            'rejected': rejected
        })
    print(f'Generated {num_pairs} pairs.')

    with open(args.output, 'w') as output_file:
        for instance in sorted_instances:
            output_file.write(json.dumps(instance))
            output_file.write('\n')