import argparse
import json
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, ALL_COMPLETED, wait
import multiprocessing
import os
import signal

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

def process_instance(instance: dict[str, object]) -> dict[str, object]:
    # load completions
    completions = instance['completions']
    num_passes = [0] * len(completions)

    # process each verifier
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
    scores = {}
    num_verifiers = len(instance['verifiers'])
    for completion_idx, completion in enumerate(completions):
        pass_rate = num_passes[completion_idx] / num_verifiers
        scores[completion] = num_passes[completion_idx]
        print(f'Pass rate: {pass_rate}')
        if pass_rate >= 0.5:
            chosen.append(completion)
        elif pass_rate == 0:
            rejected.append(completion)
    
    print(f'Processed instance with {len(chosen)} chosen and {len(rejected)} rejected completions.')
    processed_instance = {
        'query': instance['query'],
        'instruction': instance['instruction'],
        'verifiers': instance['verifiers'],
        'chosen': chosen,
        'rejected': rejected,
        'scores': scores
    }
    return processed_instance

if __name__ == '__main__':
    args = parse_args()
    proceed = input('This script executes model-generated Python code. Please confirm that you are running this script in a sandboxed environment. (y/N)')
    if proceed != 'y':
        print('Aborting.')
        exit()

    with open(args.input) as f:
        instances = [json.loads(line) for line in f.read().splitlines()]

    with ProcessPoolExecutor(16) as executor:
        futures = []
        for instance in instances:
            future = executor.submit(process_instance, instance)
            futures.append(future)

        # wait at most 3 minutes
        done, not_done = wait(futures, timeout=180, return_when=ALL_COMPLETED)
        print('Done waiting.')
        
        sorted_instances = []
        for future in done:
            sorted_instances.append(future.result())
        
        num_pairs = 0
        num_unique_completions = 0
        num_unique_prompts = 0
        num_chosen_completions = 0
        num_rejected_completions = 0
        for instance in sorted_instances:
            chosen = instance['chosen']
            rejected = instance['rejected']
            if min(len(chosen), len(rejected)) > 0:
                num_pairs += max(len(chosen), len(rejected))
                num_unique_completions += len(chosen) + len(rejected)
                num_unique_prompts += 1
            num_chosen_completions += len(chosen)
            num_rejected_completions += len(rejected)

        print(f'Processed {len(instances)} instances.')
        print(f'Generated {num_pairs} pairs, {num_unique_completions} unique completions, sourced from {num_unique_prompts} unique instances.')
        print(f'Chosen completions: {num_chosen_completions}')
        print(f'Rejected completions: {num_rejected_completions}')

        with open(args.output, 'w') as output_file:
            for instance in sorted_instances:
                output_file.write(json.dumps(instance))
                output_file.write('\n')

        # kill straggler processes
        for proc in multiprocessing.active_children():
            print(f'Killing stalled process: {proc.pid}')
            os.kill(proc.pid, signal.SIGTERM)

    