import argparse
import json
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import multiprocessing
import os
import signal

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to filter (instruction, verifiers, test_cases) tuples for self-consistency')
    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of 2_generate_instruction_artifacts.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write filtered (instruction, verifiers, test_cases) tuples')
    return parser.parse_args()

def can_compile(function_str: str) -> tuple[Optional[Callable[[str], bool]], bool]:
    try:
        exec(function_str, globals())
        local_evaluate = evaluate
        return local_evaluate, True # evaluate is generated dynamically from exec
    except:
        return None, False

def validate_testcase(test: dict[str, object]) -> bool:
    if ('response' not in test) or ('result' not in test):
        return False
    response = test['response']
    if not isinstance(response, str):
        return False
    result = test['result']
    if not isinstance(result, bool):
        return False
    return True

def parse_test_cases(test_group_strs: list[str]) -> list[tuple[str, bool]]:
    res = []
    for group_str in test_group_strs:
        # group_str should contain multiple newline-delimited tests, each in JSON format
        test_strs = group_str.splitlines()
        for test_str in test_strs:
            try:
                test = json.loads(test_str)
                if not validate_testcase(test):
                    print(f'Bad test case: {test}')
                    continue
            except:
                # model produced invalid json; skip it
                print(f'Bad test case: {test}')
                continue
            res.append((test['response'], test['result']))

    return res

def passes(verifier_fn: Callable[[str], bool], input_str: str, result: bool) -> bool:
    try:
        return verifier_fn(input_str) == result
    except Exception as e:
        return False


def passes_validation(query: str, instruction: str, verifiers: list[str], testcases: list[str]) -> tuple[dict[str, object], bool]:
    stats = {
        'verifier_count': len(verifiers),
        'testcase_str_count': len(testcases)
    }
   
    # First filter the verifiers to those that compile; # TODO is that how the paper does it?
    filtered_verifiers = []
    verifier_functions = []
    for verifier in verifiers:
        f, ok = can_compile(verifier)
        if ok:
            filtered_verifiers.append(verifier)
            verifier_functions.append(f)
        else:
            print(verifier)
    verifiers = filtered_verifiers
    print(f'Verifiers compile: {len(verifiers)}')
    stats['compiled_verifiers'] = len(verifiers)
    if not verifiers:
        return stats, False
    
    print(verifier_functions)

    # parse test cases into tuples
    test_cases = parse_test_cases(testcases) # TODO use better naming for test_cases vs testcases
    print(f'Test cases parse: {len(test_cases)}')
    stats['test_cases_parse'] = len(test_cases)
    if not test_cases:
        return stats, False
        
    # remove test cases that don't pass at least one verifier
    filtered_test_cases = []
    for input_str, result in test_cases:
        ok = False
        for verifier_fn in verifier_functions:
            if passes(verifier_fn, input_str, result):
                ok = True
                break
        if ok:
            filtered_test_cases.append((input_str, result))
    test_cases = filtered_test_cases
    num_test_cases = len(test_cases)
    print(f'Test cases pass at least one verifier: {num_test_cases}')
    stats['test_cases_pass_gte1'] = num_test_cases
    if not num_test_cases:
        return stats, False
    
    # filter verifiers to those that pass at least 80% of test cases
    filtered_verifiers = []
    total_passes = 0
    for verifier, verifier_fn in zip(verifiers, verifier_functions):
        num_passed = 0
        for input_str, result in test_cases:
            if passes(verifier_fn, input_str, result):
                num_passed += 1
        pass_rate = num_passed / num_test_cases
        if pass_rate >= 0.8:
            filtered_verifiers.append(verifier)
        total_passes += num_passed
    verifiers = filtered_verifiers
    print(f'Verifiers pass at least 80% of test cases: {len(verifiers)}')
    stats['verifiers_pass_80pct'] = len(verifiers)
    stats['total_passes'] = total_passes
    if not verifiers:
        return stats, False

    # TODO note!!! Verifiers and test cases have been parsed in the returned object
    stats.update({
        'query': query,
        'instruction': instruction,
        'test_cases': test_cases,
        'verifiers': verifiers,
        'unique_instances': 1
    })
    return stats, True
        

if __name__ == '__main__':
    args = parse_args()
    proceed = input('This script executes model-generated Python code. Please confirm that you are running this script in a sandboxed environment. (y/N)')
    if proceed != 'y':
        print('Aborting.')
        exit()

    with open(args.input) as f:
        orig_instances = [json.loads(line) for line in f.read().splitlines()]
    
    stats = {
        'verifier_count': 0,
        'testcase_str_count': 0,
        'compiled_verifiers': 0,
        'test_cases_parse': 0,
        'test_cases_pass_gte1': 0,
        'verifiers_pass_80pct': 0,
        'total_passes': 0,
        'unique_instances': 0
    }

    with ProcessPoolExecutor(16) as executor:
        futures = []
        for instance in orig_instances:
            future = executor.submit(
                passes_validation,
                instance['query'],
                instance['instruction'],
                instance['verifiers'],
                instance['testcases']
            )
            futures.append(future)
        
        filtered_instances = []

        # wait at most 3 minutes
        done, not_done = wait(futures, timeout=180, return_when=ALL_COMPLETED)
        print('Done waiting.')
        
        for future in done:
            filtered_instance, ok = future.result()
            if ok:
                filtered_instances.append(filtered_instance)
            for k in stats:
                stats[k] += filtered_instance.get(k, 0)
        print('Writing output.')

        with open(args.output, 'w') as output_file:
            for instance in filtered_instances:
                output_file.write(json.dumps(instance))
                output_file.write('\n')
        
        # kill straggler processes
        for proc in multiprocessing.active_children():
            print(f'Killing stalled process: {proc.pid}')
            os.kill(proc.pid, signal.SIGTERM)
    
        print(stats)
    
    