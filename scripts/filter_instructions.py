import argparse
import json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to filter (instruction, verifiers, test_cases) tuples for self-consistency')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of generate_instruction_artifacts.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write filtered (instruction, verifiers, test_cases) tuples')
    return parser.parse_args()

def can_compile(function_str: str) -> bool:
    try:
        exec(function_str) # TODO in sandbox
        return True
    except:
        return False
    
def parse_test_cases(test_group_strs: list[str]) -> list[tuple[str, bool]]:
    res = []
    for group_str in test_group_strs:
        # group_str should contain multiple newline-delimited tests, each in JSON format
        test_strs = group_str.splitlines()
        for test_str in test_strs:
            try:
                test = json.loads(test_str)
                if ('response' not in test) or ('result' not in test):
                    continue
                response = test['response']
                if not isinstance(response, str):
                    continue
                result = test['result']
                if not isinstance(result, bool):
                    continue
            except:
                # model produced invalid json; skip it
                continue
            res.append((response, result))

    return res

def passes(verifier: str, input_str: str, result: bool) -> bool:
    exec(verifier)
    try:
        # the function 'evaluate' is parsed from the verifier
        return evaluate(input_str) == result
    except:
        return False


def passes_validation(instruction: str, verifiers: list[str], testcases: list[str]) -> tuple[dict[str, object], bool]:
    # First filter the verifiers to those that compile; # TODO is that how the paper does it?
    verifiers = list(filter(can_compile, verifiers))
    if not verifiers:
        return {}, False

    # parse test cases into tuples
    test_cases = parse_test_cases(testcases) # TODO use better naming for test_cases vs testcases
    if not test_cases:
        return {}, False

    # remove test cases that don't pass at least one verifier
    filtered_test_cases = []
    for input_str, result in test_cases:
        ok = False
        for verifier in verifiers:
            if passes(verifier, input_str, result):
                ok = True
                break
        if ok:
            filtered_test_cases.append((input_str, result))
    test_cases = filtered_test_cases
    num_test_cases = len(test_cases)
    if not num_test_cases:
        return {}, False
    
    # filter verifiers to those that pass at least 80% of test cases
    filtered_verifiers = []
    for verifier in verifiers:
        num_passed = 0
        for input_str, result in test_cases:
            if passes(verifier, input_str, result):
                num_passed += 1
        pass_rate = num_passed / num_test_cases

        pass_rate = compute_verifier_pass_rate(verifier, test_cases)
        if pass_rate >= 0.8:
            filtered_verifiers.append(verifier)
    verifiers = filtered_verifiers

    # TODO note!!! Verifiers and test cases have been parsed in the returned object
    return {
        'instruction': instruction,
        'verifiers': verifiers,
        'testcases': test_cases
    }, True
        

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        orig_instances = [json.loads(line) for line in f.read().splitlines()]

    filtered_instances = []
    for instance in orig_instances:
        filtered_instance, ok = passes_validation(**instance)
        if ok:
            filtered_instances.append(filtered_instance)
        