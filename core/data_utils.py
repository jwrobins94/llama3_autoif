from datasets import load_dataset
import glob
import re

HH_DELIMITERS = ["\n\nHuman: ", "\n\nAssistant: "]
HH_PATTERN = '|'.join(map(re.escape, HH_DELIMITERS))

def extract_query_hh(row: str) -> str:
    query = re.split(HH_PATTERN, row['chosen'], maxsplit=2)[1]
    return {'query': query}

def load_hh_queries(min_query_len: int = 5, max_query_len: int = 200) -> list[str]:
    data_train = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")['train']
    data_train = data_train.map(
        extract_query_hh,
        batched=False,
        remove_columns=data_train.column_names,
        load_from_cache_file=False # the cache creates a lot of problems when debugging
    ).filter(lambda row : min_query_len <= len(row['query']) <= max_query_len)

    return [row['query'] for row in data_train]

def extract_query_sharegpt(row) -> dict[str, str]:
    conversations = row['conversations']
    if not conversations:
        return {'query': ''}
    value = conversations[0]['value']
    if not value:
        return {'query': ''}
    return {'query': value}

def load_sharegpt_queries(min_query_len: int = 5, max_query_len: int = 200) -> list[str]:
    data = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', data_files='ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json')
    data_train = data['train']
    # TODO: filter out ~2% of data that is not from a human
    data_train = data_train.map(
        extract_query_sharegpt,
        batched=False,
        remove_columns=data_train.column_names,
        load_from_cache_file=False # the cache creates a lot of problems when debugging
    ).filter(lambda row : min_query_len <= len(row['query']) <= max_query_len)

    return [row['query'] for row in data_train]

def merge_outputs(output_prefix: str, suffix=".jsonl"):
    all_results = []
    for path in glob.glob(f'{output_prefix}-*{suffix}'):
        with open(path) as f:
            local_data = f.read()
            all_results.append(local_data)
    
    with open(f'{output_prefix}{suffix}', 'w') as f:
        f.write(''.join(all_results))