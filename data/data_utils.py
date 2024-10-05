from datasets import load_dataset

def extract_query(row) -> dict[str, str]:
    conversations = row['conversations']
    if not conversations:
        return {'query': ''}
    value = conversations[0]['value']
    if not value:
        return {'query': ''}
    return {'query': value}

def load_sharegpt_queries() -> list[str]:
    # TODO allow picking the dataset from command line
    data = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', data_files='ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json')
    data_train = data['train']
    data_train = data_train.map(
        extract_query,
        batched=False,
        remove_columns=data_train.column_names,
        #load_from_cache_file=False # the cache creates a lot of problems when debugging
    ).filter(lambda row : 5 <= len(row['query']) <= 200)

    res = []
    for row in data_train:
        res.append(row['query'])
    return res