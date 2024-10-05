from transformers import PreTrainedTokenizerFast

PREFIXES = ['chosen', 'rejected', 'context']

class DPODataCollator:

    def __init__(self, tokenizer: PreTrainedTokenizerFast, context_length: int):
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __call__(self, features: list[dict[str, object]]) -> dict[str, object]:
        feature_groups = {group_name: [] for group_name in PREFIXES}

        # Separate out {input_ids, attention_mask} for chosen, rejected, and context strings
        for feature in features:
            for group_name in feature_groups.keys():
                feature_groups[group_name].append({
                    {
                        "input_ids": feature[f"input_ids_{group_name}"],
                        "attention_mask": feature[f"attention_mask_{group_name}"],
                    }
                })

        # tokenize each group and remap back to the original names
        res = {}
        for group_name, rows in feature_groups.items():
            batch = self.tokenizer.pad(
                rows,
                padding=True,
                max_length=self.context_length,
                return_tensors='pt',
            )
            for k, v in batch.items():
                res[f"{k}_{group_name}"] = v
        
        return res