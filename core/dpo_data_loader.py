from transformers import PreTrainedTokenizerFast
from torch.utils.data.dataloader import DataLoader
from core.dpo_data_collator import DPODataCollator
from itertools import cycle
import random

def construct_dpo_dataloader(tokenizer: PreTrainedTokenizerFast, rows: list[dict[str, object]], context_length: int, batch_size: int) -> DataLoader:
    rows_tokenized = []
    random.seed(1234)
    for row in rows:
        prompt = f'{row["query"]}\n{row["instruction"]}'
        
        # As in the paper, we loop the shorter of the two lists to ensure that all generated completions are used at least once.
        if len(row['chosen']) > len(row['rejected']):
            zip_list_old = list(zip(row['chosen'], cycle(row['rejected'])))
        else:
            zip_list_old = list(zip(cycle(row['chosen']), row['rejected']))

        score_entries = []
        for completion, score in row['scores'].items():
            score_entries.append((score, completion))
        score_entries.sort()

        zip_list = []
        usage_counts = {}
        print(score_entries)
        for score, chosen in score_entries:
            eligible = [x for x in score_entries if x[0] < score] # find everything with a lower score
            if not eligible:
                continue
            min_usage = min((usage_counts.get(x[1], 0) for x in eligible)) # compute min usage across those
            eligible = [x for x in eligible if usage_counts.get(x[1], 0) == min_usage] # filter to min usage
            if not eligible:
                continue
            rejected = random.choice(eligible)[1] # pick a random one
            usage_counts[chosen] = usage_counts.get(chosen, 0) + 1
            usage_counts[rejected] = usage_counts.get(rejected, 0) + 1
            zip_list.append((chosen, rejected))
        
        if not (len(zip_list) >= len(zip_list_old)):
            print(zip_list_old)
            print(zip_list)
            raise ValueError('')


        for chosen, rejected in zip_list:
            messages_chosen = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': chosen}
            ]
            chosen_tokens = tokenizer.apply_chat_template(
                messages_chosen,
                tokenize=True,
                return_dict=True,
                max_length=context_length
            )

            messages_rejected = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': rejected}
            ]
            rejected_tokens = tokenizer.apply_chat_template(
                messages_rejected,
                tokenize=True,
                return_dict=True,
                max_length=context_length
            )

            messages_context = [
                {'role': 'user', 'content': prompt}
            ]
            context_tokens = tokenizer.apply_chat_template(
                messages_context,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                max_length=context_length
            )
            
            row_tokenized = {}
            for k, v in chosen_tokens.items():
                row_tokenized[f'{k}_chosen'] = v
            for k, v in rejected_tokens.items():
                row_tokenized[f'{k}_rejected'] = v
            for k, v in context_tokens.items():
                row_tokenized[f'{k}_context'] = v

            rows_tokenized.append(row_tokenized)

    collator = DPODataCollator(tokenizer, context_length)
    train_dataloader = DataLoader(rows_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)
    return train_dataloader