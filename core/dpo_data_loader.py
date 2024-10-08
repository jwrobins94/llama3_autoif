from transformers import PreTrainedTokenizerFast
from torch.utils.data.dataloader import DataLoader
from core.dpo_data_collator import DPODataCollator
from itertools import cycle

def construct_dpo_dataloader(
        tokenizer: PreTrainedTokenizerFast,
        rows: list[dict[str, object]],
        context_length: int,
        batch_size: int,
        no_loop: bool
    ) -> DataLoader:
    rows_tokenized = []
    for row in rows:
        prompt = f'{row["query"]}\n{row["instruction"]}'
        
        if no_loop:
            zip_list = zip(row['chosen'], row['rejected'])
        else:
            # As in the paper, we loop the shorter of the two lists to ensure that all generated completions are used at least once.
            if len(row['chosen']) > len(row['rejected']):
                zip_list = zip(row['chosen'], cycle(row['rejected']))
            else:
                zip_list = zip(cycle(row['chosen']), row['rejected'])
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