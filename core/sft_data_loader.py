from transformers import PreTrainedTokenizerFast
from torch.utils.data.dataloader import DataLoader
from core.sft_data_collator import SFTDataCollator

def construct_sft_dataloader(
        tokenizer: PreTrainedTokenizerFast,
        rows: list[dict[str, object]],
        context_length: int,
        batch_size: int,
        chosen_threshold: float = 1.0,
    ) -> DataLoader:
    rows_tokenized = []
    for row in rows:
        prompt = f'{row["query"]}\n{row["instruction"]}'
        
        all_chosen = []
        for completion, score in row['scores'].items():
            if score >= chosen_threshold:
                all_chosen.append(completion)
        
        for chosen in all_chosen:
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
            for k, v in context_tokens.items():
                row_tokenized[f'{k}_context'] = v

            rows_tokenized.append(row_tokenized)

    collator = SFTDataCollator(tokenizer, context_length)
    train_dataloader = DataLoader(rows_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)
    return train_dataloader