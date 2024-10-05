import torch
from transformers import PreTrainedTokenizerFast, StopStringCriteria

@torch.inference_mode()
def generate_completions(model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, prompts: list[str], stop_str: str, max_tokens: int) -> list[str]:
    # TODO: Llama sometimes outputs <|eot|> when it shouln't, e.g. inside of a code block
    # We can make a slight improvement by accepting multiple stop strings to account for cases like this.
    
    batch = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left') # left padding so that completions are all at the end
    batch.to(model.device)

    outputs = model.generate(
        **batch,
        max_new_tokens=max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=True,
        temperature=1.0,
        stopping_criteria=[StopStringCriteria(tokenizer, [stop_str])]
    )
    outputs = outputs[:, batch['input_ids'].shape[-1]:]
    decoded = tokenizer.batch_decode(outputs)
    for i in range(len(decoded)):
        if stop_str in decoded[i]:
            decoded[i] = decoded[i][:decoded[i].index(stop_str)]
    return decoded