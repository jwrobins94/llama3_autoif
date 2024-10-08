import torch
from transformers import PreTrainedTokenizerFast, StopStringCriteria

@torch.inference_mode()
def generate_completions(model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, prompts: list[str], stop_strings: list[str], max_tokens: int) -> list[str]:
    batch = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left') # left padding so that completions are all at the end
    batch.to(model.device)

    outputs = model.generate(
        **batch,
        max_new_tokens=max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=True,
        temperature=1.0,
        stopping_criteria=[StopStringCriteria(tokenizer, stop_strings)]
    )
    outputs = outputs[:, batch['input_ids'].shape[-1]:]
    decoded = tokenizer.batch_decode(outputs)
    for i in range(len(decoded)):
        for stop_str in stop_strings:
            if stop_str in decoded[i]:
                decoded[i] = decoded[i][:decoded[i].index(stop_str)]
    return decoded