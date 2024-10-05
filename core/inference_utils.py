import torch
from transformers import PreTrainedTokenizerFast, StopStringCriteria

def wrap_with_deepspeed_inference(model: torch.nn.Module) -> torch.nn.Module:
    import deepspeed
    #deepspeed.init_distributed()
    ds_engine = deepspeed.init_inference(model,
                                dtype=torch.bfloat16,
                                #injection_policy={LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')},
                                tensor_parallel={"tp_size": torch.cuda.device_count()},
                                checkpoint=None, # TODO load checkpoint from args
                                )
    return ds_engine.module

def generate_completions(model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, prompts: list[str], stop_str: str, max_tokens: int) -> list[str]:
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