from transformers import AutoConfig, PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch
from typing import Optional

# Keeping the scope small to start.
# It is safe to include any models here that share a tokenizer with the Llama 3 series.
ALLOWED_MODELS = [
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-3.2-1B-Instruct'
]

def load_state_dict(ckpt: str) -> dict[str, torch.Tensor]:
    print(f'Reloading weights from ckpt: {ckpt}')
    checkpoint = torch.load(ckpt)
    for state_dict_key in ['state_dict', 'module']:
        if state_dict_key in checkpoint:
            checkpoint = checkpoint[state_dict_key]
            break
    return checkpoint

def load_model(model_name: str, tokenizer: PreTrainedTokenizerFast, context_length: int, hf_api_token: str, state_dict_path: Optional[str] = None):   
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'model name {model_name} is not one of the allowed values {ALLOWED_MODELS}')
    
    model_config = AutoConfig.from_pretrained(
        model_name,
        n_ctx=context_length,
        token=hf_api_token
    )

    if state_dict_path:
        state_dict = load_state_dict(state_dict_path)
        pretrained_name = None # don't reload weights since we're going to overwrite them anyways
    else:
        state_dict = None
        pretrained_name = model_name

    model = LlamaForCausalLM.from_pretrained(pretrained_name, config=model_config, token=hf_api_token,
                                                state_dict=state_dict)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model
