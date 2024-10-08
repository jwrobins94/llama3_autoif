import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedTokenizerFast
import nltk
from typing import Optional

@torch.inference_mode()
def run_eval(benchmark_name: str,
             model_name: str,
             tokenizer: PreTrainedTokenizerFast,
             batch_size: int,
             context_length: int,
             hf_api_token: str,
             limit: Optional[int] = None,
             state_dict: Optional[dict[str, object]] = None) -> tuple[dict[str, object] | None, list[object] | None]:
    if benchmark_name == 'ifeval':
        # this download is needed for ifeval to run
        nltk.download('punkt_tab')
    elif benchmark_name == 'hellaswag':
        pass
    else:
        return ValueError('Only ifeval and hellaswag have been tested')
    
    model = HFLM(
            pretrained=model_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            dtype=torch.bfloat16,
            token=hf_api_token,
            max_length=context_length
        )
    if state_dict:
        # reload weights from the state_dict
        # TODO: debug why weights are not properly loaded when passing state_dict=state_dict directly to HFLM(...)
        model.model.load_state_dict(state_dict)
    result = simple_evaluate(
        model=model,
        tasks=[benchmark_name],
        cache_requests=True,
        log_samples=True,
        limit=limit,
        num_fewshot=0,
        batch_size=batch_size,
        apply_chat_template=True
    )
    if result is None:
        # rank > 0
        return None, None 
    scores = result['results'][benchmark_name]
    samples = result['samples'][benchmark_name]

    return scores, samples
