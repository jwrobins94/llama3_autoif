import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedTokenizerFast
import nltk
from typing import Optional

@torch.no_grad()
def run_ifeval(model_name: str,
               tokenizer: PreTrainedTokenizerFast,
               batch_size: int,
               context_length: int,
               hf_api_token: str,
               limit: Optional[int] = None,
               state_dict: Optional[dict[str, object]] = None) -> tuple[dict[str, object] | None, list[object] | None]:
    # this download is needed for ifeval to run
    nltk.download('punkt_tab')

    print(f'State dict: {state_dict}')
    result = simple_evaluate(
        model=HFLM(
            pretrained=model_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            dtype=torch.bfloat16,
            state_dict=state_dict,
            token=hf_api_token,
            max_length=context_length
        ),
        tasks=['ifeval'],
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
    scores = result['results']['ifeval']
    samples = result['samples']['ifeval']

    return scores, samples