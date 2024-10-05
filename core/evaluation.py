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
               limit: Optional[int] = None,
               state_dict: Optional[dict[str, object]] = None) -> tuple[dict[str, object], list[object]]:
    # this download is needed for ifeval to run
    nltk.download('punkt_tab')

    result = simple_evaluate(
        model=HFLM(
            pretrained=model_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            parallelize=True
        ),
        tasks=['ifeval'],
        cache_requests=True,
        log_samples=True,
        limit=limit,
        num_fewshot=0,
        batch_size=batch_size,
        apply_chat_template=True,
        state_dict=state_dict
    )
    scores = result['results']['ifeval']
    samples = result['samples']['ifeval']

    return scores, samples