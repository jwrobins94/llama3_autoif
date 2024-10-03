import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedTokenizerFast
import nltk
import pandas as pd

@torch.no_grad()
def run_ifeval(model: torch.nn.Module, tokenizer: PreTrainedTokenizerFast, limit=None) -> tuple[dict[str, object], list[object]]:
    # this download is needed for ifeval to run
    nltk.download('punkt_tab')

    model.eval()
    result = simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=10),
        tasks=['ifeval'],
        device=model.device,
        cache_requests=True,
        log_samples=True,
        limit=limit,
        num_fewshot=0,
        batch_size=10, # TODO ensure that results are correct when using larger batch sizes
        apply_chat_template=True
    )
    scores = result['results']['ifeval']
    samples = result['samples']['ifeval']

    return scores, samples