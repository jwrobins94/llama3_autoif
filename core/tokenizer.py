from transformers import AutoTokenizer, PreTrainedTokenizerFast

# We take the tokenizer from any Llama 3 Instruct model
# model.py enforces that only matching models are loaded
TOKENIZER_MODEL_ID = 'meta-llama/Llama-3.2-1B-Instruct'

def load_tokenizer(hf_api_token: str):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, token=hf_api_token, legacy=False)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # we need a pad token with left padding in order to support batched generation
    # just pick something; the choice of pad token doesn't matter since we'll mask out these tokens in the attention block
    tokenizer.pad_token = '<|finetune_right_pad_id|>'
    tokenizer.padding_side = 'left'

    return tokenizer