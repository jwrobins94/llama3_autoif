import lightning.pytorch
import lightning.pytorch.loggers
from core.model import load_model
from core.tokenizer import load_tokenizer
from core.dpo_lightning_model import DPOLightningModel
import argparse
import json
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from data.dpo_data_collator import DPODataCollator
import lightning
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import WandbLogger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to generate completions for each instruction')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--deepspeed', default=False, action='store_true', help='Enables DeepSpeed Inference')
    parser.add_argument(f'--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument(f'--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument(f'--kl-beta', type=float, default=0.1, help='KL beta')
    parser.add_argument(f'--lr', type=float, default=1e-5, help='Peak learning rate')
    parser.add_argument(f'--warm-up-steps', type=int, default=1, help='Number of steps for linear LR warm-up')

    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of sort_completions.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write the final model checkpoint')
    return parser.parse_args()

# TODO extract this out & cite source from my old project


def construct_dataloader(tokenizer: PreTrainedTokenizerFast, rows: list[dict[str, object]], context_length: int, batch_size: int) -> DataLoader:
    rows_tokenized = []
    for row in rows:
        prompt = f'{row["query"]}\n{row["instruction"]}'
        
        for chosen, rejected in zip(row['chosen'], row['rejected']):
            # unlike the paper, we don't repeat any completions here and instead pick the shortest of the two
            messages_chosen = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': chosen}
            ]
            chosen_tokens = tokenizer.apply_chat_template(
                messages_chosen,
                tokenize=True,
                return_dict=True
            )

            messages_rejected = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': rejected}
            ]
            rejected_tokens = tokenizer.apply_chat_template(
                messages_rejected,
                tokenize=True,
                return_dict=True
            )

            messages_context = [
                {'role': 'user', 'content': prompt}
            ]
            context_tokens = tokenizer.apply_chat_template(
                messages_context,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True
            )
            
            row_tokenized = {}
            for k, v in chosen_tokens.items():
                row_tokenized[f'{k}_chosen'] = v
            for k, v in rejected_tokens.items():
                row_tokenized[f'{k}_rejected'] = v
            for k, v in context_tokens.items():
                row_tokenized[f'{k}_context'] = v

            rows_tokenized.append(row_tokenized)

    collator = DPODataCollator(tokenizer, context_length)
    train_dataloader = DataLoader(rows_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)
    print(f'Number of batches: {len(train_dataloader)}')

    return train_dataloader


if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        lines = f.read().splitlines()
        data = list(map(json.loads, lines))

    tokenizer = load_tokenizer(args.hf_api_token)

    dataloader = construct_dataloader(tokenizer, data, args.context_length, args.batch_size)

    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict
    # load the model a second time as our reference policy for the KL penalty
    ref_model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    lightning_model = DPOLightningModel(
        model,
        ref_model,
        tokenizer,
        args.kl_beta,
        args.lr,
        len(dataloader) * args.epochs,
        args.warm_up_steps
    )

    logger = WandbLogger()

    trainer = lightning.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=args.epochs,
        accumulate_grad_batches=1, # TODO
        precision='bf16', # TODO
        strategy='deepspeed_stage_2' if args.deepspeed else 'auto',
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False
    )

    trainer.fit(model=lightning_model, train_dataloaders=dataloader)

    @rank_zero_only
    def save(model):
        torch.save(model.state_dict(), args.output)
    save(model)
