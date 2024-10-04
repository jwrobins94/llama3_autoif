import lightning.pytorch
import lightning.pytorch.loggers
from core.model import load_model
from core.tokenizer import load_tokenizer
from core.dpo_lightning_model import DPOLightningModel
import argparse
import json
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from typing import Optional
from dataclasses import dataclass
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
    parser.add_argument(f'--kl-beta', type=float, default=0.1, help='KL beta')
    parser.add_argument(f'--lr', type=float, default=1e-5, help='Peak learning rate')
    parser.add_argument(f'--warm-up-steps', type=int, default=1, help='Number of steps for linear LR warm-up')

    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of sort_completions.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write the final model checkpoint')
    return parser.parse_args()

# TODO extract this out & cite source from my old project
@dataclass
class RewardDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerFast
    padding: bool|str = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, object]]) -> dict[str, object]:
        features_context = []
        features_chosen = []
        features_rejected = []
        # check if we have a margin. If we do, we need to batch it as well
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
                or "attention_mask_context" not in feature
                or "attention_mask_context" not in feature
            ):
                raise ValueError("Missing column in batch")

            features_context.append(
                {
                    "input_ids": feature["input_ids_context"],
                    "attention_mask": feature["attention_mask_context"],
                }
            )
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_context = self.tokenizer.pad(
            features_context,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_context": batch_context["input_ids"],
            "attention_mask_context": batch_context["attention_mask"],
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch

def construct_dataloader(tokenizer: PreTrainedTokenizerFast, rows: list[dict[str, object]], batch_size: int) -> DataLoader:
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

    collator = RewardDataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(rows_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)
    print(f'Number of batches: {len(train_dataloader)}')

    return train_dataloader


if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        lines = f.read().splitlines()
        data = list(map(json.loads, lines))

    tokenizer = load_tokenizer(args.hf_api_token)

    dataloader = construct_dataloader(tokenizer, data, args.batch_size)

    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict
    # load the model a second time as our reference policy for the KL penalty
    ref_model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token) # TODO add support for state_dict

    lightning_model = DPOLightningModel(
        model,
        ref_model,
        args.kl_beta,
        args.lr,
        len(dataloader),
        args.warm_up_steps
    )

    logger = WandbLogger()
    trainer = lightning.Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=1, # TODO
        accumulate_grad_batches=1, # TODO
        precision='bf16-mixed', # TODO
        strategy='deepspeed_stage_2' if args.deepspeed else 'auto',
        logger=logger,
        log_every_n_steps=1
    )

    trainer.fit(model=lightning_model, train_dataloaders=dataloader)

    @rank_zero_only
    def save(model):
        torch.save(model.state_dict(), args.output)
    save(model)

    