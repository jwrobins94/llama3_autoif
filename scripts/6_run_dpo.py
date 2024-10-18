import lightning.pytorch
import lightning.pytorch.loggers
from core.model import load_model
from core.tokenizer import load_tokenizer
from core.dpo_lightning_model import DPOLightningModel
import argparse
import json
from core.dpo_data_loader import construct_dpo_dataloader
import lightning
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import WandbLogger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to run DPO')
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. "meta-llama/Llama-3.1-8B-Instruct"')
    parser.add_argument('--hf-api-token', type=str, required=True, help='HuggingFace API token')
    
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path for trained model checkpoint')
    parser.add_argument(f'--context-length', type=int, default=2048, help='Context length')
    parser.add_argument(f'--strategy', type=str, default='deepspeed_stage_2', help='Distributed training strategy')
    parser.add_argument(f'--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument(f'--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument(f'--grad-acc-steps', type=int, default=1, help='Number of steps for gradient accumulation')
    parser.add_argument(f'--kl-beta', type=float, default=0.1, help='KL beta')
    parser.add_argument(f'--lr', type=float, default=5e-6, help='Peak learning rate')
    parser.add_argument(f'--beta1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument(f'--beta2', type=float, default=0.95, help='AdamW beta2')
    parser.add_argument(f'--warm-up-steps', type=int, default=1, help='Number of steps for linear LR warm-up')
    parser.add_argument('--include-chosen-nll-loss', action='store_true', help='If true, include an additional NLL loss term on the chosen response', default=False)
    parser.add_argument('--nll-loss-weight', type=float, default=1.0, help='Loss multiplier for NLL loss (if enabled)')

    parser.add_argument(f'--chosen-threshold', type=float, default=0.5, help='Chosen responses will have pass rate >= threshold')
    parser.add_argument(f'--rejected-threshold', type=float, default=0.0, help='Rejected responses will have pass rate < threshold (or == threshold if threshold is 0)')
    parser.add_argument('--max-loop', type=int, default=None, help='Optional max number of times to loop the smaller of (chosen, rejected). Used to control the number of times a completion is reused.')

    parser.add_argument(f'--input', type=str, required=True, help='Path to the output of 5_sort_completions.py')
    parser.add_argument(f'--output', type=str, required=True, help='Path to write the final model checkpoint')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    with open(args.input) as f:
        lines = f.read().splitlines()
        data = list(map(json.loads, lines))

    tokenizer = load_tokenizer(args.hf_api_token)

    dataloader = construct_dpo_dataloader(
        tokenizer,
        data,
        args.context_length,
        args.batch_size,
        args.max_loop,
        args.chosen_threshold,
        args.rejected_threshold
    )
    print(f'Number of batches: {len(dataloader)}')

    model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token, args.ckpt)
    # load the model a second time as our reference policy for the KL penalty
    ref_model = load_model(args.model, tokenizer, args.context_length, args.hf_api_token, args.ckpt)

    lightning_model = DPOLightningModel(
        model,
        ref_model,
        tokenizer,
        args.kl_beta,
        args.lr,
        len(dataloader) * args.epochs // args.grad_acc_steps,
        args.warm_up_steps,
        args.beta1,
        args.beta2,
        args.include_chosen_nll_loss,
        args.nll_loss_weight
    )

    logger = WandbLogger()

    trainer = lightning.Trainer(
        max_epochs=args.epochs,
        precision='bf16', # hardcode to bf16 since the model itself is loaded in bf16
        strategy=args.strategy,
        logger=logger,
        accumulate_grad_batches=args.grad_acc_steps,
        log_every_n_steps=1,
        enable_checkpointing=False
    )

    trainer.fit(model=lightning_model, train_dataloaders=dataloader)

    @rank_zero_only
    def save(model):
        torch.save(model.state_dict(), args.output)
    save(model)
