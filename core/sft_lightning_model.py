import lightning
import torch
from transformers import PreTrainedTokenizerFast

class SFTLightningModel(lightning.LightningModule):

    def __init__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizerFast,
                 lr: float,
                 num_train_steps: int,
                 warm_up_steps: int,
                 beta1: float,
                 beta2: float
        ):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.num_train_steps = num_train_steps
        self.warm_up_steps = warm_up_steps

        self.tokenizer = tokenizer
        self.beta1 = beta1
        self.beta2 = beta2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate,
                                      betas=(self.beta1, self.beta2))

        # Calculate total training steps
        num_devices = self.trainer.num_devices
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, 1e-7 / self.learning_rate, 1.0, total_iters=self.warm_up_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min = 0.1 * self.learning_rate, T_max=self.num_train_steps // num_devices - self.warm_up_steps)
        ], milestones=[self.warm_up_steps])
        return [optimizer], {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
    
    def log_learning_rate(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, on_step=True, logger=True, prog_bar=True, rank_zero_only=True)

    def _compute_logprob_sum(self,
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor,
                             model: torch.nn.Module,
                             completion_lengths: torch.Tensor) -> torch.Tensor:
        targets = input_ids[:, 1:].unsqueeze(-1)

        logits = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        use_cache=False).logits[:, :-1]
        logprobs = torch.log_softmax(logits, dim=-1).gather(2, targets).squeeze(-1)

        per_row_sums = []
        for i, completion_length in enumerate(completion_lengths):
            per_row_sums.append(torch.sum(logprobs[i, -completion_length:]))
        res = torch.stack(per_row_sums)
        return res

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            context_lengths = torch.sum(batch['attention_mask_context'], dim=-1)
            seq_lengths_chosen = torch.sum(batch['attention_mask_chosen'], dim=-1)
            completion_lengths_chosen = seq_lengths_chosen - context_lengths   
        
        pi_lps_chosen = self._compute_logprob_sum(batch["input_ids_chosen"],
                                                    batch["attention_mask_chosen"],
                                                    self.model,
                                                    completion_lengths_chosen)

        loss = -torch.sum(pi_lps_chosen) / torch.sum(completion_lengths_chosen)
        
        self.log('loss', loss, on_step=True, sync_dist=True, logger=True, prog_bar=True)
        return loss
    