import lightning
import torch
from transformers import PreTrainedTokenizerFast

class DPOLightningModel(lightning.LightningModule):

    def __init__(self,
                 model: torch.nn.Module,
                 ref_model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizerFast,
                 kl_beta: float,
                 lr: float,
                 num_train_steps: int,
                 warm_up_steps: int
        ):
        super().__init__()
        self.model = model
        self.ref_model = ref_model
        self.kl_beta = kl_beta
        self.learning_rate = lr
        self.num_train_steps = num_train_steps
        self.warm_up_steps = warm_up_steps

        self.tokenizer = tokenizer

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW({"params": list(self.model.parameters()), "weight_decay": 0.0},
                                      lr=self.learning_rate,
                                      betas=(0.9, 0.95)) # TODO extract to command line args

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

    def _compute_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        targets = input_ids[:, 1:].unsqueeze(-1)

        logits = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        use_cache=False).logits[:, :-1, :]
        # logits has shape [batch, seq - 1, vocab_size]
        logprobs = torch.log_softmax(logits, dim=-1).gather(2, targets).squeeze(-1)
        return logprobs

    def training_step(self, batch, batch_idx):
        self.ref_model.eval()
        with torch.no_grad():
            context_lengths = torch.sum(batch['attention_mask_context'], dim=-1)
            seq_lengths_chosen = torch.sum(batch['attention_mask_chosen'], dim=-1)
            seq_lengths_rejected = torch.sum(batch['attention_mask_rejected'], dim=-1)
            completion_lengths_chosen = seq_lengths_chosen - context_lengths
            completion_lengths_rejected = seq_lengths_rejected - context_lengths
            
            ref_lps_chosen = self._compute_logprob_sum(batch["input_ids_chosen"],
                                                       batch["attention_mask_chosen"],
                                                       self.ref_model,
                                                       completion_lengths_chosen)
            ref_lps_rejected = self._compute_logprob_sum(batch["input_ids_rejected"],
                                                       batch["attention_mask_rejected"],
                                                       self.ref_model,
                                                       completion_lengths_rejected)
        
        pi_lps_chosen = self._compute_logprob_sum(batch["input_ids_chosen"],
                                                    batch["attention_mask_chosen"],
                                                    self.model,
                                                    completion_lengths_chosen)
        pi_lps_rejected = self._compute_logprob_sum(batch["input_ids_rejected"],
                                                    batch["attention_mask_rejected"],
                                                    self.model,
                                                    completion_lengths_rejected)

        logprob_ratio_delta = (pi_lps_chosen - pi_lps_rejected) - (ref_lps_chosen - ref_lps_rejected)
        loss = -torch.mean(torch.nn.functional.logsigmoid(self.kl_beta * logprob_ratio_delta))

        with torch.no_grad():
            pi_margin = torch.mean(pi_lps_chosen - pi_lps_rejected)
            self.log('pi_margin', pi_margin, on_step=True, sync_dist=True, logger=True, prog_bar=True)

            ref_margin = torch.mean(ref_lps_chosen - ref_lps_rejected)
            self.log('ref_margin', ref_margin, on_step=True, sync_dist=True, logger=True, prog_bar=True)

            margin_delta = pi_margin - ref_margin
            self.log('margin_delta', margin_delta, on_step=True, sync_dist=True, logger=True, prog_bar=True)

            dpo_accuracy = float(torch.mean(((pi_lps_chosen - ref_lps_chosen) - (pi_lps_rejected - ref_lps_rejected) > 0).float()).cpu().numpy())
            self.log('dpo_accuracy', dpo_accuracy, on_step=True, sync_dist=True, logger=True, prog_bar=True)

            self.log('train_loss', loss, on_step=True, sync_dist=True, logger=True, prog_bar=True)
        
            self.log_learning_rate()
        return loss
    