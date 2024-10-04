import lightning
import torch

class DPOLightningModel(lightning.LightningModule):

    def __init__(self, model: torch.nn.Module, ref_model: torch.nn.Module, kl_beta: float):
        super().__init__()
        self.model = model
        self.ref_model = ref_model
        self.kl_beta = kl_beta

    def _compute_logprob_sum(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, model: torch.nn.Module, completion_lengths: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        targets = input_ids[:, 1:].unsqueeze(-1)

        logits = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        use_cache=False).logits[:, :-1]
        logprobs = torch.log_softmax(logits, dim=-1).gather(2, targets).squeeze(-1)

        res = torch.zeros([batch_size], device=input_ids.device)
        for i, completion_length in enumerate(completion_lengths):
            res[i] = torch.sum(logprobs[i, -completion_length:])
        return res

    def compute_loss(self, batch) -> torch.Tensor:
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
        return loss, (pi_lps_chosen, ref_lps_chosen, pi_lps_rejected, ref_lps_rejected)

    @torch.no_grad()
    def log_metrics(self,
                    pi_lps_chosen: torch.Tensor,
                    ref_lps_chosen: torch.Tensor,
                    pi_lps_rejected: torch.Tensor,
                    ref_lps_rejected: torch.Tensor) -> None:
        self.log('pi_lps_chosen', torch.mean(pi_lps_chosen), on_step=True, sync_dist=True, logger=True, prog_bar=True)
        self.log('ref_lps_chosen', torch.mean(ref_lps_chosen), on_step=True, sync_dist=True, logger=True, prog_bar=True)
        self.log('pi_lps_rejected', torch.mean(pi_lps_rejected), on_step=True, sync_dist=True, logger=True, prog_bar=True)
        self.log('ref_lps_rejected', torch.mean(ref_lps_rejected), on_step=True, sync_dist=True, logger=True, prog_bar=True)

        pi_margin = torch.mean(pi_lps_chosen - pi_lps_rejected)
        self.log('pi_margin', pi_margin, on_step=True, sync_dist=True, logger=True, prog_bar=True)

        ref_margin = torch.mean(ref_lps_chosen - ref_lps_rejected)
        self.log('ref_margin', ref_margin, on_step=True, sync_dist=True, logger=True, prog_bar=True)

        margin_delta = pi_margin - ref_margin
        self.log('margin_delta', margin_delta, on_step=True, sync_dist=True, logger=True, prog_bar=True)

        chosen_delta = torch.mean(pi_lps_chosen - ref_lps_chosen)
        self.log('chosen_delta', chosen_delta, on_step=True, sync_dist=True, logger=True, prog_bar=True)

        rej_delta = torch.mean(pi_lps_rejected - ref_lps_rejected)
        self.log('rej_delta', rej_delta, on_step=True, sync_dist=True, logger=True, prog_bar=True)

        pi_accuracy = float(torch.mean((pi_lps_chosen - pi_lps_rejected > 0).float()).cpu().numpy())
        self.log('pi_accuracy', pi_accuracy, on_step=True, sync_dist=True, logger=True, prog_bar=True)

        ref_accuracy = float(torch.mean((ref_lps_chosen - ref_lps_rejected > 0).float()).cpu().numpy())
        self.log('ref_accuracy', ref_accuracy, on_step=True, sync_dist=True, logger=True, prog_bar=True)

        dpo_accuracy = float(torch.mean(((pi_lps_chosen - ref_lps_chosen) - (pi_lps_rejected - ref_lps_rejected) > 0).float()).cpu().numpy())
        self.log('dpo_accuracy', dpo_accuracy, on_step=True, sync_dist=True, logger=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss, (pi_lps_chosen, ref_lps_chosen, pi_lps_rejected, ref_lps_rejected) = self.compute_loss(batch)
        self.log('train_loss', loss, on_step=True, sync_dist=True, logger=True, prog_bar=True)
        self.log_metrics(pi_lps_chosen, ref_lps_chosen, pi_lps_rejected, ref_lps_rejected)

        self.log_learning_rate()
        return loss
    