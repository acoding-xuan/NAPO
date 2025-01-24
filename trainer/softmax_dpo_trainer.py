# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import importlib
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import numpy as np

from .utils import DPODataCollatorWithPadding, pad_to_length


def is_peft_available():
    return importlib.util.find_spec("peft") is not None

if is_peft_available():
    from peft import get_peft_model, prepare_model_for_kbit_training


class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 0.1,
        gamma: float = 1.0,
        alpha1: float = 0.9,
        alpha2: float = 0.9,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        rec_model: Optional[nn.Module] = None,
        batch_shared: bool = True,
        move_ratio: float = 0.9,
        world_size: int = 4,
        neg_accept_ratio: float = 0.7,
        similarity_score:Callable[[int, int], float] = None, 
    ):
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_int8_training(model)
            model = get_peft_model(model, peft_config)

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.beta = beta
        self.gamma = gamma
        self.ref_model = ref_model
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.move_ratio = move_ratio
        self.M_mean = torch.zeros(1, device='cuda')
        self.A_mean = torch.zeros(1, device='cuda')
        #self.m_std = torch.zeros(1, device='cuda')
        self.world_size = world_size
        self.batch_shared = batch_shared
        self.similarity_score = similarity_score
        self.neg_accept_ratio = neg_accept_ratio
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            None,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        # Since we inherit from trainer we always have access to an accelerator
        # if hasattr(self, "accelerator"):
        #     self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        # else:
        #     raise AttributeError(
        #         "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
        #     )

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):] 
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        return concatenated_batch

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: Dict[str, torch.FloatTensor],
        M_mean: float,
        A: torch.FloatTensor

        # reference_chosen_logps: torch.FloatTensor,
        # reference_rejected_logps: Dict[str, torch.FloatTensor],
        # reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        # chosen_logratios = policy_chosen_logps
        # # print(f"chosen:{chosen_logratios}")
        # rejected_logratios = {}
       
        # if self.batch_shared:
        #     tensors = [value for value in policy_rejected_logps.values()]
        #     # 拼接 tensor
        #     result = torch.cat(tensors)
        #     #for key in policy_rejected_logps:
        #         #rejected_logratios[key] = result
        #     temp = sum(torch.exp(self.beta * (rejected_log - chosen_logratios)) for rejected_log in result) 
        # else:
        #     for key in policy_rejected_logps:
        #         rejected_logratios[key] = policy_rejected_logps[key]
        #     #gamma = self.gamma - self.alpha1 * (M_mean - self.M_mean.item()) * self.gamma
        #     # logits = pi_logratios - ref_logratios
        #     temp = sum(torch.exp(self.beta * (rejected_logratios[key] - chosen_logratios)) for key in rejected_logratios)
        # A = -torch.log(temp)
        
        gamma = self.gamma - self.alpha1 * (M_mean - self.M_mean.item()) * self.gamma 
        #- self.alpha2 * (A.mean().item() - self.A_mean.item()) * self.gamma
        losses = -F.logsigmoid(A - gamma)
        # losses = -F.logsigmoid(self.beta * logits)
        rejected_rewards = {}
        chosen_rewards = self.beta * policy_chosen_logps .detach()
        for key in policy_rejected_logps:
            rejected_rewards[key] = self.beta * policy_rejected_logps[key].detach()
        return losses, chosen_rewards, rejected_rewards, gamma

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        # print(concatenated_batch["concatenated_input_ids"].shape)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        step = batch["chosen_input_ids"].shape[0]
        rejected_logps = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logps[f"rejected{cnt}"] = all_logps[step*cnt : step*(cnt+1)]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logits[f"rejected{cnt}"] = all_logits[step*cnt : step*(cnt+1)]
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)


    def update_and_sync_neg_sample(self, rec_score_rejected, neg_logits,dataset_ids):
        with torch.no_grad():
            # Create lists to gather results from all processes
            gathered_rec_scores = [torch.zeros_like(rec_score_rejected) for _ in range(self.world_size)]
            gathered_neg_logits = [torch.zeros_like(neg_logits) for _ in range(self.world_size)]
            gathered_dataset_ids = [torch.zeros_like(dataset_ids) for _ in range(self.world_size)]
            # Gather tensors from all processes
            dist.all_gather(gathered_rec_scores, rec_score_rejected)
            dist.all_gather(gathered_neg_logits, neg_logits)
            dist.all_gather(gathered_dataset_ids, dataset_ids)

            # Concatenate the results
            rec_score_rejected_concat = torch.cat(gathered_rec_scores,dim=0)
            neg_logits_concat = torch.cat(gathered_neg_logits, dim=0)
            dataset_ids_concat = torch.cat(gathered_dataset_ids, dim=0)

            return rec_score_rejected_concat, neg_logits_concat, dataset_ids_concat


    def update_and_sync_tensor_mean(self, M_local, A_local, move_ratio=0.9):
        with torch.no_grad():
            # batch_gap_mean = M_local.mean()
            # batch_gap_mean = M_local.mean()
            #batch_gap_std = m_local.std()
            # 更新M_mean
            self.M_mean.mul_(move_ratio).add_(M_local.mean(), alpha=1-move_ratio)
            #self.A_mean.mul_(move_ratio).add_(A_local.mean(), alpha=1-move_ratio)
            #self.m_std.mul_(move_ratio).add_(batch_gap_std, alpha=1-move_ratio)
            if self.world_size > 1:
                dist.all_reduce(self.M_mean, op=dist.ReduceOp.SUM)
                #dist.all_reduce(self.A_mean, op=dist.ReduceOp.SUM)
                #dist.all_reduce(self.m_std, op=dist.ReduceOp.SUM)
                self.M_mean /= self.world_size
                #self.A_mean /= self.world_size
                #self.m_std /= self.world_size

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        rec_score_rejected = []
        dataset_ids = []   
        for key in batch:
            if key.startswith("score_rejected"):
                rec_score_rejected.append(batch[key])
            elif key.startswith("dataset_id"):
                dataset_ids.extend(batch[key])  

        rec_score_rejected = torch.tensor(rec_score_rejected).T.contiguous().cuda()
        dataset_ids = torch.tensor(dataset_ids).cuda()
        if self.batch_shared:
            neg_logits = [value for value in policy_rejected_logps.values()]
            neg_logits = torch.stack(neg_logits, dim=0).T.contiguous()
            rec_score_rejected, neg_logits, all_dataset_ids = self.update_and_sync_neg_sample(rec_score_rejected, neg_logits, dataset_ids)
            neg_accept_ratio = self.neg_accept_ratio
            similarity_matrix = self.similarity_score(dataset_ids, all_dataset_ids)
            rec_scores_rej_list = []
            neg_logits_rej_list = []
            num_to_take = int(neg_accept_ratio * (len(all_dataset_ids) - 1))
            for i, dataset_ids_i in enumerate(dataset_ids): 
                scores_others = similarity_matrix[i]
                k = (all_dataset_ids == dataset_ids_i.item()).nonzero(as_tuple=True)[0]
                #k = all_dataset_ids.index(dataset_ids_i)
                scores_others[k] = float('-inf')
                sorted_indices = torch.argsort(scores_others, descending=True)
                top_k_indices = sorted_indices[:num_to_take]
                rec_scores_rej = [rec_score_rejected[i]]
                neg_logits_rej = [neg_logits[i]]
                sorted_rec_scores_rej_others = rec_score_rejected[top_k_indices]
                sorted_neg_logits_rej_others = neg_logits[top_k_indices]

                rec_scores_rej.extend(sorted_rec_scores_rej_others)
                neg_logits_rej.extend(sorted_neg_logits_rej_others)
                
                rec_scores_rej_list.append(torch.cat(rec_scores_rej, dim=0))
                neg_logits_rej_list.append(torch.cat(neg_logits_rej, dim=0))

            try:
                rec_scores_rej = torch.stack(rec_scores_rej_list)
                neg_logits_rej = torch.stack(neg_logits_rej_list)
            except Exception as e:
                print(f"Error: {e}")
                print(f"rec_scores_rej_list dimensions: {[x.shape for x in rec_scores_rej_list]}")
                print(f"neg_logits_rej_list dimensions: {[x.shape for x in neg_logits_rej_list]}")  
            M_local = torch.mean(rec_scores_rej)
            M_mean = M_local.item()
        else:
            M_local = torch.mean(rec_score_rejected)
            M_mean = M_local.item()
        chosen_logratios = policy_chosen_logps
        # print(f"chosen:{chosen_logratios}")
        rejected_logratios = {}
        if self.batch_shared:
            temp = sum(torch.exp(self.beta * (neg_logits_rej[: ,i] - chosen_logratios)) for i in range(neg_logits_rej.shape[1]) ) 
        else:
            for key in policy_rejected_logps:
                rejected_logratios[key] = policy_rejected_logps[key]
            temp = sum(torch.exp(self.beta * (rejected_logratios[key] - chosen_logratios)) for key in rejected_logratios)
        A = -torch.log(temp)
        A_local = A.mean()

        self.update_and_sync_tensor_mean(M_local, A_local, self.move_ratio)
        
        #self.update_and_sync_tensor_mean(m_local, self.alpha)
        losses, chosen_rewards, rejected_rewards, gamma = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
                    M_mean,
                    A,
            
        )
        reward_accuracies = None
        for key in rejected_rewards:
            if reward_accuracies is None:
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            else:
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}self_M_mean"] = self.M_mean.item()
        metrics[f"{prefix}M_mean"] = M_mean
        metrics[f"{prefix}self_A_mean"] = self.A_mean.item()
        metrics[f"{prefix}A_mean"] = A_local.item()
        metrics[f"{prefix}gamma"] = gamma
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:    
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        for key in policy_rejected_logits:    
            metrics[f"{prefix}logits/rejected-{key}"] = policy_rejected_logits[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # print(inputs.keys())
        # print(inputs)
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    # def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
    #     """Generate samples from the model and reference model for the given batch of inputs."""

    #     policy_output = model.generate(
    #         batch["prompt_input_ids"],
    #         attention_mask=batch["prompt_attention_mask"],
    #         max_length=self.config.max_length,
    #         do_sample=True,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #     )

    #     reference_output = self.ref_model.generate(
    #         batch["prompt_input_ids"],
    #         attention_mask=batch["prompt_attention_mask"],
    #         max_length=self.config.max_length,
    #         do_sample=True,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #     )

    #     policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
    #     policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

    #     reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
    #     reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

    #     return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            # "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

