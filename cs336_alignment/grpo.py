from typing import Callable, Literal

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
        reward_fn: Callable[[str, str], dict[str, float]],
        scores the rollout responses against the ground truths,
        producing a dict with keys
        "reward", "format_reward", and "answer_reward".
    rollout_responses: list[str], rollouts from the policy.
        The length of this list is
        `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
    repeated_ground_truths: list[str], the ground truths for the examples.
        The length of this list is `rollout_batch_size`,
        because the ground truth for each example is repeated `group_size` times.
    group_size: int, number of rollouts per group.
    advantage_eps: float, epsilon to avoid division by zero
        during group normalization.
    normalize_by_std: bool, whether to normalize the rewards by
        std(rewards).
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    n_prompts = len(rollout_responses) // group_size

    reward_dicts = [
        reward_fn(rollout_response, group_truth)
        for rollout_response, group_truth in zip(
            rollout_responses, repeated_ground_truths
        )
    ]
    raw_rewards = torch.tensor(
        [reward_dict["reward"] for reward_dict in reward_dicts], dtype=torch.float32
    ).view(n_prompts, group_size)

    mean_rewards = torch.mean(raw_rewards, dim=1, keepdim=True)

    advantages = raw_rewards - mean_rewards

    if normalize_by_std:
        std_rewards = torch.std(raw_rewards, dim=1, keepdim=True)
        advantages = advantages / (std_rewards + advantage_eps)
    return advantages.flatten(), raw_rewards.flatten(), {}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in the training loop).

    """
    loss = -raw_rewards_or_advantages * policy_log_probs
    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the old policy.
        cliprange: float Clip parameter ε (e.g. 0.2).
    Returns:  tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
        metadata dict containing whatever you want to log. We suggest logging whether each token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of the min was lower than the LHS.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_surrogate_objective = ratio * advantages
    clipped_surrogate_objective = (
        torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    )
    loss = -torch.min(unclipped_surrogate_objective, clipped_surrogate_objective)
    return loss, {}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
        old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for "grpo_clip"; scalar ε used for clipping.
    Returns:  tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        return (
            compute_naive_policy_gradient_loss(
                raw_rewards,
                policy_log_probs,
            ),
            {},
        )
    if loss_type == "reinforce_with_baseline":
        return (
            compute_naive_policy_gradient_loss(
                advantages,
                policy_log_probs,
            ),
            {},
        )
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all masked elements.
    Returns:  torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Specifically, given the raw rewards or advantages and log probs, we will compute the per-token loss, use masked_mean to aggregate to a scalar loss per example, average over the batch dimension, adjust for gradient accumulation, and backpropagate
    """

    per_token_loss, _ = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    loss = (
        masked_mean(per_token_loss, response_mask, None) / gradient_accumulation_steps
    )
    loss.backward()
    return loss, {}


def compute_log_probs(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    model: AutoModelForCausalLM,
    mem_optimize: bool = False,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Args:
        input_ids: [Total_Batch, Seq_Len]
        labels:    [Total_Batch, Seq_Len]
    Returns:
        log_probs: [Total_Batch, Seq_Len]
    """

    def _compute_chunk(ids_chunk, labels_chunk):
        logits = model(ids_chunk).logits  # [Chunk_Size, Seq_Len, Vocab]
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels_chunk.unsqueeze(-1),
        ).squeeze(-1)

        return per_token_log_probs

    # Fast path when batch is small or memory is ample.
    if not mem_optimize or input_ids.shape[0] <= chunk_size:
        return _compute_chunk(input_ids, labels)

    log_probs_list = []
    total_samples = input_ids.shape[0]
    # Chunk to reduce peak memory while keeping outputs contiguous.
    for i in range(0, total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        chunk_input = input_ids[i:end_idx]
        chunk_labels = labels[i:end_idx]
        chunk_result = _compute_chunk(chunk_input, chunk_labels)
        log_probs_list.append(chunk_result)
    return torch.cat(log_probs_list, dim=0)
