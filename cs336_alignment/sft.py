from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM, SamplingParams
import torch
from transformers import PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
    device: str | None = None,
):
    prompt_strs = prompt_strs
    B = len(prompt_strs)
    promopt_ids = tokenizer(prompt_strs)["input_ids"]
    out_ids = tokenizer(output_strs)["input_ids"]
    prompt_and_output = [x + y for x, y in zip(promopt_ids, out_ids)]
    max_prompt_and_output_len = max([len(x) for x in prompt_and_output])
    tokenized = torch.full(
        (B, max_prompt_and_output_len), tokenizer.pad_token_id, dtype=torch.long
    ).to(device)
    response_mask = torch.zeros(
        (B, max_prompt_and_output_len - 1), dtype=torch.bool
    ).to(device)
    for i in range(B):
        for p in range(len(promopt_ids[i])):
            tokenized[i, p] = promopt_ids[i][p]
        for l in range(len(out_ids[i])):
            tokenized[i, len(promopt_ids[i]) + l] = out_ids[i][l]
            response_mask[i, len(promopt_ids[i]) + l - 1] = True

    return {
        "input_ids": tokenized[:, :-1],
        "labels": tokenized[:, 1:],
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: batch_size, sequence_length, vocab_size
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    entroy = -torch.sum(probs * log_probs, dim=-1)
    return entroy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    res = {}
    res["log_probs"] = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze()
    if return_token_entropy:
        res["token_entropy"] = compute_entropy(logits)
    return res


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return torch.sum(mask * tensor, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = (
        torch.sum(
            masked_normalize(
                -policy_log_probs, response_mask, normalize_constant, dim=-1
            )
        )
        / torch.sum(response_mask)
        / gradient_accumulation_steps
    )
    loss.backward()
    meta_data = {}

    return loss, meta_data


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
