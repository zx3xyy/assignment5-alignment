from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
import wandb
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.expert_iteration import load_policy_into_vllm_instance, run_eval
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    compute_log_probs,
    grpo_microbatch_train_step,
)
from cs336_alignment.sft import tokenize_prompt_and_output
from expert_iteration import init_policy_model, init_vllm


@dataclass
class Config:
    enable_wandb: bool = True
    # Optimization / GRPO hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 32  # microbatch size is 2, will fit on H100
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 256  # On-policy

    # Sampling / reward shaping
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    use_std_normalization: bool = True

    # Model / system configs
    model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_memory_utilization: float = 0.09
    use_eager_vllm: bool = (
        False  # When True, skips CUDA graph capture to speed debugging
    )
    reset_vllm_cache_each_step: bool = True  # Optional: clear KV cache to limit growth
    empty_cuda_cache: bool = True  # Optional: call torch.cuda.empty_cache() each step

    # Data + eval
    train_data: str = "cs336_alignment/results/math_1.5B_train.jsonl"
    eval_data: str = "jeggers/competition_math"
    prompt_template: str = ""
    eval_reader_local_batch_size: int = 32
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    def __post_init__(self):
        self.mini_batch_size = (
            self.rollout_batch_size // self.gradient_accumulation_steps
        )
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size


def init_wandb(cfg: Config):
    wandb_log = lambda *_args, **_kwargs: None
    if cfg.enable_wandb:
        wandb.init(project="math-sft", config=cfg)
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")
        wandb_log = wandb.log
    return wandb_log


def load_prompt_template(cfg: Config):
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        cfg.prompt_template = f.read()


def build_rollout(cfg: Config) -> Tuple[DataLoader, SamplingParams]:
    # rollout_dataset = load_dataset("json", data_files=cfg.train_data, split="train")
    rollout_dataset = load_dataset(cfg.eval_data, "original", split="train")

    rollout_loader = torch.utils.data.DataLoader(
        rollout_dataset,
        batch_size=cfg.n_prompts_per_rollout_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    rollout_sampling_params: SamplingParams = SamplingParams(
        temperature=cfg.sampling_temperature,
        top_p=1.0,
        max_tokens=cfg.sampling_max_tokens,
        min_tokens=cfg.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=cfg.group_size,
    )
    return rollout_loader, rollout_sampling_params


def build_eval_loader(cfg: Config) -> DataLoader:
    eval_dataset = load_dataset(cfg.eval_data, "original", split="test")
    return torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.eval_reader_local_batch_size,
        shuffle=False,  # Consistent eval across steps
    )


def init_models_and_optimizer(
    cfg: Config,
) -> Tuple[torch.nn.Module, LLM, AutoTokenizer, torch.optim.AdamW, CosineAnnealingLR]:
    policy = init_policy_model(cfg.model_id, cfg.device)
    eval_model = init_vllm(
        cfg.model_id,
        cfg.device,
        seed=42,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=cfg.use_eager_vllm,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.n_grpo_steps, eta_min=cfg.learning_rate * 0.1
    )
    return policy, eval_model, tokenizer, optimizer, scheduler


def train(cfg: Config):
    global_train_step = 0
    global_eval_step = 0
    print(f"[config] device={cfg.device}, model={cfg.model_id}")
    print(
        f"[config] wandb={'on' if cfg.enable_wandb else 'off'}, "
        f"eager_vllm={'on' if cfg.use_eager_vllm else 'off'}"
    )

    wandb_log = init_wandb(cfg)
    load_prompt_template(cfg)
    rollout_loader, rollout_sampling_params = build_rollout(cfg)
    eval_loader = build_eval_loader(cfg)
    policy, eval_model, tokenizer, optimizer, scheduler = init_models_and_optimizer(cfg)

    # Zero gradients at start
    optimizer.zero_grad()

    data_iter = iter(rollout_loader)

    for grpo_step in range(cfg.n_grpo_steps):
        load_policy_into_vllm_instance(policy, eval_model)

        # Handle data iterator exhaustion
        try:
            examples = next(data_iter)
        except StopIteration:
            data_iter = iter(rollout_loader)
            examples = next(data_iter)

        prompts = [
            cfg.prompt_template.format(question=problem)
            for problem in examples["problem"]
        ]
        repeated_ground_truths = [
            extracted_solution
            for extracted_solution in examples["extracted_solution"]
            for _ in range(cfg.group_size)
        ]
        flat_prompt_strs = [prompt for prompt in prompts for _ in range(cfg.group_size)]
        responses = eval_model.generate(prompts, rollout_sampling_params)
        flat_response_sts = [
            output.text for response in responses for output in response.outputs
        ]
        tokenized_dict = tokenize_prompt_and_output(
            flat_prompt_strs, flat_response_sts, tokenizer
        )
        prompts = tokenized_dict["input_ids"].to(cfg.device)
        responses = tokenized_dict["labels"].to(cfg.device)
        response_mask = tokenized_dict["response_mask"].to(cfg.device)
        with torch.no_grad():
            old_log_probs = compute_log_probs(
                prompts, responses, policy, mem_optimize=True, chunk_size=32
            )
            advantages, raw_rewards, group_rewards = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=flat_response_sts,
                repeated_ground_truths=repeated_ground_truths,
                group_size=cfg.group_size,
                advantage_eps=cfg.advantage_eps,
                normalize_by_std=cfg.use_std_normalization,
            )
            advantages = advantages.unsqueeze(-1).to(cfg.device)
            raw_rewards = raw_rewards.unsqueeze(-1).to(cfg.device)
            raw_rewards_mean = raw_rewards.mean().item()
            raw_rewards_std = raw_rewards.std().item()
            advantages_mean = advantages.mean().item()
            advantages_std = advantages.std().item()

        for epoch in range(cfg.epochs_per_rollout_batch):
            for mb_step, mini_batch_start_idx in enumerate(
                range(0, cfg.rollout_batch_size, cfg.mini_batch_size)
            ):
                mb_idx = range(cfg.rollout_batch_size)[
                    mini_batch_start_idx : mini_batch_start_idx + cfg.mini_batch_size
                ]
                mb_prompts = prompts[mb_idx]
                mb_responses = responses[mb_idx]
                mb_response_mask = response_mask[mb_idx]
                mb_raw_rewards = raw_rewards[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                policy_log_probs = compute_log_probs(mb_prompts, mb_responses, policy)

                loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    loss_type=cfg.loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=0.2,
                )

                if (mb_step + 1) % cfg.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), max_norm=1
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_train_step += 1
                    # Step the scheduler after optimizer step
                    scheduler.step()

                    wandb_log(
                        {
                            "train_step": global_train_step,
                            "train/loss": loss.item() * cfg.gradient_accumulation_steps,
                            "train/grad_norm": grad_norm.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/reward_mean": raw_rewards_mean,
                            "train/reward_std": raw_rewards_std,
                            "train/adv_mean": advantages_mean,
                            "train/adv_std": advantages_std,
                        }
                    )
                    # Eval after each update!
                    load_policy_into_vllm_instance(policy, eval_model)
                    eval_res = run_eval(cfg, eval_model, eval_loader)
                    avg_format_reward = np.mean(
                        [x["score"]["format_reward"] for x in eval_res]
                    )
                    avg_answer_reward = np.mean(
                        [x["score"]["answer_reward"] for x in eval_res]
                    )
                    print(
                        f"[train] step={global_train_step} "
                        f"loss={loss.item() * cfg.gradient_accumulation_steps:.4f} "
                        f"grad_norm={grad_norm.item():.2f} "
                        f"reward_mean={raw_rewards_mean:.4f} "
                        f"adv_mean={advantages_mean:.4f} "
                        f"eval_format={avg_format_reward:.3f} "
                        f"eval_answer={avg_answer_reward:.3f}"
                    )
                    global_eval_step += 1
                    wandb_log(
                        {
                            "eval_step": global_eval_step,
                            "eval/format_reward": avg_format_reward,
                            "eval/answer_reward": avg_answer_reward,
                        }
                    )
        if cfg.reset_vllm_cache_each_step and hasattr(eval_model, "llm_engine"):
            try:
                eval_model.llm_engine.cache_engine.reset()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] failed to reset vLLM cache: {exc}")

        if cfg.empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        del (
            prompts,
            responses,
            response_mask,
            raw_rewards,
            advantages,
            old_log_probs,
            flat_response_sts,
            tokenized_dict,
        )


if __name__ == "__main__":
    train(Config())
