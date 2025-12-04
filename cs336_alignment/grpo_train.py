from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import SamplingParams

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

    # Data + eval
    train_data: str = "results/math_1.5B_train.jsonl"
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


def main():
    cfg = Config()
    global_train_step = 0
    global_eval_step = 0
    wandb.init(project="math-sft", config=cfg)  # Track config and metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        cfg.prompt_template = f.read()

    # Evaluation dataloader uses the held-out split
    eval_dataset = load_dataset("jeggers/competition_math", "original", split="test")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.eval_reader_local_batch_size,
        shuffle=True,
    )
    policy = init_policy_model(cfg.model_id, cfg.device)
    eval_model = init_vllm(
        cfg.model_id,
        cfg.device,
        seed=42,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    rollout_dataset = load_dataset(cfg.eval_data, "original", split="train")
    rollout_loader = torch.utils.data.DataLoader(
        rollout_dataset,
        batch_size=cfg.n_prompts_per_rollout_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
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
    data_iter = iter(rollout_loader)

    # Main GRPO training loop
    for grpo_step in range(cfg.n_grpo_steps):
        load_policy_into_vllm_instance(policy, eval_model)
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
            old_log_probs = compute_log_probs(prompts, responses, policy, True)
            advantages, raw_rewards, _ = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=flat_response_sts,
                repeated_ground_truths=repeated_ground_truths,
                group_size=cfg.group_size,
                advantage_eps=1e-8,
                normalize_by_std=False,
            )
            advantages = advantages.unsqueeze(-1).to(cfg.device)
            raw_rewards = raw_rewards.unsqueeze(-1).to(cfg.device)

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
                    wandb.log(
                        {
                            "train_step": global_train_step,
                            "train/loss": loss.item() * cfg.gradient_accumulation_steps,
                            "train/grad_norm": grad_norm.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
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
                    global_eval_step += 1
                    wandb.log(
                        {
                            "eval_step": global_eval_step,
                            "eval/format_reward": avg_format_reward,
                            "eval/answer_reward": avg_answer_reward,
                        }
                    )


if __name__ == "__main__":
    main()
