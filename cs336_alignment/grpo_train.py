import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import logging
import tyro
from dataclasses import dataclass, asdict
from typing import Literal, Tuple, Optional

import numpy as np
import torch
import wandb
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.expert_iteration import (
    evaluate_vllm,
    load_policy_into_vllm_instance,
    run_eval,
)
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    compute_log_probs,
    grpo_microbatch_train_step,
)
from cs336_alignment.sft import tokenize_prompt_and_output
from expert_iteration import init_policy_model, init_vllm
from cs336_alignment.utils import setup_experiment


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
    rollout_model_device: str = "cuda:0"
    policy_model_device: str = "cuda:1"
    gpu_memory_utilization: float = 0.85
    use_eager_vllm: bool = (
        False  # When True, skips CUDA graph capture to speed debugging
    )
    reset_vllm_cache_each_step: bool = True  # Optional: clear KV cache to limit growth
    empty_cuda_cache: bool = True  # Optional: call torch.cuda.empty_cache() each step

    # Data + eval
    # train_data: str = "cs336_alignment/results/math_1.5B_train.jsonl"
    dataset: str = "jeggers/competition_math"
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

    # Experiment tracking
    exp_name: str = "grpo_baseline"
    wandb_group: Optional[str] = None
    output_dir: str = "cs336_alignment/results"

    def __post_init__(self):
        self.mini_batch_size = (
            self.rollout_batch_size // self.gradient_accumulation_steps
        )
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size


def init_wandb(cfg: Config):
    wandb_log = lambda *_args, **_kwargs: None
    if cfg.enable_wandb:
        wandb.init(
            project="math-grpo", 
            config=asdict(cfg),
            name=cfg.exp_name,
            group=cfg.wandb_group
        )
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
    rollout_dataset = load_dataset(cfg.dataset, "original", split="train")

    rollout_loader = torch.utils.data.DataLoader(
        rollout_dataset,  # type: ignore
        batch_size=cfg.n_prompts_per_rollout_batch,
        shuffle=False,
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
    eval_dataset = load_dataset(cfg.dataset, "original", split="test")
    return torch.utils.data.DataLoader(
        eval_dataset,  # type: ignore
        batch_size=cfg.eval_reader_local_batch_size,
        shuffle=False,  # Consistent eval across steps
    )


def init_models_and_optimizer(
    cfg: Config,
) -> Tuple[PreTrainedModel, LLM, AutoTokenizer, torch.optim.AdamW, CosineAnnealingLR]:
    eval_model = init_vllm(
        cfg.model_id,
        cfg.rollout_model_device,
        seed=42,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=cfg.use_eager_vllm,
    )
    policy_model = init_policy_model(cfg.model_id, cfg.policy_model_device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.n_grpo_steps, eta_min=cfg.learning_rate * 0.1
    )
    return policy_model, eval_model, tokenizer, optimizer, scheduler


def train(cfg: Config):
    setup_experiment(cfg)
    logger = logging.getLogger(__name__)

    global_train_step = 0
    global_eval_step = 0
    logger.info(f"[config] rollout_model_device={cfg.rollout_model_device}, policy_model_device={cfg.policy_model_device}, model={cfg.model_id}")
    logger.info(
        f"[config] wandb={'on' if cfg.enable_wandb else 'off'}, "
        f"eager_vllm={'on' if cfg.use_eager_vllm else 'off'}"
    )

    wandb_log = init_wandb(cfg)
    load_prompt_template(cfg)
    rollout_loader, rollout_sampling_params = build_rollout(cfg)
    eval_loader = build_eval_loader(cfg)
    policy_model, eval_model, tokenizer, optimizer, scheduler = init_models_and_optimizer(cfg)

    # Zero gradients at start
    optimizer.zero_grad()

    data_iter = iter(rollout_loader)

    for grpo_step in range(cfg.n_grpo_steps):
        load_policy_into_vllm_instance(policy_model, eval_model)

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
        # Move tensors to policy_model_device for compute_log_probs
        prompts = tokenized_dict["input_ids"].to(cfg.policy_model_device)
        responses = tokenized_dict["labels"].to(cfg.policy_model_device)
        response_mask = tokenized_dict["response_mask"].to(cfg.policy_model_device)
        with torch.no_grad():
            old_log_probs = compute_log_probs(
                prompts, responses, policy_model, mem_optimize=True, chunk_size=32
            )
            advantages, raw_rewards, group_rewards = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=flat_response_sts,
                repeated_ground_truths=repeated_ground_truths,
                group_size=cfg.group_size,
                advantage_eps=cfg.advantage_eps,
                normalize_by_std=cfg.use_std_normalization,
            )
            advantages = advantages.unsqueeze(-1).to(cfg.policy_model_device)
            raw_rewards = raw_rewards.unsqueeze(-1).to(cfg.policy_model_device)
            raw_rewards_mean = raw_rewards.mean().item()
            raw_rewards_std = raw_rewards.std().item()
            advantages_mean = advantages.mean().item()
            advantages_std = advantages.std().item()

        # Log/Print example rollouts every 10 steps
        if grpo_step % 10 == 0:
            if cfg.enable_wandb:
                table_data = []
                # Log only the first group to avoid excessive data upload
                for i in range(cfg.group_size):
                    table_data.append([
                        flat_prompt_strs[i],
                        flat_response_sts[i],
                        repeated_ground_truths[i],
                        raw_rewards[i].item()
                    ])
                wandb_log({"rollout_examples": wandb.Table(columns=["Prompt", "Response", "Ground Truth", "Reward"], data=table_data)})

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
                policy_log_probs = compute_log_probs(mb_prompts, mb_responses, policy_model)

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
                        policy_model.parameters(), max_norm=1
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
                    load_policy_into_vllm_instance(policy_model, eval_model)
                    eval_res = run_eval(cfg, eval_model, eval_loader)
                    avg_format_reward = np.mean(
                        [x["score"]["format_reward"] for x in eval_res]
                    )
                    avg_answer_reward = np.mean(
                        [x["score"]["answer_reward"] for x in eval_res]
                    )
                    logging.info(
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

    # Save model
    logging.info("Saving final model...")
    save_path = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(save_path, exist_ok=True)
    policy_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info(f"Model saved to {save_path}")

    # Full evaluation
    logging.info("Running full evaluation on test set...")
    load_policy_into_vllm_instance(policy_model, eval_model)
    
    all_eval_res = []
    # Re-create eval loader to ensure we iterate from start
    full_eval_loader = build_eval_loader(cfg)
    
    from tqdm import tqdm
    for batch in tqdm(full_eval_loader, desc="Evaluating"):
        prompts = [
            cfg.prompt_template.format(question=problem) for problem in batch["problem"]
        ]
        extracted_solutions = [solution for solution in batch["extracted_solution"]]
        
        batch_res = evaluate_vllm(
            eval_model, 
            r1_zero_reward_fn, 
            prompts, 
            extracted_solutions, 
            cfg.sampling_params
        )
        all_eval_res.extend(batch_res)

    avg_format_reward = np.mean([x["score"]["format_reward"] for x in all_eval_res])
    avg_answer_reward = np.mean([x["score"]["answer_reward"] for x in all_eval_res])
    
    logging.info(f"[Final Eval] Format Reward: {avg_format_reward:.4f}, Answer Reward: {avg_answer_reward:.4f}")
    
    if cfg.enable_wandb:
        wandb_log({
            "final_eval/format_reward": avg_format_reward,
            "final_eval/answer_reward": avg_answer_reward,
        })


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    train(cfg)
