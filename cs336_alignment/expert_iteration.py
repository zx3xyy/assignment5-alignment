from datasets import load_dataset
from typing import Callable, List
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
import pandas as pd
from cs336_alignment.sft import *
import wandb
import numpy as np
from dataclasses import dataclass

from dataclasses import dataclass, field
from vllm import SamplingParams
import tyro

from datasets import load_dataset
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sft import init_vllm


@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sampling_params: SamplingParams = field(
        default_factory=lambda: SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
    )
    prompt_template: str = ""
    train_data: str = "results/math_1.5B_train.jsonl"
    eval_data: str = "jeggers/competition_math"

    gradient_accumulation_steps: int = 16
    eval_gap: int = 16
    train_reader_local_batch_size: int = 4
    eval_reader_local_batch_size: int = 32

    n_epochs: int = 3  # 统一用 n_epochs
    n_steps_per_epoch: int = 78
    peak_lr: float = 2e-5
    total_steps: int = 1000
    warmup_steps: int = int(0.1 * 1000)

    expert_iteration: bool = True
    G: int = 4  # rollout per prompt
    D_B: int = 512  # 统一用 D_B

    vllm_gpu_memory_utilization: float = 0.1
    n_ei_steps: int = 5


cfg = Config()


def init_policy_model(
    model_id: str,
    device: str,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    return model


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truth_list: List[str],
    eval_sampling_params: SamplingParams,
):
    res = []
    model_outputs = vllm_model.generate(prompts, eval_sampling_params)
    for (
        prompt,
        output,
        ground_truth,
    ) in zip(prompts, model_outputs, ground_truth_list):
        for completition in output.outputs:
            model_answer = completition.text
            res.append(
                {
                    "prompt": prompt,
                    "generated_text": model_answer,
                    "ground_truth": ground_truth,
                    "score": reward_fn(model_answer, ground_truth),
                }
            )
    return res


def run_eval(cfg, eval_model, eval_loader):
    examples = next(iter(eval_loader))
    prompts = [
        cfg.prompt_template.format(question=problem) for problem in examples["problem"]
    ]
    extracted_solutions = [solution for solution in examples["extracted_solution"]]
    batch_results = evaluate_vllm(
        eval_model, r1_zero_reward_fn, prompts, extracted_solutions, cfg.sampling_params
    )
    return batch_results


class SFTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {"prompt": item["prompt"], "generated_text": item["generated_text"]}


def collate_fn(batch):
    prompts = [b["prompt"] for b in batch]
    generated_text = [b["generated_text"] for b in batch]

    return {
        "prompt": prompts,  # List[str] or Tokenized Tensors
        "generated_text": generated_text,
    }


def get_train_dataloader(cfg, eval_model):
    rollout_dataset = load_dataset(cfg.eval_data, "original", split="train")
    rollout_loader = torch.utils.data.DataLoader(
        rollout_dataset,
        batch_size=cfg.D_B,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    examples = next(iter(rollout_loader))

    rollout_sampling_params: SamplingParams = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=cfg.G,
    )
    prompts = [
        cfg.prompt_template.format(question=problem) for problem in examples["problem"]
    ]

    rollout_results = evaluate_vllm(
        eval_model,
        r1_zero_reward_fn,
        prompts,
        examples["extracted_solution"],
        rollout_sampling_params,
    )
    correct = [x for x in rollout_results if x["score"]["reward"] == 1.0]
    dataset = SFTDataset(correct)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.train_reader_local_batch_size,
        shuffle=True,
        collate_fn=collate_fn,  # Critical for variable length sequences
    )
    return train_loader


def train(cfg, model, tokenizer, optimizer, eval_model, train_loader, eval_loader):
    global_train_step = 0
    global_eval_step = 0

    for epoch in range(cfg.n_epochs):
        print(f"=== Starting Epoch {epoch + 1} / {cfg.n_epochs} ===")
        for idx, examples in enumerate(train_loader):
            prompt_strs = examples["prompt"]
            output_strs = examples["generated_text"]
            model_input = tokenize_prompt_and_output(
                prompt_strs, output_strs, tokenizer, cfg.device
            )
            model_output = get_response_log_probs(
                model, model_input["input_ids"], model_input["labels"], True
            )
            loss, metadata = sft_microbatch_train_step(
                model_output["log_probs"],
                model_input["response_mask"],
                cfg.gradient_accumulation_steps,
                1.0,
            )

            if (idx + 1) % cfg.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                global_train_step += 1
                wandb.log(
                    {
                        "train_step": global_train_step,
                        "train/loss": loss.item() * cfg.gradient_accumulation_steps,
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/token_entropy": torch.mean(
                            model_output["token_entropy"]
                        ).item(),
                    }
                )

                # Eval after each update!
                # if idx % cfg.eval_gap == 0:
                global_eval_step += 1
                load_policy_into_vllm_instance(model, eval_model)
                eval_res = run_eval(cfg, eval_model, eval_loader)
                avg_format_reward = np.mean(
                    [x["score"]["format_reward"] for x in eval_res]
                )
                avg_answer_reward = np.mean(
                    [x["score"]["answer_reward"] for x in eval_res]
                )
                table = wandb.Table(
                    columns=["prompt", "generated_text", "ground_truth", "score"],
                    data=pd.DataFrame(eval_res).astype(str),
                )

                wandb.log(
                    {
                        "eval_step": global_eval_step,
                        "eval/format_reward": avg_format_reward,
                        "eval/answer_reward": avg_answer_reward,
                        "eval/examples": table,
                    }
                )


def main(cfg):
    with open("prompts/r1_zero.prompt") as f:
        cfg.prompt_template = f.read()

    wandb.init(project="math-sft", config=cfg)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    model = init_policy_model(cfg.model_id, cfg.device)
    eval_model = init_vllm(
        cfg.model_id,
        cfg.device,
        seed=42,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    eval_dataset = load_dataset(cfg.eval_data, "original", split="test")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.eval_reader_local_batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )

    for step in range(cfg.n_ei_steps):
        load_policy_into_vllm_instance(model, eval_model)
        train_dataloader = get_train_dataloader(cfg, eval_model)
        train(
            cfg, model, tokenizer, optimizer, eval_model, train_dataloader, eval_loader
        )


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
