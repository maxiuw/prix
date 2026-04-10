#!/usr/bin/env python
"""
Measure single-GPU inference FPS for any NavSim Agent (e.g. DiffusionDrive
TransfuserAgent) using Hydra configs.

Usage example (same overrides as run_pdm_score.py):

python test_fps_agent.py \
       train_test_split=navtest \
       agent=diffusiondrive_agent \
       worker=ray_distributed \
       agent.checkpoint_path=/proj/.../loss_16.ckpt \
       experiment_name=diffusiondrive_agent_fps
"""

import time, torch, logging
from typing import Dict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent

# ----------------------------------------------------------------------
# 常量：可按需改
WARM_ITERS   = 20
TIMED_ITERS  = 200

IMG_H, IMG_W      = 256, 1024   # camera_feature
LIDAR_H, LIDAR_W  = 256, 256    # lidar_feature
STATUS_DIM        = 8

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
CONFIG_PATH = "config/pdm_scoring"      # 与原脚本保持一致，方便直接复用配置
CONFIG_NAME = "default_run_pdm_score"
# ----------------------------------------------------------------------


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint – instantiates agent and measures FPS."""
    # 1. 构造 / 加载 Agent
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device).eval()

    logger.info("✅ Agent instantiated: %s", agent.__class__.__name__)

    # 2. 虚拟输入特征
    features: Dict[str, torch.Tensor] = {
        "camera_feature": torch.randn(1, 3, IMG_H, IMG_W, device=device),
        "lidar_feature":  torch.randn(1, 1, LIDAR_H, LIDAR_W, device=device),
        "status_feature": torch.randn(1, STATUS_DIM, device=device),
    }

    # 3. warm-up
    with torch.no_grad():
        for _ in range(WARM_ITERS):
            agent.forward(features)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 4. 计时
    if device.type == "cuda":
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    else:
        t0 = time.time()

    with torch.no_grad():
        for _ in range(TIMED_ITERS):
            agent.forward(features)

    if device.type == "cuda":
        end_evt.record()
        torch.cuda.synchronize()
        total_ms = start_evt.elapsed_time(end_evt)        # CUDA event 返回毫秒
    else:
        total_ms = (time.time() - t0) * 1e3               # 转毫秒

    avg_ms = total_ms / TIMED_ITERS
    fps    = 1_000.0 / avg_ms

    # 5. 打印结果
    print("\n========= FPS BENCHMARK =========")
    print(f"Agent               : {agent.__class__.__name__}")
    print(f"Device              : {device}")
    print(f"Warm-up iterations  : {WARM_ITERS}")
    print(f"Timed iterations    : {TIMED_ITERS}")
    print(f"Avg inference time  : {avg_ms:.3f} ms")
    print(f"FPS                 : {fps:.2f}")
    print("=================================\n")


if __name__ == "__main__":
    main()
