"""
列出模型在一次前/反向后仍然没有梯度的参数，
并把名字保存到 unused_param_list.json 方便后续冻结。
"""

from pathlib import Path
import json, logging, torch
from typing import Dict, Tuple, List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from navsim.planning.training.dataset import CacheOnlyDataset
from navsim.agents.abstract_agent import AbstractAgent

log = logging.getLogger("find_unused")

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


# ---------------------- 帮助函数 ----------------------
def _one_batch(
    agent: AbstractAgent,
    cfg: DictConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    直接复用正式训练时的 feature_builders，从 cache 拿 1 条样本
    """
    cache_path = Path(cfg.cache_path)
    assert cache_path.is_dir(), f"cache_path={cache_path} 不存在！请先跑一次正式训练生成缓存"

    log_names = cfg.train_logs[:1]

    # 1. 拿 builders
    feature_builders = agent.get_feature_builders()
    target_builders  = agent.get_target_builders()   # ← 这里改回来！

    # 2. 构造 CacheOnlyDataset
    dummy_ds = CacheOnlyDataset(
        cache_path       = str(cache_path),
        feature_builders = feature_builders,
        target_builders  = target_builders,          # ← 传完整
        log_names        = cfg.train_logs[:1],
    )
    dl = DataLoader(dummy_ds, batch_size=1, shuffle=False, num_workers=0)

    features, targets = next(iter(dl))
    features = {k: v.cuda() for k, v in features.items()}
    targets = {k: v.cuda() for k, v in targets.items()}

    return features, targets


# ----------------------- 主入口 -----------------------
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):

    # 1. 实例化 agent（这里用你的 diffusiondrive_agent）和模型
    log.info("Instantiate agent && model")
    agent: AbstractAgent = instantiate(cfg.agent)
    model = agent._transfuser_model.cuda()          # 类型: V2TransfuserModel
    model.train()

    # 2. 拿一批缓存样本
    features, targets = _one_batch(agent, cfg)

    # 3. 前向 / 反向
    log.info("Forward / backward once")
    out = model(features, targets)
    loss = out["trajectory_loss"] if "trajectory_loss" in out else out["loss"]
    loss.backward()

    # 4. 统计无梯度参数
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]

    print("\n=== UNUSED PARAMS (%d) ===" % len(unused))
    for n in unused:
        print(n)

    # 5. 保存到文件，方便后续冻结
    dump_path = Path.cwd() / "unused_param_list.json"
    json.dump(unused, open(dump_path, "w"), indent=2)
    print(f"\n已写入 {dump_path.resolve()}")

if __name__ == "__main__":
    main()
