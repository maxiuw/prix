# from typing import Any, Dict, List, Union, Tuple
# from pathlib import Path
# from dataclasses import asdict
# from datetime import datetime
# import traceback
# import logging
# import lzma
# import pickle
# import os
# import uuid

# import hydra
# from hydra.utils import instantiate
# from omegaconf import DictConfig
# import pandas as pd

# from nuplan.planning.script.builders.logging_builder import build_logger
# from nuplan.planning.utils.multithreading.worker_utils import worker_map

# from navsim.agents.abstract_agent import AbstractAgent
# from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
# from navsim.common.dataclasses import SensorConfig
# from navsim.evaluate.pdm_score import pdm_score
# from navsim.planning.script.builders.worker_pool_builder import build_worker
# from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
# from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
# from navsim.planning.metric_caching.metric_cache import MetricCache
# from navsim.visualization.plots import plot_cameras_frame
# from matplotlib import pylab as plt
# from navsim.visualization.plots import plot_bev_with_agent


# logger = logging.getLogger(__name__)

# CONFIG_PATH = "config/pdm_scoring"
# CONFIG_NAME = "default_run_pdm_score"


# def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
#     """
#     Helper function to run PDMS evaluation in.
#     :param args: input arguments
#     """
#     node_id = int(os.environ.get("NODE_RANK", 0))
#     thread_id = str(uuid.uuid4())
#     logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

#     log_names = [a["log_file"] for a in args]
#     tokens = [t for a in args for t in a["tokens"]]
#     cfg: DictConfig = args[0]["cfg"]

#     simulator: PDMSimulator = instantiate(cfg.simulator)
#     scorer: PDMScorer = instantiate(cfg.scorer)
#     assert (
#         simulator.proposal_sampling == scorer.proposal_sampling
#     ), "Simulator and scorer proposal sampling has to be identical"
#     agent: AbstractAgent = instantiate(cfg.agent)
#     agent.initialize()
#         # ===== 添加可视化保存功能 =====
    
#     # 创建保存目录
#     viz_save_dir = Path("/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/exp/traj_viz")
#     viz_save_dir.mkdir(parents=True, exist_ok=True)
    
#     # 生成唯一文件名（包含时间戳和线程ID）
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"bev_agent_comparison_{timestamp}_{thread_id[:8]}.png"
#     save_path = viz_save_dir / filename
    
#     try:
#         # 生成BEV可视化图
#         fig, ax = plot_bev_with_agent(scene, agent)
        
#         # 添加标题信息
#         ax.set_title(f"BEV Agent Comparison\nThread: {thread_id[:8]}, Node: {node_id}")
        
#         # 保存图像
#         fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         logger.info(f"BEV visualization saved to: {save_path}")
        
#         # 关闭图像以释放内存
#         plt.close(fig)
        
#     except Exception as e:
#         logger.warning(f"Failed to save BEV visualization: {e}")
    
#     # ===== 可视化保存功能结束 =====
#     #/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/exp/traj_viz
#     metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
#     scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
#     scene_filter.log_names = log_names
#     scene_filter.tokens = tokens
#     scene_loader = SceneLoader(
#         sensor_blobs_path=Path(cfg.sensor_blobs_path),
#         data_path=Path(cfg.navsim_log_path),
#         scene_filter=scene_filter,
#         sensor_config=agent.get_sensor_config(),
#     )

#     tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
#     pdm_results: List[Dict[str, Any]] = []
#     tokens_to_evaluate = [
#         'e36edd3aedf05e30',
#         'f99a74d444e651d3',
#         '67a6bdeb096350ec',
#         '5b7700fa99d95a94',
#         '564bb94f846e5fe1',
#         '87efb8cf52135247',
#         'f3e0463f3cf4505e',
#         '16aa734bed8a5f81',
#         '948e6a45c7cd5837',
#         '5d68790fd55c5e41',
#     ]
#     for idx, (token) in enumerate(tokens_to_evaluate):
#         logger.info(
#             f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
#         )
#         score_row: Dict[str, Any] = {"token": token, "valid": True}
#         try:
#             metric_cache_path = metric_cache_loader.metric_cache_paths[token]
#             with lzma.open(metric_cache_path, "rb") as f:
#                 metric_cache: MetricCache = pickle.load(f)

#             agent_input = scene_loader.get_agent_input_from_token(token)
#             if agent.requires_scene:
#                 scene = scene_loader.get_scene_from_token(token)
#                 trajectory = agent.compute_trajectory(agent_input, scene)
#             else:
#                 viz(agent_input)
#                 trajectory = agent.compute_trajectory(agent_input)

#             pdm_result = pdm_score(
#                 metric_cache=metric_cache,
#                 model_trajectory=trajectory,
#                 future_sampling=simulator.proposal_sampling,
#                 simulator=simulator,
#                 scorer=scorer,
#             )
#             score_row.update(asdict(pdm_result))
#         except Exception as e:
#             logger.warning(f"----------- Agent failed for token {token}:")
#             traceback.print_exc()
#             score_row["valid"] = False

#         pdm_results.append(score_row)
        
#     return pdm_results


# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
# def main(cfg: DictConfig) -> None:
#     """
#     Main entrypoint for running PDMS evaluation.
#     :param cfg: omegaconf dictionary
#     """

#     build_logger(cfg)
#     worker = build_worker(cfg)

#     # Extract scenes based on scene-loader to know which tokens to distribute across workers
#     # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
#     scene_loader = SceneLoader(
#         sensor_blobs_path=None,
#         data_path=Path(cfg.navsim_log_path),
#         scene_filter=instantiate(cfg.train_test_split.scene_filter),
#         sensor_config=SensorConfig.build_no_sensors(),
#     )
#     metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

#     tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
#     num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
#     num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
#     if num_missing_metric_cache_tokens > 0:
#         logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
#     if num_unused_metric_cache_tokens > 0:
#         logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
#     logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
#     data_points = [
#         {
#             "cfg": cfg,
#             "log_file": log_file,
#             "tokens": tokens_list,
#         }
#         for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
#     ]
#     # score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points) # for visualization and debugging comment out 
#     score_rows = run_pdm_score(data_points)

#     pdm_score_df = pd.DataFrame(score_rows)
#     num_sucessful_scenarios = pdm_score_df["valid"].sum()
#     num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
#     average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
#     average_row["token"] = "average"
#     average_row["valid"] = pdm_score_df["valid"].all()
#     pdm_score_df.loc[len(pdm_score_df)] = average_row

#     save_path = Path(cfg.output_dir)
#     timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
#     pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

#     logger.info(
#         f"""
#         Finished running evaluation.
#             Number of successful scenarios: {num_sucessful_scenarios}.
#             Number of failed scenarios: {num_failed_scenarios}.
#             Final average score of valid results: {pdm_score_df['score'].mean()}.
#             Results are stored in: {save_path / f"{timestamp}.csv"}.
#         """
#     )


# def viz(scene):
#     cams = Data(scene.cameras)
#     idx = 3
#     fig, ax = plot_cameras_frame(cams, idx)
#     fig.savefig("test.png", dpi=300)
#     plt.show()  

# class Data():
#     def __init__(self, inpt) -> None:
#         self.frames = inpt
# if __name__ == "__main__":
#     main()


from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.visualization.plots import plot_cameras_frame, plot_bev_with_agent
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


class Data:
    """简单的数据包装类，用于可视化"""
    def __init__(self, inpt) -> None:
        self.frames = inpt


def viz_cameras(agent_input, token, save_dir):
    """可视化摄像头画面"""
    try:
        # 创建数据包装
        cameras_data = Data(agent_input.cameras)
        
        # 选择要可视化的帧（通常是当前帧）
        frame_idx = len(agent_input.cameras) // 2  # 中间帧，通常是当前帧
        
        # 生成摄像头可视化
        fig, ax = plot_cameras_frame(cameras_data, frame_idx)
        
        # 保存文件
        camera_save_path = save_dir / f"cameras_{token[:8]}.png"
        fig.savefig(camera_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Camera visualization saved to: {camera_save_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save camera visualization for token {token}: {e}")


def viz_bev(scene, agent, token, save_dir):
    """可视化BEV对比图"""
    try:
        # 生成BEV可视化图
        fig, ax = plot_bev_with_agent(scene, agent)
        
        # 添加标题信息
        ax.set_title(f"BEV Agent Comparison\nToken: {token[:8]}")
        
        # 保存文件
        bev_save_path = save_dir / f"bev_{token[:8]}.png"
        fig.savefig(bev_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"BEV visualization saved to: {bev_save_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save BEV visualization for token {token}: {e}")


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    # 初始化组件
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # 创建数据加载器
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    # ===== 创建可视化保存目录 =====
    viz_save_dir = Path("/proj/berzelius-2023-364/users/x_liali/DiffusionDrive/exp/traj_viz")
    viz_save_dir.mkdir(parents=True, exist_ok=True)

    # 获取要评估的tokens
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    
    # 如果没有找到tokens，使用硬编码的tokens
    if not tokens_to_evaluate:
        tokens_to_evaluate = [
            'e36edd3aedf05e30',
            'f99a74d444e651d3', 
            '67a6bdeb096350ec',
            '5b7700fa99d95a94',
            '564bb94f846e5fe1',
            '87efb8cf52135247',
            'f3e0463f3cf4505e',
            '16aa734bed8a5f81',
            '948e6a45c7cd5837',
            '5d68790fd55c5e41',
        ]
        # 过滤只保留实际存在的tokens
        tokens_to_evaluate = [t for t in tokens_to_evaluate if t in metric_cache_loader.tokens]

    pdm_results: List[Dict[str, Any]] = []
    
    for idx, token in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        
        try:
            # 加载metric cache
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            # 获取agent输入
            agent_input = scene_loader.get_agent_input_from_token(token)
            
            # 可视化摄像头画面
            viz_cameras(agent_input, token, viz_save_dir)
            
            # 计算轨迹
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
                
                # 可视化BEV对比图
                viz_bev(scene, agent, token, viz_save_dir)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            # 计算PDM分数
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
        
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    
    # 准备数据点
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    
    # 运行评估（直接调用而不是多进程，便于调试）
    score_rows = run_pdm_score(data_points)
    
    # 如果使用多进程，取消注释下面这行：
    # score_rows: List[Dict[str, Any]] = worker_map(worker, run_pdm_score, data_points)

    # 处理结果
    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    
    # 计算平均值
    numeric_columns = pdm_score_df.select_dtypes(include=[float, int]).columns
    average_row = pdm_score_df[numeric_columns].mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    
    # 添加平均行
    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame([average_row])], ignore_index=True)

    # 保存结果
    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df[pdm_score_df['token'] != 'average']['score'].mean() if 'score' in pdm_score_df.columns else 'N/A'}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()