#!/usr/bin/env bash
#SBATCH -t 0-04:00:00  # 4小时运行时间
#SBATCH --gpus 1 -C "thin"
#SBATCH -J MetricCache
#SBATCH -o metric_cache_%j.out
#SBATCH -e metric_cache_%j.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

# 加载Mambaforge模块
module load Mambaforge/23.3.1-1-hpc1-bdist

# 激活conda环境
conda activate navsim

# 创建实验目录
mkdir -p $NAVSIM_EXP_ROOT/metric_cache

# 切换到项目目录
cd /proj/berzelius-2023-364/users/x_liali/DiffusionDrive

# 显示系统信息
echo "==================== System Info ===================="
nvidia-smi
echo "===================================================="

# 运行指定的脚本
echo "Running metric caching..."
python navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
