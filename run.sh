#!/bin/bash
# 由 lmpgen.py 自动生成: 按配置顺序串行运行各任务 (任一步失败即停)
# 配置: lmpgen.json    生成时间: 2026-08-23 02:41
# 用法: bash run.sh
#       默认命令 mpirun -np 1 lmp < 输入文件, 可用环境变量覆盖:
#       MPIRUN=srun NP=8 LMP=lmp_gpu bash run.sh
# 集群: 在本文件开头自行加 #SBATCH/#PBS 头后提交
# 环境: 取消下行注释并按实际路径/名称修改
# source <conda路径>/etc/profile.d/conda.sh && conda activate <环境名>

set -euo pipefail

MPIRUN=${MPIRUN:-mpirun}
NP=${NP:-1}
LMP=${LMP:-lmp}

run() {
    echo "==> [$(date +%H:%M:%S)] $1/$2"
    (cd "$1" && "$MPIRUN" -np "$NP" "$LMP" < "$2")
}

run 1-opt in.opt.lammps
run 2-nvt in.nvt_500K.lammps

echo "==> 全部任务完成"
echo "提示: MSD 数据在 msd_*.dat, 对线性区拟合斜率/(6t) 即得扩散系数 D"
