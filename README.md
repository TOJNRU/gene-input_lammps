# lmpgen — LAMMPS 输入文件生成器

单文件 Python 工具（零第三方依赖，Python 3.7+），面向 ML 势（DeePMD / MACE /
EAM 等）分子动力学模拟：用一份 JSON 配置批量生成带中文注释的 LAMMPS 输入
文件和串行执行脚本 `run.sh`，输出风格与手写脚本一致。

## 快速开始

```bash
python3 lmpgen.py init              # 1. 生成配置模板 lmpgen.json
vim lmpgen.json                     # 2. 修改参数 (每个参数的说明就在文件里)
python3 lmpgen.py from-json         # 3. 生成输入文件 + run.sh
bash run.sh                         # 4. 串行运行全部任务
```

其他命令：

```bash
python3 lmpgen.py init --tasks opt,npt,nvt,heat   # 自定义模板任务组合
python3 lmpgen.py init -f                         # 强制覆盖已有配置文件
python3 lmpgen.py from-json my.json               # 指定配置文件
python3 lmpgen.py from-json --only 2_nvt          # 只重新生成指定段 (总是强制;
                                                   #   temps 段可用原名选整个序列或展开名单温度)
python3 lmpgen.py from-json -f                    # 忽略已存在, 全部重新生成
```

## 任务类型

| 类型 | 用途 | 专属键 |
|------|------|--------|
| `opt` | 结构优化（0 K 能量最小化） | `fix_box` `relax` `pressure` `vmax` `min_style` `etol` `ftol` `maxiter` `maxeval` `thermo_every` `write_data` |
| `nvt` | 恒温恒容 MD（平衡 + 产线两阶段） | MD 公共键 + `temps` |
| `npt` | 恒温恒压 MD（盒子随压强伸缩） | MD 公共键 + `temps` `pressure` `baro` `pdamp` |
| `heat` | 线性升温扫描（找相变/超离子转变） | MD 公共键（不含 temp）+ `temp_start` `temp_stop` |

## 配置文件 lmpgen.json

结构：`total` 全局段 + 编号任务段（`1_opt`、`2_nvt`、`3_nvt` …），任务段可
覆盖 `total` 的同名键；`total` 也可以携带任务专属键（如 `msd`、`dump_elements`、
`equil`）作为各任务的默认值（`output` 除外——只能按任务段设置）。
段数和类型不限，按文件顺序依次生成。

```json
{
    "total": {
        "data": "NaBH.dat",
        "elements": ["B", "H", "Na"],
        "potential": "deepmd:/home/user/models/frozen_model.pt2"
    },
    "1_opt": { "dir": "1-opt", "type": "opt",
               "write_data": "NaBH_opt.dat" },
    "2_nvt": { "dir": "2-nvt", "type": "nvt", "data": null,
               "temps": 500, "prod": 500000, "msd": "Na" }
}
```

### 参数说明都在文件里

模板每节第一个键是 `"_说明"`：值为一个对象，逐条注释本节全部参数的含义和
可选值（如 `"relax": "iso | aniso | tri"`），照着选即可。所有 `_` 开头的键
读取时一律忽略——可以随手用 `_备注` 之类的键记录实验信息，会一直保留。
文件是纯标准 JSON，任何编辑器/工具都不报错（读取时兼容 `//`、`#` 行注释
和尾逗号；重复键会告警，后者生效）。

### 链式工作流

任务段 `"data": null` 表示自动引用**上一步 opt** 的 `write_data` 输出，路径
自动换算（如生成 `read_data ../1-opt/NaBH_opt.dat`）；改名/插段后下游自动
跟上。也可写显式路径覆盖。JSON 里所有路径相对项目根目录（运行 `from-json`
的位置）。

### 势函数

`"potential": "style:file"`，style 支持 `deepmd` | `mace` | `eam/alloy` |
`eam/fs` | `tersoff`。**推荐 file 写绝对路径**：带 `dir` 的任务会把势文件
软链接进自己的目录，生成文件直接引用文件名，任务目录自包含，整个项目打包
到集群即可运行。势文件暂不在本地（集群路径）也能创建软链接；重复生成时
链接自动重建，目录下已有同名实体文件则不覆盖、仅告警。

### 固定原子

`freeze_type: [1, 2]`（按类型编号冻结，如冻 B/H 骨架只让 Na 扩散）或
`freeze_expr: "region bottom"`（任意 LAMMPS group 表达式，region 经 `extra`
定义即可，extra 会插入在使用之前），二选一，四种任务通用。冻结原子通过
`fix setforce 0 0 0` 实现，速度初始化与控温只作用于可动组 `mobile`；
NPT 下自动加 `dilate mobile`（盒子伸缩只缩放可动原子，冻结原子保持绝对
坐标不动）；thermo 温度读数换用 mobile 组温度（冻结原子无速度，全体系
温度分母含其自由度，读数会严重偏低）。

### 轨迹输出

MD 任务始终输出全原子轨迹，`dump_every` 控制间隔；每帧内容为
`id type x y z 每原子势能 ix iy iz`（含 image flags，便于离线处理周期性），
按 id 排序。文件为 LAMMPS custom dump 格式（扩展名沿用 .xyz 的历史习惯）。
三种补充输出：

```json
"dump_elements": ["Na"],                       // 每种元素单独一个文件 traj_Na.xyz
"dump_groups":   {"framework": ["B", "H"]},    // 几种元素合并一个文件 traj_framework.xyz
"dump_atoms":    {"jump": [12, 45, 78]}        // 指定原子 id traj_jump.xyz
```

`"msd": "Na"` 让 LAMMPS 实时计算该元素组的 MSD，直接输出 `msd_Na.dat`
（已扣除组内质心漂移），免后处理。

**同目录多任务自动隔离**：同一目录下有多个 MD 任务时（如温度序列），
所有产物文件名自动追加任务标签，避免互相覆盖——`traj_all.xyz` →
`traj_all_nvt_400K.xyz`、`msd_Na.dat` → `msd_Na_nvt_400K.dat`、日志重定向到
`log_nvt_400K.lammps`（并先 `log none` 关闭默认日志）。各任务独立目录时不
重定向，使用 LAMMPS 默认的 `log.lammps`，产物保持原名。

### 默认命名与已存在跳过

默认输出名带系综和温度：`in.nvt_500K.lammps`、`in.npt_300K.lammps`、
`in.heat_300K-800K.lammps`、`in.opt.lammps`（无温度）；显式指定 `output`
则完全按指定。**输出文件已存在的段自动跳过**——反复运行 `from-json` 只补
缺失的步骤，不会覆盖已完成步骤；`--only` 点名的段总是重新生成，`-f` 全部强制。

## 键名参考

| 段 | 键 | 默认 | 说明 |
|----|-----|------|------|
| total / 任务 | `data` | 必填 | 结构文件；任务段写 `null` = 链式引用 |
| | `elements` | 必填 | 元素列表，顺序对应 data 文件类型编号，质量自动查表 |
| | `potential` | 必填 | `style:file`，file 可为绝对路径 |
| | `units` / `boundary` / `atom_style` / `skin` | metal / p p p / atomic / 2.0 | |
| | `freeze_type` / `freeze_expr` | `[]` / `null` | 固定原子（二选一） |
| | `extra` / `extra_file` | 空 | 追加自定义命令（插入在势函数之后、任务设置之前，可定义 region 等） |
| | `no_check` | false | 跳过生成前校验 |
| | `dir` / `output` | 无 / 自动 | 输出目录 / 输出文件名 |
| 任务 | `type` | 必填 | opt / nvt / npt / heat |
| opt | `fix_box` | false | true = 固定晶格参数仅弛豫位置 |
| | `relax` | iso | iso / aniso / tri（`free` 等价 `fix_box: true`） |
| | `pressure` | 0.0 | 目标外压 (bar) |
| | `vmax` / `min_style` | 0.001 / cg | box/relax 体积上限 / cg、fire、sd、quickmin |
| | `etol` / `ftol` / `maxiter` / `maxeval` | 1e-12 / 1e-12 / 1e5 / 1e6 | 收敛判据 |
| | `thermo_every` / `write_data` | 10 / `<data名>_opt.dat` | |
| MD (nvt/npt) | `temps` | 必填 | 温度：单温度写数值（`500`，布局扁平）；多温度写数组（`[400, 500]`，每温度独立子目录 `md/400K/` 式）。heat 不用 temps，用 temp_start/temp_stop |
| | `timestep` | 按单位制 | metal 0.001 ps；real 1.0 fs |
| | `seed` | null | 速度种子；null = 随机（实际值记录在生成文件头） |
| | `tdamp` | 按单位制 | 控温阻尼；metal 0.1 ps |
| | `equil` / `prod` | 10000 / 50000 | 平衡 / 产线步数 |
| | `thermo_every` / `dump_every` | 100 / 50 | 输出间隔 |
| | `dump_elements` / `dump_groups` / `dump_atoms` | [] / {} / {} | 三种轨迹输出 |
| | `msd` | null | 实时 MSD 输出的元素 |
| npt | `pressure` / `baro` / `pdamp` | 0.0 / iso / 按单位制 (metal 1.0) | 压强 / iso-aniso-tri / 阻尼 |
| heat | `temp_start` / `temp_stop` | 300 / 800 | 升温区间（平衡段恒温在 temp_start） |

## 运行：run.sh

每次 `from-json` 同步（覆盖）生成 `run.sh`，按配置顺序串行运行各任务，
任一步失败立即停止（保证 opt 成功后才会跑依赖它的 MD）。默认命令
`mpirun -np 1 lmp < 输入文件`，启动器/进程数/可执行文件可用环境变量覆盖：

```bash
bash run.sh                                # mpirun -np 1 lmp < 输入文件
NP=8 bash run.sh                           # mpirun -np 8 lmp < ...
MPIRUN=srun NP=4 LMP=lmp_gpu bash run.sh   # srun -np 4 lmp_gpu < ...
```

集群提交：在 `run.sh` 开头加 `#SBATCH`/`#PBS` 头（模板里有一行现成的
conda 环境激活注释，按需取消注释）后 `sbatch run.sh`。

## 生成前自动校验

- **类型与枚举校验始终生效**（不受 `no_check` 影响）：units/relax/min_style/baro
  的可选值、字符串/数组/数值类型、inf/NaN、未知键一律报错（拼错键名会直接
  拦截，不会静默回落默认值——自定义备注请用 `_` 前缀键）
- 物理校验（`no_check: true` 时跳过）：结构文件存在性与格式、元素数与
  data 文件原子类型数一致性、温度合理范围（< 10000 K）、势文件存在性
  （告警）、含 H 体系时间步长偏大提醒
- freeze / dump_groups / dump_atoms 的元素范围、组名合法性（不得与任何元素名
  或保留组 all/frozen/mobile 重名）、原子 id 范围；重复键告警（后者生效）
- 覆盖保护：强制重生成时拒绝覆盖非 lmpgen 生成的文件（防止 output 手误
  指向 lmpgen.json / 结构文件等）

## 常见场景

**温度序列（Arrhenius）**——推荐用 `temps` 批量语法，每个温度独立子目录：

```json
"2_arrhenius": {"dir": "md", "type": "nvt", "data": null,
                "temps": [400, 500, 600],
                "prod": 500000, "msd": "Na"}
```

自动展开为 3 个任务，分别放在 `md/400K/`、`md/500K/`、`md/600K/` 独立目录
（未设 dir 时为根下 `<温度>K/`），共享其余参数，链式引用与势文件链接自动
换算到各级子目录。改某个温度用 `--only 2_arrhenius_500K`，整序列重生成用
`--only 2_arrhenius`。传统写法（每温度一段、`temp` 单值）仍然支持。

**相变扫描**：`heat` 从 `temp_start` 线性升到 `temp_stop`（升温跨越产线全程，
`prod` 给足、升温速率 ≲1 K/ps），跑完从 log 的能量/体积随温度曲线和
`msd_元素.dat` 找跳变/onset；反转 start/stop 即降温线，可看回滞。

**固定骨架看扩散上限**：`freeze_type` 冻结阴离子骨架 + `msd` 挂阳离子，
得无骨架弛豫贡献的扩散系数上限。
