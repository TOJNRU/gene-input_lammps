#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lmpgen.py — LAMMPS 输入文件生成器 (JSON 配置模式)

针对 ML 势 (DeePMD / MACE / EAM 等) 分子动力学模拟, 生成带完整中文注释的
LAMMPS 输入文件, 输出风格与手写脚本保持一致。

流程 (仿 neb-flow: 生成模板 -> 手动修改 -> 读取生成):
    python3 lmpgen.py init              # 1. 生成 lmpgen.json 模板 (标准 JSON,
    vim lmpgen.json                     #    每节首键 "_说明" 注释本节全部参数)
    python3 lmpgen.py from-json         # 2. 修改参数后批量生成输入文件
    python3 lmpgen.py from-json --only 2_nvt   # 只重新生成其中一步

JSON 结构: total 全局段 + 编号任务段 (1_opt, 2_nvt, ...),
每段可选 "dir" 指定输出目录; 任务段 "data": null 表示
自动引用上一步的输出结构文件 (链式工作流)。
势函数 potential 可写绝对路径, 生成时自动软链接到各任务目录下,
生成文件直接引用文件名 (任务目录自包含)。

任务类型:
    opt    结构优化 (弛豫原子位置, 可选 iso/aniso/tri 弛豫晶胞)
    nvt    NVT 分子动力学 (两阶段: 平衡 + 产线)
    npt    NPT 分子动力学 (恒温恒压)
    heat   线性升温扫描 (T1 -> T2, 识别相变/超离子转变)

固定原子: 所有任务均可用 freeze_type / freeze_expr 冻结部分原子
          (fix setforce 0 0 0), 控温只作用于可动原子组 mobile。
"""

import argparse
import datetime
import json
import math
import os
import random
import re
import shlex
import sys
from collections import Counter

# ============================================================
# 元素质量表 (IUPAC 标准值, amu)
# ============================================================
ELEMENT_MASSES = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Th": 232.04, "U": 238.03,
}

# 势函数注册表: style -> (pair_style 模板, pair_coeff 模板)
# {file} 为势文件路径, {els} 为空格分隔的元素符号 (顺序对应 data 文件类型)
POTENTIAL_STYLES = {
    "deepmd": ("deepmd {file}", "* * {els}"),
    "mace": ("mace", "* * {file} {els}"),
    "eam/alloy": ("eam/alloy", "* * {file} {els}"),
    "eam/fs": ("eam/fs", "* * {file} {els}"),
    "tersoff": ("tersoff", "* * {file} {els}"),
}

# 每种 units 的时间步长/阻尼时间默认值 (metal: ps, real: fs)
UNITS_DEFAULTS = {
    "metal": {"timestep": 0.001, "tdamp": 0.1, "pdamp": 1.0},
    "real": {"timestep": 1.0, "tdamp": 100.0, "pdamp": 1000.0},
}

TASK_DESC = {
    "opt": "结构优化",
    "nvt": "NVT 分子动力学 (两阶段: 平衡 + 产线)",
    "npt": "NPT 分子动力学 (恒温恒压)",
    "heat": "线性升温扫描",
}

# JSON 配置文件的键名 (与命令行参数一一对应)
COMMON_KEYS = ["data", "elements", "potential", "units", "boundary",
               "atom_style", "skin", "freeze_type", "freeze_expr",
               "extra", "extra_file", "no_check", "output", "dir"]
MD_KEYS = ["timestep", "seed", "tdamp", "equil", "prod",
           "thermo_every", "dump_every", "dump_elements", "dump_groups",
           "dump_atoms", "msd", "temps"]
TASK_KEYS = {
    "opt": ["relax", "fix_box", "pressure", "vmax", "min_style", "etol",
            "ftol", "maxiter", "maxeval", "thermo_every", "write_data"],
    "nvt": MD_KEYS,
    "npt": MD_KEYS + ["pressure", "baro", "pdamp"],
    # heat 自带温度区间 temp_start/temp_stop, 不用 temps
    "heat": [k for k in MD_KEYS if k != "temps"] + ["temp_start", "temp_stop"],
}
INT_KEYS = {"equil", "prod", "thermo_every", "dump_every", "maxiter",
            "maxeval", "seed"}
FLOAT_KEYS = {"temp_start", "temp_stop", "pressure", "vmax", "etol",
              "ftol", "timestep", "tdamp", "pdamp", "skin"}
# JSON 键的类型/枚举约束 (apply_json_to_ns 强制执行, 不受 no_check 影响;
# JSON 模式不走 argparse, choices 必须在这里等效重建)
STRING_KEYS = {"data", "potential", "output", "dir", "extra", "extra_file",
               "freeze_expr", "boundary", "atom_style", "msd", "write_data"}
DICT_KEYS = {"dump_groups", "dump_atoms"}
ENUM_CHOICES = {
    "units": list(UNITS_DEFAULTS),
    "relax": ["free", "iso", "aniso", "tri"],
    "min_style": ["cg", "fire", "sd", "quickmin"],
    "baro": ["iso", "aniso", "tri"],
}
LIST_OF_STR_KEYS = {"elements", "dump_elements"}
LIST_OF_INT_KEYS = {"freeze_type"}
BOOL_KEYS = {"no_check", "fix_box"}
# total 段的合法键 = 全部任务类型键的并集 (某键只要对任一任务类型有意义
# 即可放 total; 对不适用的任务段无副作用, 也不告警)
TOTAL_ALLOWED = sorted(set(COMMON_KEYS) |
                       {k for keys in TASK_KEYS.values() for k in keys})

JSON_NAME = "lmpgen.json"

BAR = "=" * 60
SEP = "-" * 60


# ============================================================
# 终端输出样式 (重定向或设置 NO_COLOR 时自动退化为纯文本)
# ============================================================

def paint(text, code):
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        return f"\033[{code}m{text}\033[0m"
    return text


def dim(text):
    return paint(text, "90")


def bold(text):
    return paint(text, "1")


def green(text):
    return paint(text, "32")


def yellow(text):
    return paint(text, "33")


def red(text):
    return paint(text, "31")


def c(name, *rest, comment=None):
    """生成一条 LAMMPS 命令, 命令名左对齐 15 列 (与手写脚本风格一致)。"""
    line = f"{name:<15}{' '.join(str(r) for r in rest)}"
    if comment:
        line = f"{line:<55}# {comment}"
    return line


def section(title, lines):
    """生成一个带标题分隔线的命令块。"""
    out = [f"# {SEP}", f"# {title}", f"# {SEP}"]
    out.extend(lines)
    return out


# ============================================================
# 命令块生成器
# ============================================================

def block_header(task, ns):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    keys = ["data", "elements", "potential", "relax", "fix_box", "temp",
            "temp_start", "temp_stop", "pressure", "equil", "prod",
            "freeze_type", "freeze_expr", "msd", "seed"]
    if task == "heat":
        keys = [k for k in keys if k != "temp"]
    else:
        keys = [k for k in keys if k not in ("temp_start", "temp_stop")]

    def fmt(v):
        return "/".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v

    summary = ", ".join(
        f"{k}={fmt(getattr(ns, k))}" for k in keys
        if getattr(ns, k, None) not in (None, [], "")
    )
    return [
        f"# {BAR}",
        f"# 任务: {TASK_DESC[task]}",
        f"# 生成时间: {now}   (lmpgen)",
        f"# 生成命令: {' '.join(sys.argv)}",
        f"# 参数: {summary}",
        f"# {BAR}",
    ]


def block_init(ns):
    lines = []
    art = getattr(ns, "_art", "")
    if art:
        # log none: 关闭启动时默认打开的 log.lammps, 避免同目录多任务
        # 在重定向前往默认日志写入残留内容
        lines += [c("log", "none"),
                  c("log", f"log_{art}.lammps",
                    comment="日志带任务标签, 避免同目录多任务互相覆盖")]
    lines += [
        c("units", ns.units),
        c("boundary", ns.boundary),
        c("atom_style", ns.atom_style),
        c("atom_modify", "map yes"),
        "",
        c("neighbor", ns.skin, "bin"),
        c("neigh_modify", "every 1 delay 0 check yes"),
    ]
    return section("初始化设置", lines)


def block_read_data(ns, data_info):
    lines = [
        c("read_data", ns.data),
        "",
        "# 元素顺序与 data 文件中的原子类型编号一一对应 (质量自动查表)",
    ]
    for i, el in enumerate(ns.elements, start=1):
        mass = ELEMENT_MASSES[el]
        lines.append(c("mass", i, mass, comment=el))
    n = data_info.get("natoms")
    if n:
        lines.append(f"# 体系: {n} 原子, {len(ns.elements)} 种类型 "
                     f"({'/'.join(ns.elements)})")
    return section("读取结构 + 质量", lines)


def block_potential(ns):
    style, path = ns._potential_style, ns._potential_file
    ps_tpl, pc_tpl = POTENTIAL_STYLES[style]
    lines = [
        c("pair_style", ps_tpl.format(file=path)),
        c("pair_coeff", pc_tpl.format(file=path, els=" ".join(ns.elements)),
          comment=f"{style}"),
    ]
    return section(f"势函数: {style}", lines)


def block_freeze(ns):
    """固定原子块: 冻结指定原子 (fix setforce 0 0 0)。

    MD 任务中速度初始化与控温只作用于 mobile 组, 冻结原子全程不动;
    优化任务中冻结原子受力清零, 位置保持不变。
    """
    lines = []
    if ns.freeze_type:
        types = " ".join(str(t) for t in ns.freeze_type)
        elmap = ", ".join(f"type {t}={ns.elements[t-1]}"
                          for t in ns.freeze_type
                          if t <= len(ns.elements))
        lines.append(c("group", "frozen", "type", types, comment=elmap))
    else:
        lines.append(c("group", "frozen", ns.freeze_expr))
    lines += [
        c("fix", "freeze", "frozen", "setforce", "0.0", "0.0", "0.0"),
        c("group", "mobile", "subtract", "all", "frozen"),
    ]
    return section("固定原子 (冻结组 frozen, 可动组 mobile)", lines)


def block_minimize(ns):
    # fix_box=true 或 relax="free" 都表示固定晶格参数, 仅弛豫原子位置
    fixed = getattr(ns, "fix_box", False) or ns.relax == "free"
    lines = [c("thermo", ns.thermo_every),
             c("thermo_style", "custom", "step", "temp", "pe", "ke",
               "etotal", "press", "vol")]
    if fixed:
        lines.append("")
        lines.append("# 晶格参数固定, 仅弛豫原子位置")
    else:
        lines.append("")
        lines.append(f"# 晶胞弛豫: {ns.relax} 模式, 目标外压 "
                     f"{ns.pressure} bar, 每步体积变化上限 {ns.vmax}")
        lines.append(c("fix", "boxrelax", "all", "box/relax", ns.relax,
                       ns.pressure, "vmax", ns.vmax))
    lines += [
        "",
        f"# 最小化: {ns.min_style} 法, 能量/力收敛判据 {ns.etol}/{ns.ftol}",
        c("min_style", ns.min_style),
        c("minimize", ns.etol, ns.ftol, ns.maxiter, ns.maxeval),
    ]
    if not fixed:
        lines.append(c("unfix", "boxrelax"))
    lines.append("")
    stem = os.path.splitext(os.path.basename(ns.data))[0]
    out = ns.write_data or f"{stem}_opt.dat"
    lines.append("# 输出优化后结构 (供后续 MD 使用)")
    lines.append(c("write_data", out))
    return section("优化设置" + (" (晶格固定)" if fixed else ""), lines)


def block_velocity(ns, target="all", tvar="${T}"):
    return c("velocity", target, "create", tvar, ns.seed, "dist gaussian")


def block_dump(ns):
    """产线阶段的输出: 全原子轨迹 + 元素/元素组/指定原子轨迹 (+ 可选 MSD)。

    dump_elements: 每种元素单独一个文件 traj_<元素>.xyz
    dump_groups:   几种元素合并一个文件, 如 {"anion": ["B","H"]} -> traj_anion.xyz
    dump_atoms:    指定原子 id, 如 {"jump": [12,45,78]} -> traj_jump.xyz
    同目录存在多个 MD 任务时 (_art 非空), 所有产物名追加任务标签防覆盖。
    """
    art = getattr(ns, "_art", "")

    def name(base, ext):
        return f"{base}_{art}.{ext}" if art else f"{base}.{ext}"

    cols = ["id", "type", "x", "y", "z", "c_pe_atom", "ix", "iy", "iz"]
    lines = [
        c("compute", "pe_atom", "all", "pe/atom"),
        "",
        "# 全原子轨迹 (id, type, 坐标, 每原子势能, image flags), 按 id 排序",
        c("dump", "d_all", "all", "custom", ns.dump_every,
          name("traj_all", "xyz"), *cols),
        c("dump_modify", "d_all", "sort", "id"),
    ]
    tmap = {el: i + 1 for i, el in enumerate(ns.elements)}

    # 轨迹输出元素与 MSD 元素统一建组 (组名 g_<元素>)
    els_group = list(dict.fromkeys(
        list(ns.dump_elements or []) + ([ns.msd] if ns.msd else [])))
    for el in els_group:
        lines += [
            "",
            c("group", f"g_{el}", "type", tmap[el], comment=f"{el} 原子"),
        ]
        if el in (ns.dump_elements or []):
            lines += [
                c("dump", f"d_{el}", f"g_{el}", "custom", ns.dump_every,
                  name(f"traj_{el}", "xyz"), *cols),
                c("dump_modify", f"d_{el}", "sort", "id"),
            ]

    # 几种元素合并输出 (dump_groups)
    for gname, els in (getattr(ns, "dump_groups", None) or {}).items():
        els = [el for el in els if el in tmap]
        if not els:
            continue
        types = " ".join(str(tmap[el]) for el in els)
        lines += [
            "",
            f"# {gname} 组轨迹 ({'+'.join(els)} 合并输出)",
            c("group", f"g_{gname}", "type", types,
              comment="+".join(els)),
            c("dump", f"d_{gname}", f"g_{gname}", "custom", ns.dump_every,
              name(f"traj_{gname}", "xyz"), *cols),
            c("dump_modify", f"d_{gname}", "sort", "id"),
        ]

    # 指定原子 id 输出 (dump_atoms)
    for gname, ids in (getattr(ns, "dump_atoms", None) or {}).items():
        id_str = " ".join(str(int(i)) for i in ids)
        lines += [
            "",
            f"# {gname} 组轨迹 (指定原子 id: {id_str})",
            c("group", f"g_{gname}", "id", id_str),
            c("dump", f"d_{gname}", f"g_{gname}", "custom", ns.dump_every,
              name(f"traj_{gname}", "xyz"), *cols),
            c("dump_modify", f"d_{gname}", "sort", "id"),
        ]

    if ns.msd:
        lines += [
            "",
            f"# MSD 实时计算 ({ns.msd} 组, 已扣除组内质心漂移), "
            f"每 {ns.dump_every} 步输出一次位移平方",
            c("compute", f"msd_{ns.msd}", f"g_{ns.msd}", "msd", "com", "yes"),
            c("fix", "msd_out", "all", "ave/time", ns.dump_every, "1",
              ns.dump_every, f"c_msd_{ns.msd}[4]", "file",
              name(f"msd_{ns.msd}", "dat")),
        ]
    return lines


def block_extra(ns):
    if not ns.extra and not ns.extra_file:
        return []
    lines = ["# ---- 用户自定义命令 (插入在势函数之后、任务设置之前) ----"]
    if ns.extra:
        lines.append(ns.extra)
    if ns.extra_file:
        try:
            with open(ns.extra_file) as f:
                lines.append(f.read().rstrip())
        except OSError as e:
            raise ValueError(f"extra_file 读取失败: {ns.extra_file} ({e})")
    return lines


# ============================================================
# data 文件解析与校验
# ============================================================

def parse_data_file(path):
    """解析 LAMMPS data 文件头部, 返回原子数/原子类型数等信息。"""
    info = {}
    with open(path, errors="replace") as f:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            m = re.match(r"\s*(\d+)\s+atoms\s*$", line)
            if m:
                info["natoms"] = int(m.group(1))
            m = re.match(r"\s*(\d+)\s+atom types\s*$", line)
            if m:
                info["ntypes"] = int(m.group(1))
    return info


class CheckError(Exception):
    pass


def validate(ns):
    """生成前校验, 返回解析到的 data 文件信息; 严重问题直接抛 CheckError。"""
    problems, warnings = [], []

    if ns.no_check:
        return {}

    # 1. data 文件
    data_path = getattr(ns, "_data_path", None) or ns.data
    if ns.data and not os.path.isfile(data_path):
        if getattr(ns, "_chained", False):
            warnings.append(f"链式输入 {ns.data} 尚不存在 (上一步运行后生成)")
        else:
            problems.append(f"结构文件不存在: {data_path}")
        data_info = {}
    elif ns.data:
        data_info = parse_data_file(data_path)
        ntypes = data_info.get("ntypes")
        if ntypes and ntypes != len(ns.elements or []):
            problems.append(
                f"元素数 ({len(ns.elements)}) 与 data 文件原子类型数 "
                f"({ntypes}) 不一致")
        if not data_info.get("natoms") and not ntypes:
            warnings.append(f"{data_path} 头部未找到 atoms/atom types 行, "
                            f"请确认是 LAMMPS data 文件")
    else:
        data_info = {}

    # 2. 元素符号
    for el in (ns.elements or []):
        if el not in ELEMENT_MASSES:
            problems.append(f"未知元素符号: {el} (质量表未收录)")

    # 3. 势函数
    pot_path = getattr(ns, "_potential_check_path", None) or ns._potential_file
    if ns._potential_file and not os.path.isfile(pot_path):
        warnings.append(f"势文件不存在: {pot_path} (集群路径可能不同)")

    # 4. 固定原子
    if ns.freeze_type and ns.freeze_expr:
        problems.append("freeze_type 与 freeze_expr 只能指定其一")
    if ns.freeze_type:
        ntypes = data_info.get("ntypes", len(ns.elements or []))
        for t in ns.freeze_type:
            if not 1 <= t <= ntypes:
                problems.append(
                    f"freeze_type 编号 {t} 超出类型范围 1-{ntypes}")

    # 5. 输出元素/元素组/指定原子 与 MSD
    dump_elements = getattr(ns, "dump_elements", None) or []
    dump_groups = getattr(ns, "dump_groups", None) or {}
    dump_atoms = getattr(ns, "dump_atoms", None) or {}
    msd = getattr(ns, "msd", None)
    for el in dump_elements + ([msd] if msd else []):
        if el not in (ns.elements or []):
            problems.append(f"元素 {el} 不在 elements 列表中")

    # 组名规则: 字母/下划线开头, 只含字母数字下划线; 不得与任何元素名、
    # 生成器保留组 (all/frozen/mobile) 或其他自定义组重名
    reserved = {"all", "frozen", "mobile"}
    used = set(ns.elements or []) | reserved
    for where, spec in (("dump_groups", dump_groups),
                        ("dump_atoms", dump_atoms)):
        if not isinstance(spec, dict):
            problems.append(f"{where} 必须是对象, 如 {{\"anion\": [\"B\",\"H\"]}}")
            continue
        for name, val in spec.items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(name)):
                problems.append(f"{where}.{name}: 组名只能用字母/数字/下划线, "
                                f"且不以数字开头")
            elif name in used:
                problems.append(f"{where}.{name}: 组名与已有组重名 ({name})")
            else:
                used.add(name)
            if not isinstance(val, list) or not val:
                problems.append(f"{where}.{name}: 必须是非空数组")
                continue
            if where == "dump_groups":
                for el in val:
                    if el not in (ns.elements or []):
                        problems.append(f"dump_groups.{name}: 元素 {el} "
                                        f"不在 elements 列表中")
            else:
                natoms = data_info.get("natoms")
                for i in val:
                    if not isinstance(i, int) or isinstance(i, bool) or i < 1:
                        problems.append(f"dump_atoms.{name}: 无效原子 id {i}"
                                        f" (需为正整数)")
                    elif natoms and i > natoms:
                        problems.append(f"dump_atoms.{name}: 原子 id {i} 超出"
                                        f"体系总数 {natoms}")

    # 6. 数值健全性 (temp/步长/步数等, 防止生成必挂的输入文件)
    def check_num(key, positive=False, nonneg=False):
        v = getattr(ns, key, None)
        if v is None:
            return
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            problems.append(f"{key} 需为有限数值, 当前为 {v!r}")
        elif positive and v <= 0:
            problems.append(f"{key} 必须为正数, 当前为 {v}")
        elif nonneg and v < 0:
            problems.append(f"{key} 不能为负数, 当前为 {v}")
        elif key in ("temp", "temp_start", "temp_stop") and v > 10000:
            problems.append(f"{key}={v} 超出合理范围 (通常 < 10000 K)")

    for key in ("temp", "temp_start", "temp_stop", "timestep", "tdamp",
                "pdamp", "skin", "vmax", "maxiter", "maxeval", "seed"):
        check_num(key, positive=True)
    for key in ("equil", "prod", "thermo_every", "dump_every"):
        # equil 允许 0 (跳过平衡), 其余必须为正
        check_num(key, positive=(key != "equil"), nonneg=(key == "equil"))

    # extra_file 需存在 (data 有检查, 这里补齐)
    extra_file = getattr(ns, "extra_file", None)
    if extra_file and not os.path.isfile(extra_file):
        problems.append(f"extra_file 不存在: {extra_file}")

    # 7. 物理常识提醒
    timestep = getattr(ns, "timestep", None)
    if (ns.elements and "H" in ns.elements and ns.units == "metal"
            and timestep is not None and timestep > 0.0015):
        warnings.append(
            f"含 H 体系时间步长 {timestep} ps 偏大, 建议 0.0005-0.001")

    for w in warnings:
        print(f"{yellow('!')} {w}")
    if problems:
        for p in problems:
            print(f"{red('✗')} {p}")
        raise CheckError("校验未通过, 未生成文件 (可用 no_check 跳过校验)")

    return data_info


# ============================================================
# 任务组装
# ============================================================

def assemble(task, ns, data_info):
    """按顺序组装命令块, 返回完整输入文件文本。"""
    blocks = [
        block_header(task, ns),
        block_init(ns),
        block_read_data(ns, data_info),
        block_potential(ns),
    ]
    # extra 紧跟势函数之后 (而非文件末尾): freeze_expr 引用的 region 等
    # 必须先定义后使用
    extra = block_extra(ns)
    if extra:
        blocks.append(extra)
    frozen = ns.freeze_type or ns.freeze_expr

    if task == "opt":
        if frozen:
            blocks.append(block_freeze(ns))
        blocks.append(block_minimize(ns))

    elif task in ("nvt", "npt", "heat"):
        md = [c("timestep", ns.timestep, comment="ps (metal) / fs (real)")]

        if task == "heat":
            md.append(c("variable", "T1", "equal", ns.temp_start))
            md.append(c("variable", "T2", "equal", ns.temp_stop))
            t_eq = "${T1}"
            desc = (f"线性升温: {ns.temp_start} -> {ns.temp_stop} K "
                    f"(先恒温平衡, 升温跨越产线全程)")
        else:
            md.append(c("variable", "T", "equal", ns.temp))
            t_eq = "${T}"
            desc = {"nvt": f"NVT 分子动力学: 温度 {ns.temp} K, "
                           f"阻尼 {ns.tdamp}",
                    "npt": f"NPT 分子动力学: {ns.temp} K, "
                           f"{getattr(ns, 'pressure', 0.0)} bar"}[task]

        # 速度初始化 + 系综 (有冻结原子时只作用于 mobile)
        target = "mobile" if frozen else "all"
        if frozen:
            blocks.append(block_freeze(ns))
            # 冻结原子无速度, 默认 thermo 的 temp 分母含其自由度, 读数严重
            # 偏低; 换用可动组温度, 否则用户会误以为控温失效
            md.append(c("compute", "Tmob", "mobile", "temp",
                        comment="可动组温度, thermo 输出用它"))
        temp_col = "c_Tmob" if frozen else "temp"
        md.append("")
        md.append("# 初始化速度 (随机种子记录于文件头, 高斯分布)")
        md.append(block_velocity(ns, target=target, tvar=t_eq))

        if task == "npt":
            md.append(c("variable", "P", "equal", ns.pressure))
            md.append("")
            md.append(f"# NPT 系综: 恒温 {ns.temp} K (阻尼 {ns.tdamp}), "
                      f"恒压 {ns.pressure} bar ({ns.baro}, 阻尼 {ns.pdamp})")
            # 注意: fix npt 自带控温与时间积分, 不能再叠加 fix nvt;
            # 冻结原子时 dilate mobile: 盒子伸缩只缩放可动原子坐标,
            # 冻结原子保持绝对坐标不动
            fix_args = ["temp", t_eq, t_eq, ns.tdamp,
                        ns.baro, "${P}", "${P}", ns.pdamp]
            if frozen:
                fix_args += ["dilate", "mobile"]
            md.append(c("fix", "integrator", target, "npt", *fix_args))
        else:
            md.append("")
            ens = {"nvt": "NVT 系综 (固定晶胞)",
                   "heat": f"NVT 系综: 先恒温 {ns.temp_start} K 平衡"}[task]
            md.append(f"# {ens}")
            md.append(c("fix", "thermostat", target, "nvt", "temp",
                        t_eq, t_eq, ns.tdamp))

        # ---- 平衡阶段 ----
        md += section(
            f"平衡阶段: {ns.equil} 步",
            [c("thermo", ns.thermo_every),
             c("thermo_style", "custom", "step", temp_col, "pe", "ke",
               "etotal", "press", "vol", "lx", "ly", "lz"),
             c("run", ns.equil)])
        md += ["", c("reset_timestep", "0")]

        # ---- 产线阶段 (heat: 换控温目标, 升温 T1 -> T2 跨越全程) ----
        if task == "heat":
            md += ["",
                   f"# 重新设置控温目标: {ns.temp_start} -> {ns.temp_stop} K "
                   f"线性升温, 跨越下方 run 的全部 {ns.prod} 步",
                   c("unfix", "thermostat"),
                   c("fix", "thermostat", target, "nvt", "temp",
                     "${T1}", "${T2}", ns.tdamp)]
        prod_title = "升温阶段" if task == "heat" else "产线阶段"
        prod = section(
            f"{prod_title}: {ns.prod} 步, 每 {ns.dump_every} 步输出一帧",
            block_dump(ns) + [c("run", ns.prod)])
        md += [""] + prod
        blocks.append(section(desc, md))

    text = "\n\n".join("\n".join(b) for b in blocks if b) + "\n"
    return text


def write_output(ns, text):
    if ns.output == "-":
        sys.stdout.write(text)
        return
    out_dir = os.path.dirname(ns.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(ns.output, "w") as f:
        f.write(text)


def default_output_name(task, ns):
    """默认输出名: MD 任务带温度标识, 如 in.nvt_500K.lammps / in.heat_300-800K.lammps。"""
    def kfmt(t):
        t = float(t)
        if abs(t) >= 1e6:
            return f"{t:g}K"     # 科学计数法, 避免超长文件名
        return f"{int(t)}K" if t == int(t) else f"{t}K"

    if task == "heat":
        return f"in.{task}_{kfmt(ns.temp_start)}-{kfmt(ns.temp_stop)}.lammps"
    if task in ("nvt", "npt") and getattr(ns, "temp", None) is not None:
        return f"in.{task}_{kfmt(ns.temp)}.lammps"
    return f"in.{task}.lammps"


def is_lmpgen_output(path):
    """判断文件是否本工具生成的输入文件 (文件头含 lmpgen 标记, 不可伪造)。"""
    try:
        with open(path, errors="replace") as f:
            head = "".join(f.readline(64) for _ in range(3))
    except OSError:
        return False
    return head.startswith("# ===") and "(lmpgen)" in head


def finalize_and_generate(task, ns):
    """填充按 units 区分的默认值 -> 校验 -> 组装 -> 写文件。

    命令行模式与 JSON 模式共用此入口; 返回是否成功。
    """
    # 必填参数缺失无法生成, 此检查不受 no_check 影响
    for attr, desc in [("data", "结构文件"), ("elements", "元素列表"),
                       ("potential", "势函数")]:
        if not getattr(ns, attr, None):
            print(f"{red('✗')} 缺少必填参数: {desc} (JSON 键 {attr})")
            return False

    # extra_file 需为普通文件且不过大 (fifo/设备文件会挂起或 OOM),
    # 此检查不受 no_check 影响
    ef = getattr(ns, "extra_file", None)
    if ef:
        if not os.path.isfile(ef):
            print(f"{red('✗')} extra_file 不存在或不是普通文件: {ef}")
            return False
        if os.stat(ef).st_size > 1_000_000:
            print(f"{red('✗')} extra_file 过大 (>1MB): {ef} "
                  f"(请确认没有误指向数据/模型文件)")
            return False

    # 成员/组名检查同样始终生效 (no_check 不跳过): 生成器依赖这些查表,
    # 不检查会在生成阶段 KeyError 崩溃或静默产出非法组名
    bad = [el for el in (ns.elements or []) if el not in ELEMENT_MASSES]
    if bad:
        print(f"{red('✗')} 未知元素符号: {', '.join(map(str, bad))} (质量表未收录)")
        return False
    els = ns.elements or []
    for el in (getattr(ns, "dump_elements", None) or []) + \
              ([ns.msd] if getattr(ns, "msd", None) else []):
        if el not in els:
            print(f"{red('✗')} 元素 {el} 不在 elements 列表中")
            return False
    for where, spec in (("dump_groups", getattr(ns, "dump_groups", None) or {}),
                        ("dump_atoms", getattr(ns, "dump_atoms", None) or {})):
        for gname in spec:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(gname)):
                print(f"{red('✗')} {where}.{gname}: 组名只能用字母/数字/下划线, "
                      f"且不以数字开头")
                return False
    for gname, gl in (getattr(ns, "dump_groups", None) or {}).items():
        for el in gl:
            if el not in els:
                print(f"{red('✗')} dump_groups.{gname}: 元素 {el} "
                      f"不在 elements 列表中")
                return False

    d = UNITS_DEFAULTS[ns.units]
    if getattr(ns, "timestep", None) is None:
        ns.timestep = d["timestep"]
    if getattr(ns, "tdamp", None) is None:
        ns.tdamp = d["tdamp"]
    if getattr(ns, "pdamp", None) is None:
        ns.pdamp = d["pdamp"]
    if getattr(ns, "seed", None) is None:
        ns.seed = random.randint(10000, 999999)
    if hasattr(ns, "temp"):
        ns.temp_start = getattr(ns, "temp_start", ns.temp)
        ns.temp_stop = getattr(ns, "temp_stop", ns.temp)
    if ns.potential:
        ns._potential_style, ns._potential_file = parse_potential(ns.potential)
    else:
        ns._potential_style, ns._potential_file = "", ""
    if not getattr(ns, "_data_path", None):
        ns._data_path = ns.data
    if not ns.output:
        ns.output = default_output_name(task, ns)

    try:
        data_info = validate(ns)
    except CheckError:
        return False

    # 覆盖保护: 强制重生成时, 拒绝覆盖非本工具生成的文件 (如手误把
    # output 写成 lmpgen.json / lmpgen.py / 结构文件)。
    # 用 lexists: 悬空软链接也会被拦截, 不会穿透到目标位置创建文件
    if ns.output != "-" and os.path.lexists(ns.output) \
            and not is_lmpgen_output(ns.output):
        print(f"{red('✗')} 拒绝覆盖 {ns.output}: 不是 lmpgen 生成的输入文件")
        return False

    try:
        text = assemble(task, ns, data_info)
    except ValueError as e:
        print(f"{red('✗')} {e}")
        return False

    try:
        write_output(ns, text)
    except OSError as e:
        print(f"{red('✗')} 写入 {ns.output} 失败: {e}")
        return False

    if ns.output != "-":
        print(f"{green('✔')} {ns.output}")
    return True


# ============================================================
# JSON 配置模式 (仿 neb-flow: init 生成模板 -> 手动修改 -> from-json 生成)
# 模板每节首键 "_说明" 集中注释参数; 读取时兼容 // 和 # 行注释
# ============================================================

def strip_jsonc_comments(text):
    """剥离 JSONC 风格的 // 与 # 行注释 (字符串内部的不受影响)。"""
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
        elif ch == '"':
            in_str = True
            out.append(ch)
            i += 1
        elif ch == "#" or (ch == "/" and text[i:i + 2] == "//"):
            while i < n and text[i] != "\n":
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def make_template(tasks):
    """构建模板配置 dict: 每节第一个键为 "_说明", 集中注释本节全部参数。

    "_说明" 是标准 JSON 键, 读取时自动忽略, 用户也可随手在里面补充备注。
    """
    total = [
        ("data", "STRUCTURE.dat", "结构文件, 路径相对项目根目录"),
        ("elements", ["B", "H", "Na"],
         "元素列表, 顺序对应 data 文件类型编号; 质量自动查表"),
        ("potential", "deepmd:/PATH/TO/MODEL.pt",
         "style:file; style 可选 deepmd | mace | eam/alloy | eam/fs | tersoff;"
         " file 可写绝对路径, 带 dir 的任务会自动软链接势文件到任务目录"),
        ("units", "metal", "metal | real (时间步长/阻尼默认值随单位制切换)"),
        ("boundary", "p p p", "边界条件: p=周期 f=固定, 如 'p p f'"),
        ("atom_style", "atomic", "原子风格, 一般保持 atomic"),
        ("skin", 2.0, "neighbor skin (A)"),
        ("freeze_type", [],
         "固定原子: 冻结的类型编号, 如 [1, 2] 冻结 B/H 骨架; [] = 不固定; 各任务段可覆盖"),
        ("freeze_expr", None,
         "固定原子: 按 group 表达式冻结, 如 'region bottom' (region 在 extra 中定义);"
         " 与 freeze_type 二选一, null = 不用"),
        ("extra", "",
         "追加自定义 LAMMPS 命令 (插入在势函数之后、任务设置之前, 可定义 region 等)"),
        ("extra_file", None, "同 extra, 从文件读取; null = 不用"),
        ("no_check", False, "true = 跳过生成前校验 (仅跳过物理校验, 类型检查仍生效)"),
    ]

    def md_items(drop=(), **over):
        base = {k: [k, v, n] for k, v, n in [
            ("timestep", 0.001, "时间步长 (ps, metal 单位)"),
            ("seed", None, "速度随机种子; null=自动随机, 实际值记录在生成文件头"),
            ("temps", 500,
             "温度 (必填): 单温度写数值如 500; 多温度写数组如 [400, 500, 600], "
             "每个温度独立子目录 (如 md/400K/)"),
            ("tdamp", 0.1, "控温阻尼 (ps)"),
            ("equil", 10000, "平衡阶段步数"),
            ("prod", 500000, "产线阶段步数"),
            ("thermo_every", 100, "热力学量输出间隔 (步)"),
            ("dump_every", 50, "轨迹输出间隔 (步)"),
            ("dump_elements", ["Na"], "额外单独输出轨迹的元素; [] = 不单独输出"),
            ("dump_groups", {},
             '几种元素合并输出一个轨迹文件, 如 {"anion": ["B", "H"]} -> traj_anion.xyz; {} = 不用'),
            ("dump_atoms", {},
             '按原子 id 输出轨迹, 如 {"jump": [12, 45, 78]} -> traj_jump.xyz; {} = 不用'),
            ("msd", "Na", "实时计算 MSD 的元素, 直接输出 msd_元素.dat; null = 不算"),
        ] if k not in drop}
        for k, v in over.items():
            base[k][1] = v
        return [tuple(x) for x in base.values()]

    tpl = {
        "opt": [
            ("dir", "DIR", "输出目录 (自动创建; 任务目录自包含, 可整体打包到集群)"),
            ("type", "opt", "任务类型: opt | nvt | npt | heat"),
            ("fix_box", False,
             "true = 固定晶格参数, 仅弛豫原子位置; false = 按 relax 弛豫晶胞"),
            ("relax", "iso", "晶胞弛豫模式: iso | aniso | tri | free (=固定晶格); fix_box=true 时忽略"),
            ("pressure", 0.0, "目标外压 (bar)"),
            ("vmax", 0.001, "box/relax 每步体积变化上限"),
            ("min_style", "cg", "最小化算法: cg | fire | sd | quickmin"),
            ("etol", 1e-12, "能量收敛判据"),
            ("ftol", 1e-12, "力收敛判据"),
            ("maxiter", 100000, "最大迭代步数"),
            ("maxeval", 1000000, "最大力计算次数"),
            ("thermo_every", 10, "热力学量输出间隔 (步)"),
            ("write_data", "STRUCTURE_opt.dat",
             "优化后结构输出名 (下一步 data 写 null 即自动引用它)"),
        ],
        "nvt": ([("dir", "DIR", "输出目录 (自动创建)"),
                 ("type", "nvt", "任务类型: opt | nvt | npt | heat")]
                + md_items()),
        "npt": ([("dir", "DIR", "输出目录 (自动创建)"),
                 ("type", "npt", "任务类型: opt | nvt | npt | heat")]
                + md_items(temps=300, dump_elements=[], msd=None) + [
                    ("pressure", 0.0, "目标压强 (bar)"),
                    ("baro", "iso", "控压模式: iso | aniso | tri"),
                    ("pdamp", 1.0, "控压阻尼 (ps)"),
                ]),
        "heat": ([("dir", "DIR", "输出目录 (自动创建)"),
                  ("type", "heat", "任务类型: opt | nvt | npt | heat")]
                 + md_items(drop=("temps",), msd=None) + [
                     ("temp_start", 300, "起始温度 (K), 平衡阶段恒温在此"),
                     ("temp_stop", 800, "终止温度 (K), 升温跨越整个产线阶段"),
                 ]),
    }

    def with_notes(items):
        """一节配置: 首键 _说明 (参数名 -> 注释), 其后为各参数及其值。"""
        sec = {"_说明": {k: note for k, _, note in items}}
        sec.update({k: v for k, v, _ in items})
        return sec

    conf = {"total": with_notes(total)}
    for i, t in enumerate(tasks, start=1):
        items = [(k, f"{i}-{t}" if k == "dir" else v, n)
                 for k, v, n in tpl[t]]
        if i > 1:
            items.insert(2, ("data", None,
                             "null = 链式引用上一步 opt 的输出; 也可写显式路径覆盖"))
        conf[f"{i}_{t}"] = with_notes(items)
    return conf


def json_init(args):
    """生成 JSON 配置模板 (已存在则跳过, --force 覆盖)。"""
    target = args.file
    if os.path.exists(target) and not args.force:
        print(f"{yellow('!')} {target} 已存在, 未覆盖 (使用 -f/--force 强制覆盖)")
        return 1

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        print(f"{red('✗')} --tasks 不能为空 (可选: {', '.join(TASK_DESC)})")
        return 1
    for t in tasks:
        if t not in TASK_DESC:
            print(f"{red('✗')} 未知任务类型 '{t}', 可选: {', '.join(TASK_DESC)}")
            return 1

    conf = make_template(tasks)
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(json.dumps(conf, indent=4, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"{red('✗')} 写入 {target} 失败: {e}")
        return 1

    run_cmd = "python3 lmpgen.py from-json" + \
              ("" if target == JSON_NAME else f" {target}")
    print(f"{green('✔')} {target}  任务: {' → '.join(tasks)}")
    print(f"  下一步: vim {target} 修改参数 → {run_cmd}")
    return 0


def apply_json_to_ns(ns, sec, allowed, where):
    """把一个 JSON 段的键值合并进 Namespace。

    集中式类型/枚举校验: 违规收集后统一抛 ValueError (调用方打印并中止),
    不受 no_check 影响 (类型错误与物理校验无关, 必须始终拦截)。
    未知键视为配置错误直接报错 (拼错键名会导致参数静默回落默认值,
    对科研输入文件比崩溃更危险; 自定义备注请用 _ 前缀键)。
    """
    problems = []

    def fail(key, msg):
        problems.append(f"{where}.{key}: {msg}")

    for key, val in sec.items():
        if key.startswith("_"):
            continue
        if key not in allowed:
            if key == "temp":
                fail(key, "已移除, 请改用 temps (单温度写数值如 500, "
                          "多温度写数组如 [400, 500])")
            else:
                fail(key, "未知键 (请检查拼写; 查看可用键请看模板 _说明, "
                          "自定义备注用 _ 前缀键)")
            continue
        if val is None:
            continue  # null 表示用默认值/自动处理
        if key in STRING_KEYS:
            if not isinstance(val, str):
                fail(key, f"需为字符串, 当前为 {type(val).__name__} 类型")
                continue
            if not val.strip() and key in ("data", "potential", "output",
                                           "dir", "extra_file", "write_data"):
                fail(key, "不能为空字符串")
                continue
            # 进入 LAMMPS 命令/文件名的键禁用危险字符 (空格/$/引号/换行等
            # 会破坏生成的命令行或注入运行时展开)
            if key in ("output", "dir", "write_data") and val != "-" \
                    and re.search(r"[\s$`\"'\\;|&<>(){}*?\[\]!~^\n\r]", val):
                fail(key, "含空格/$/引号等特殊字符, 会破坏生成的 LAMMPS 文件名")
        elif key in ENUM_CHOICES:
            if val not in ENUM_CHOICES[key]:
                fail(key, f"可选 {' | '.join(ENUM_CHOICES[key])}, 当前为 {val!r}")
        elif key in BOOL_KEYS:
            if not isinstance(val, bool):
                fail(key, f"需为 true/false, 当前为 {val!r}")
        elif key in DICT_KEYS:
            if not isinstance(val, dict):
                fail(key, '需为对象, 如 {"anion": ["B", "H"]}')
            else:
                for gname, gv in val.items():
                    if not isinstance(gv, list) or not gv:
                        fail(key, f"的 {gname} 需为非空数组")
                        continue
                    if key == "dump_groups" and not all(
                            isinstance(x, str) for x in gv):
                        fail(key, f"的 {gname} 需为元素符号数组, 如 [\"B\", \"H\"]")
                    elif key == "dump_atoms" and not all(
                            isinstance(x, int) and not isinstance(x, bool)
                            for x in gv):
                        fail(key, f"的 {gname} 需为原子 id 整数数组, 如 [12, 45]")
        elif key in LIST_OF_STR_KEYS:
            if not isinstance(val, list) or not all(
                    isinstance(x, str) for x in val):
                fail(key, '需为字符串数组, 如 ["B", "H"]')
        elif key in LIST_OF_INT_KEYS:
            if not isinstance(val, list) or not all(
                    isinstance(x, int) and not isinstance(x, bool)
                    for x in val):
                fail(key, "需为整数数组, 如 [1, 2]")
        elif key in INT_KEYS:
            if isinstance(val, bool) or not isinstance(val, (int, float)) \
                    or not math.isfinite(val):
                fail(key, f"需为有限数值, 当前为 {val!r}")
                continue
            if isinstance(val, float) and val != int(val):
                fail(key, f"需为整数, 当前为 {val}")
                continue
            val = int(val)
        elif key in FLOAT_KEYS:
            if isinstance(val, bool) or not isinstance(val, (int, float)) \
                    or not math.isfinite(val):
                fail(key, f"需为有限数值, 当前为 {val!r}")
                continue
            val = float(val)
        setattr(ns, key, val)

    if problems:
        raise ValueError("\n".join(problems))


def ref_from(path, base_dir):
    """返回 path 相对于 base_dir (lmp 实际运行目录) 的引用路径。"""
    if not base_dir or os.path.isabs(path):
        return path
    return os.path.relpath(path, base_dir)


def link_potential(path, dir_, sec_name, warnings):
    """把势文件软链接到任务目录 dir_ 下, 返回生成文件中引用的名字。

    绝对路径直接作为链接目标; 相对路径换算为相对 dir_ 的目标。
    目标文件暂不存在也创建链接 (如势文件在集群上, 项目同步过去后即可解析)。
    已有同名软链接则重建 (保证与配置一致); 同名实体文件/目录不覆盖, 告警。
    """
    name = os.path.basename(path)
    link = os.path.join(dir_, name)
    target = path if os.path.isabs(path) else os.path.relpath(path, dir_)
    if os.path.lexists(link):
        if os.path.islink(link):
            old = os.readlink(link)
            if old != target:
                warnings.append(f"{sec_name}: 软链接 {name} 从 {old} 改指 "
                                f"{target}, 此前生成的输入文件将使用新目标")
            os.remove(link)
        elif os.path.isdir(link):
            warnings.append(f"{sec_name}: {link} 存在同名目录, 未创建软链接, "
                            f"生成文件仍引用 {name}")
            return name
        else:
            warnings.append(f"{sec_name}: {link} 已存在同名实体文件, 未覆盖, "
                            f"生成文件将引用该文件")
            return name
    try:
        os.symlink(target, link)
    except OSError as e:
        warnings.append(f"{sec_name}: 创建软链接失败({e}), "
                        f"改用相对路径引用")
        return ref_from(path, dir_)
    return name


def strip_trailing_commas(text):
    """删除 JSON 尾逗号: 仅当逗号后 (跳过空白) 是 } 或 ] 时删除, 字符串内部不受影响。"""
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # 丢弃尾逗号
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def expand_temps(sections, total_dir=""):
    """展开温度参数 temps: nvt/npt 段的必填键。

    - 单温度写数值: "temps": 500 -> 单个任务, 布局扁平 (dir 不变)
    - 多温度写数组: "temps": [400, 500] -> 每个温度一个任务,
      各自独立子目录 <dir>/<温度>K (如 md/400K)
    展开后内部仍用 temp 承载单值 (仅实现细节, 不再是配置键)。
    其余键原样继承 (含 data:null 链式引用)。多温度的展开段键为
    <原键>_<温度>K, 供 --only 匹配; 单温度保留原段键。
    """
    out, errors, notes, orig_map = [], [], [], {}
    if not isinstance(total_dir, str):
        total_dir = ""
    for key, sec in sections:
        if not isinstance(sec, dict):
            out.append((key, sec))
            continue
        task = sec.get("type")
        if task not in ("nvt", "npt"):
            if sec.get("temps") is not None:
                errors.append(f"{key}.temps: 只能用于 nvt/npt 段 "
                              f"(当前 type={task!r})")
            out.append((key, sec))
            continue
        temps = sec.pop("temps", None)
        if temps is None:
            if "temp" in sec:
                errors.append(f"{key}.temp: 已移除, 请改用 temps "
                              f"(单温度写数值如 500, 多温度写数组如 [400, 500])")
            else:
                errors.append(f"{key}: 缺少 temps (单温度写数值如 500, "
                              f"多温度写数组如 [400, 500])")
            out.append((key, sec))
            continue
        single = not isinstance(temps, list)
        tlist = [temps] if single else temps
        if not tlist:
            errors.append(f"{key}.temps: 不能为空数组")
            continue
        bad = [t for t in tlist
               if isinstance(t, bool) or not isinstance(t, (int, float))
               or not math.isfinite(t) or t <= 0]
        if bad:
            errors.append(f"{key}.temps: 含无效温度 {bad} (需为正的有限数值)")
            continue
        if len(set(tlist)) != len(tlist):
            errors.append(f"{key}.temps: 温度列表存在重复")
            continue
        if sec.get("output") is not None:
            errors.append(f"{key}.temps: 展开任务不能指定 output "
                          f"(各任务按温度自动命名)")
            continue
        base = None
        if not single:
            base = sec.pop("dir", None)
            if not isinstance(base, str) or not base:
                base = total_dir
        expanded = []
        for t in tlist:
            sub = dict(sec)
            sub["temp"] = t
            tk = f"{int(t)}K" if float(t) == int(t) else f"{t}K"
            if not single:
                sub["dir"] = os.path.join(base, tk) if base else tk
                out.append((f"{key}_{tk}", sub))
                expanded.append(f"{key}_{tk}")
            else:
                out.append((key, sub))
        if single:
            notes.append(f"{key}: temps={tlist[0]} K (单温度)")
        else:
            prefix = f"{base}/" if base else ""
            notes.append(f"{key}: temps -> {len(tlist)} 个任务, 各温度独立目录 "
                         f"({prefix}{{{','.join((f'{int(t)}K' if float(t) == int(t) else f'{t}K') for t in tlist)}}})")
        if expanded:
            orig_map[key] = set(expanded)
    return out, errors, notes, orig_map


def json_run(args, parsers):
    """读取 JSON 配置, 依次生成各任务段的 LAMMPS 输入文件。"""
    try:
        with open(args.file, encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        print(f"{red('✗')} 读取 {args.file} 失败: 非 UTF-8 编码")
        return 1
    except OSError as e:
        print(f"{red('✗')} 读取 {args.file} 失败: {e}")
        return 1
    try:
        text = strip_jsonc_comments(raw)
        text = strip_trailing_commas(text)

        def _pairs(pairs):
            keys = [k for k, _ in pairs]
            dup = {k for k in keys if keys.count(k) > 1}
            if dup:
                print(f"{yellow('!')} 配置中存在重复键 "
                      f"{', '.join(sorted(dup))} (后者生效)")
            return dict(pairs)

        cfg = json.loads(text, object_pairs_hook=_pairs)
    except (json.JSONDecodeError, RecursionError) as e:
        print(f"{red('✗')} 解析 {args.file} 失败: {e}")
        print(f"  提示: 注释仅支持 // 或 # 行注释 (不支持 /* */); 另请检查逗号和引号")
        return 1
    if not isinstance(cfg, dict):
        print(f"{red('✗')} {args.file} 顶层必须是 JSON 对象")
        return 1

    total = cfg.get("total", {})
    if not isinstance(total, dict):
        print(f"{red('✗')} total 段必须是 JSON 对象")
        return 1

    sections = [(k, v) for k, v in cfg.items()
                if k != "total" and not k.startswith("_")]
    if not sections:
        print(f"{yellow('!')} 配置中没有任务段 (任务段的键不能以 _ 开头)")
        return 0

    # 温度序列批量展开 (temps) + 展开后段名查重
    sections, exp_errors, exp_notes, orig_map = expand_temps(
        sections, total.get("dir", "") if isinstance(total.get("dir"), str)
        else "")
    for e in exp_errors:
        print(f"{red('✗')} {e}")
    if exp_errors:
        return 1
    keys = [k for k, _ in sections]
    if len(set(keys)) != len(keys):
        print(f"{red('✗')} 展开后存在重复段名 (temps 展开名与显式段名冲突)")
        return 1

    if args.only is not None:
        names = set()
        for s in args.only.split(","):
            s = s.strip()
            if s:
                # 原段名选中整个温度序列, 展开名 (如 2_md_500K) 选中单个
                names |= orig_map.get(s, {s})
        if not names:
            print(f"{red('✗')} --only 不能为空")
            return 1
        unknown = names - {k for k, _ in sections}
        if unknown:
            print(f"{red('✗')} --only 引用了不存在的段: "
                  f"{', '.join(sorted(unknown))}")
            return 1

    print(f"{bold(args.file)} · {len(sections)} 个任务段")
    for note in exp_notes:
        print(dim(note))
    if "output" in total and total["output"] is not None:
        print(f"{yellow('!')} total.output: output 只能按任务段设置, 已忽略")

    prev_output = None   # (dir, write_data 引用) 上一步 opt 的输出
    n_ok, skipped, warnings = 0, [], []
    run_entries = []     # (运行目录, 输入文件) —— 串行脚本 run.sh 的执行清单
    seen_outputs = set()  # (归一化目录, 文件名) —— 段间输出名冲突检测

    # 预扫描: 同一目录下有多个任务时, 输出产物/日志名追加任务标签,
    # 防止 traj/msd/log 互相覆盖 (含 opt: 其最小化日志也会被后跑的任务覆盖)。
    # 目录键做 normpath 归一化 ("sim" 与 "./sim" 是同一物理目录)
    dir_task_count = Counter()
    for _, sec in sections:
        if isinstance(sec, dict):
            d = sec.get("dir")
            if not isinstance(d, str) or not d:
                td = total.get("dir")
                d = td if isinstance(td, str) and td else ""
            dir_task_count[os.path.normpath(d) if d else ""] += 1

    for key, sec in sections:
        if not key:
            print(f"{red('✗')} 存在空字符串的任务段键")
            return 1
        if not isinstance(sec, dict):
            print(f"{red('✗')} 段 {key} 必须是 JSON 对象")
            return 1
        task = sec.get("type")
        if task not in TASK_DESC:
            print(f"{red('✗')} 段 {key} 缺少有效 type "
                  f"(可选: {', '.join(TASK_DESC)})")
            return 1

        w_sec = len(warnings)   # 本段新增告警的起点 (无论跳过与否都要显示)

        # 展开产生的内部温度载体 (temps 的单值), 不作为配置键参与校验
        internal_temp = sec.pop("temp", None)

        # 默认值 <- total 覆盖 <- 任务段覆盖 (type 是任务选择键, 已单独解析)
        # total 也可携带任务专属键 (如 msd/dump_elements) 作为各任务默认值;
        # 但 output 例外: 各任务输出名不同, 全局同名会互相覆盖
        ns = parsers[task].parse_args([])
        try:
            # total 白名单 = 全部任务键并集: 某键只要对任一任务类型有效即可
            # 放 total (对不适用的任务段无副作用, 不产生告警噪音)
            apply_json_to_ns(ns, total, TOTAL_ALLOWED, "total")
            ns.output = None   # total 的 output 已在循环外统一告警并忽略
            apply_json_to_ns(ns, sec, COMMON_KEYS + TASK_KEYS[task] + ["type"],
                             key)
        except ValueError as e:
            for line in str(e).splitlines():
                print(f"{red('✗')} {line}")
            return 1
        if internal_temp is not None:
            ns.temp = internal_temp

        # 输出目录与文件名 (目录仅在实际生成时创建; 默认名带温度标识)
        # dir 优先级: 任务段 > total 段
        dir_ = sec.get("dir") or getattr(ns, "dir", "") or ""
        if not ns.output:
            ns.output = os.path.join(dir_, default_output_name(task, ns)) \
                if dir_ else default_output_name(task, ns)

        # output 与 dir 一致性: 裸文件名自动放进 dir; 显式目录与 dir
        # 不一致时报错 (否则输入文件和它引用的相对路径会落在不同目录)
        if ns.output and ns.output != "-":
            od = os.path.dirname(ns.output)
            if dir_:
                if not od:
                    ns.output = os.path.join(dir_, ns.output)
                elif os.path.normpath(od) != os.path.normpath(dir_):
                    print(f"{red('✗')} 段 {key}: output 的目录 ({od}) 与 "
                          f"dir ({dir_}) 不一致, 请二选一")
                    return 1
            elif od:
                dir_ = od   # output 自带子目录且未设 dir: 该目录即任务目录

        # 段间输出名冲突检测 (如两个 temps 序列共享温度、dir/output 重复)
        if ns.output != "-":
            okey = (os.path.normpath(dir_) if dir_ else "",
                    os.path.basename(ns.output))
            if okey in seen_outputs:
                print(f"{red('✗')} 段 {key} 的输出 {ns.output} 与之前的段同名 "
                      f"(检查 temps 序列是否共享温度, 或 dir/output 是否重复)")
                return 1
            seen_outputs.add(okey)

        # 同目录多任务: 产物名 (traj/msd) 与日志加任务标签
        ns._art = ""
        if ns.output != "-" and dir_task_count.get(
                os.path.normpath(dir_) if dir_ else "", 0) > 1:
            stem = os.path.basename(ns.output)
            stem = stem[:-7] if stem.endswith(".lammps") else stem
            ns._art = stem[3:] if stem.startswith("in.") else stem

        # 串行脚本条目: 记录 output 的真实目录 (output 可自带路径);
        # 仅在"本次生成"或"文件已存在"时收录, 避免 --only 后 run.sh
        # 引用从未生成的目录
        entry = None
        if ns.output != "-":
            entry = (os.path.dirname(ns.output) or ".",
                     os.path.basename(ns.output))

        # 已存在跳过: 输出文件已存在 (该步骤此前已生成) 则本段不再生成,
        # 反复运行 from-json 只补缺失的步骤; --only 显式点名或 --force 时强制生成
        force = args.force or bool(args.only)
        already = ns.output != "-" and os.path.isfile(ns.output)

        # data 解析: 显式路径 / null=链式引用上一步输出 / 缺省=total 的值
        # JSON 中的路径相对项目根目录; 文件内引用换算为相对 dir_ 的路径
        if "data" in sec and sec["data"] is None:
            if not prev_output:
                print(f"{red('✗')} 段 {key} 的 data 为 null, 但前面没有"
                      f"带 write_data 的任务")
                return 1
            prev_dir, prev_ref = prev_output
            full = os.path.join(prev_dir, prev_ref) if prev_dir else prev_ref
            ns.data = ref_from(full, dir_)
            ns._data_path = full
            ns._chained = True
        else:
            full = ns.data
            ns.data = ref_from(full, dir_) if dir_ else full
            ns._data_path = full

        # 记录本步输出 (供后续链式引用), write_data 与输入文件同目录
        if task == "opt":
            wref = ns.write_data or (
                os.path.splitext(os.path.basename(ns.data))[0] + "_opt.dat")
            prev_output = (dir_, wref)

        if already and not force:
            # 已存在但不是本工具生成的文件: 直接跳过会造成"已生成"的假象,
            # run.sh 还会引用它 —— 明确报错让用户处理
            if not is_lmpgen_output(ns.output):
                print(f"{red('✗')} {ns.output} 已存在但不是 lmpgen 生成的输入文件"
                      f" (检查是否与其他段冲突, 或手动处理该文件)")
                return 1
            skipped.append(key)
            print(f"{dim('─')} {key} 已存在, 跳过")
            if entry:
                run_entries.append(entry)
            # 跳过的段也做廉价检查: data 文件若已不存在, 及时提醒
            if ns.data and not getattr(ns, "_chained", False) \
                    and not os.path.isfile(ns._data_path):
                print(f"{yellow('!')} 结构文件不存在: {ns._data_path} "
                      f"(本段被跳过, 未重新校验)")
            for w in warnings[w_sec:]:
                print(f"{yellow('!')} {w}")
            continue

        if args.only and key not in names:
            for w in warnings[w_sec:]:
                print(f"{yellow('!')} {w}")
            if entry and os.path.isfile(ns.output):
                run_entries.append(entry)
            continue  # 不生成, 但链式信息继续向后传递

        if dir_:
            try:
                os.makedirs(dir_, exist_ok=True)
            except OSError as e:
                print(f"{red('✗')} 创建目录 {dir_} 失败: {e}")
                return 1

        # 势文件: 支持绝对路径; 带 dir 的任务把势文件软链接到该目录,
        # 生成文件直接引用文件名 (任务目录自包含), 校验用原始路径
        if ns.potential:
            style, path = parse_potential(ns.potential)
            ns._potential_check_path = path
            ref = link_potential(path, dir_, key, warnings) if dir_ else path
            ns.potential = f"{style}:{ref}"

        if not finalize_and_generate(task, ns):
            print(f"{red('✗')} 段 {key} 生成失败, 中止后续任务")
            return 1
        n_ok += 1
        if entry:
            run_entries.append(entry)
        for w in warnings[w_sec:]:
            print(f"{yellow('!')} {w}")

    # 生成串行任务脚本 run.sh (每次覆盖, 与配置保持同步)
    if run_entries:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "#!/bin/bash",
            "# 由 lmpgen.py 自动生成: 按配置顺序串行运行各任务 (任一步失败即停)",
            f"# 配置: {args.file}    生成时间: {now}",
            "# 用法: bash run.sh",
            "#       默认命令 mpirun -np 1 lmp < 输入文件, 可用环境变量覆盖:",
            "#       MPIRUN=srun NP=8 LMP=lmp_gpu bash run.sh",
            "# 集群: 在本文件开头自行加 #SBATCH/#PBS 头后提交",
            "# 环境: 取消下行注释并按实际路径/名称修改",
            "# source <conda路径>/etc/profile.d/conda.sh && conda activate <环境名>",
            "",
            "set -euo pipefail",
            "",
            "MPIRUN=${MPIRUN:-mpirun}",
            "NP=${NP:-1}",
            "LMP=${LMP:-lmp}",
            "",
            "run() {",
            '    echo "==> [$(date +%H:%M:%S)] $1/$2"',
            '    (cd "$1" && "$MPIRUN" -np "$NP" "$LMP" < "$2")',
            "}",
            "",
        ]
        lines += [f"run {shlex.quote(d)} {shlex.quote(f)}"
                  for d, f in run_entries]
        lines += ["", 'echo "==> 全部任务完成"',
                  'echo "提示: MSD 数据在 msd_*.dat, 对线性区拟合斜率/(6t) 即得扩散系数 D"',
                  ""]
        try:
            with open("run.sh", "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            os.chmod("run.sh", 0o755)
        except OSError as e:
            print(f"{red('✗')} 写入 run.sh 失败: {e}")
            return 1
        print(f"{green('✔')} run.sh ({len(run_entries)} 个任务串行)")

    parts = [f"生成 {n_ok}"]
    if skipped:
        parts.append(f"跳过 {len(skipped)} (-f 强制重生成)")
    if warnings:
        parts.append(f"警告 {len(warnings)}")
    print(f"完成: {' · '.join(parts)}")
    return 0


# ============================================================
# CLI
# ============================================================

def parse_potential(spec):
    """解析 'style:file' 形式的势函数参数, 默认 style 为 deepmd。"""
    if ":" in spec:
        style, path = spec.split(":", 1)
    else:
        style, path = "deepmd", spec
    if not path:
        raise SystemExit(f"{red('✗')} potential 缺少文件路径 (格式 style:file)")
    if style not in POTENTIAL_STYLES:
        raise SystemExit(
            f"{red('✗')} 不支持的势类型 '{style}', "
            f"可选: {', '.join(POTENTIAL_STYLES)}")
    return style, path


def add_common(p):
    p.add_argument("--data", help="LAMMPS data 文件")
    p.add_argument("--elements", nargs="+",
                   help="元素符号列表, 顺序对应 data 文件类型编号, 如: B H Na")
    p.add_argument("--potential",
                   help="势函数, 格式 style:file (默认 deepmd), "
                        "如 deepmd:frozen_omat24_plus.pt2")
    p.add_argument("--units", default="metal", choices=list(UNITS_DEFAULTS),
                   help="单位制 (默认 metal)")
    p.add_argument("--boundary", default="p p p", help="边界条件 (默认 p p p)")
    p.add_argument("--atom-style", default="atomic", dest="atom_style",
                   help="原子风格 (默认 atomic)")
    p.add_argument("--skin", type=float, default=2.0,
                   help="neighbor skin (默认 2.0 A)")
    p.add_argument("--freeze-type", type=int, nargs="+", metavar="TYPE",
                   default=[], help="冻结指定类型的原子, 如 --freeze-type 1 2")
    p.add_argument("--freeze-expr", metavar="EXPR",
                   help="冻结满足 LAMMPS group 表达式的原子, "
                        "如 --freeze-expr 'region bottom'")
    p.add_argument("--extra", default="", help="追加自定义命令 (单行字符串)")
    p.add_argument("--extra-file", help="追加自定义命令文件")
    p.add_argument("-o", "--output", help="输出文件 (默认 in.<task>.lammps, "
                                          "'-' 表示打印到屏幕)")
    p.add_argument("--no-check", action="store_true",
                   help="跳过生成前校验")


def add_md(p):
    p.add_argument("--timestep", type=float, default=None,
                   help="时间步长 (metal 默认 0.001 ps, real 默认 1.0 fs)")
    p.add_argument("--seed", type=int, default=None,
                   help="速度随机种子 (默认随机生成并记录在文件头)")
    p.add_argument("--temp", type=float, default=300, help="温度 K (默认 300)")
    p.add_argument("--tdamp", type=float, default=None,
                   help="控温阻尼 (metal 默认 0.1 ps)")
    p.add_argument("--equil", type=int, default=10000,
                   help="平衡步数 (默认 10000)")
    p.add_argument("--prod", type=int, default=50000,
                   help="产线步数 (默认 50000)")
    p.add_argument("--thermo-every", type=int, default=100, dest="thermo_every",
                   help="热力学量输出间隔 (默认 100)")
    p.add_argument("--dump-every", type=int, default=50, dest="dump_every",
                   help="轨迹输出间隔 (默认 50)")
    p.add_argument("--dump-elements", nargs="+", default=[], dest="dump_elements",
                   help="额外输出轨迹的元素, 如 --dump-elements Na")
    p.add_argument("--dump-groups", default={}, dest="dump_groups",
                   help="几种元素合并输出一个轨迹文件, 如 {'anion': ['B','H']}")
    p.add_argument("--dump-atoms", default={}, dest="dump_atoms",
                   help="按原子 id 输出轨迹, 如 {'jump': [12, 45, 78]}")
    p.add_argument("--msd", metavar="ELEMENT",
                   help="实时计算该元素组的 MSD (如 --msd Na)")


def make_task_parsers():
    """构建各任务类型的参数默认值表 (内部使用, 不暴露为命令行子命令)。

    from-json 读取配置段时, 以对应 parser 的默认 Namespace 为基础,
    再叠加 total 段与任务段的覆盖。
    """
    m = {}

    p = argparse.ArgumentParser()
    add_common(p)
    p.add_argument("--fix-box", action="store_true", default=False,
                   dest="fix_box",
                   help="固定晶格参数, 仅弛豫原子位置 (默认 false, 按 relax "
                        "模式弛豫晶胞)")
    p.add_argument("--relax", default="iso",
                   choices=["free", "iso", "aniso", "tri"],
                   help="晶胞弛豫模式: iso 各向同性 / aniso / tri "
                        "(默认 iso; free 等价于 fix_box=true)")
    p.add_argument("--pressure", type=float, default=0.0,
                   help="目标外压 bar (默认 0)")
    p.add_argument("--vmax", type=float, default=0.001,
                   help="box/relax 每步体积变化上限 (默认 0.001)")
    p.add_argument("--min-style", default="cg", dest="min_style",
                   choices=["cg", "fire", "sd", "quickmin"],
                   help="最小化算法 (默认 cg)")
    p.add_argument("--etol", type=float, default=1.0e-12)
    p.add_argument("--ftol", type=float, default=1.0e-12)
    p.add_argument("--maxiter", type=int, default=100000)
    p.add_argument("--maxeval", type=int, default=1000000)
    p.add_argument("--thermo-every", type=int, default=10, dest="thermo_every",
                   help="热力学量输出间隔 (默认 10)")
    p.add_argument("--write-data", dest="write_data",
                   help="优化后结构输出名 (默认 <data名>_opt.dat)")
    m["opt"] = p

    p = argparse.ArgumentParser()
    add_common(p)
    add_md(p)
    m["nvt"] = p

    p = argparse.ArgumentParser()
    add_common(p)
    add_md(p)
    p.add_argument("--pressure", type=float, default=0.0,
                   help="目标压强 bar (默认 0)")
    p.add_argument("--baro", default="iso", choices=["iso", "aniso", "tri"],
                   help="控压模式 (默认 iso)")
    p.add_argument("--pdamp", type=float, default=None,
                   help="控压阻尼 (metal 默认 1.0 ps)")
    m["npt"] = p

    p = argparse.ArgumentParser()
    add_common(p)
    add_md(p)
    p.add_argument("--temp-start", type=float, default=300, dest="temp_start",
                   help="起始温度 K (默认 300)")
    p.add_argument("--temp-stop", type=float, default=800, dest="temp_stop",
                   help="终止温度 K (默认 800)")
    m["heat"] = p

    return m


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python3 lmpgen.py",
        description="LAMMPS 输入文件生成器 (JSON 配置 -> 输入文件 + run.sh)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
用法 (三步):
  python3 lmpgen.py init        1. 生成配置模板 lmpgen.json
  vim lmpgen.json               2. 修改参数 (参数说明都在文件里)
  python3 lmpgen.py from-json   3. 生成输入文件和 run.sh, 之后 bash run.sh

详细文档: README.md
""")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="生成 JSON 配置模板")
    p.add_argument("-f", "--force", action="store_true",
                   help="覆盖已存在的配置文件")
    p.add_argument("--tasks", default="opt,nvt",
                   help="模板包含的任务类型, 逗号分隔 (默认 opt,nvt)")
    p.add_argument("--file", default=JSON_NAME,
                   help=f"配置文件名 (默认 {JSON_NAME})")

    p = sub.add_parser("from-json", help="读取 JSON 配置批量生成输入文件")
    p.add_argument("file", nargs="?", default=JSON_NAME,
                   help=f"配置文件 (默认 {JSON_NAME})")
    p.add_argument("--only", metavar="SEC[,SEC...]",
                   help="只生成指定段, 如 --only 1_opt 或 --only 1_opt,2_nvt "
                        "(显式指定的段总是重新生成)")
    p.add_argument("-f", "--force", action="store_true",
                   help="忽略已存在的输出, 强制重新生成全部")

    return ap


def main():
    ns = build_parser().parse_args()

    if ns.command == "init":
        sys.exit(json_init(ns))
    sys.exit(json_run(ns, make_task_parsers()))


if __name__ == "__main__":
    main()
