#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学习型遥测外推器 —— harness 里 `predictor="learned"` 的实现。

背景
----
oracle 消融（真实 LLM qwen2.5:7b，n=100/格）:

    格        none    delta    oracle
    c1_s1     0/100   37/100   57/100
    c1_s2     0/100   37/100   57/100

确定性控制器上 `delta ≡ oracle`（都 0.8625），所以 **delta 与 oracle 之间那 20 点
是真实模型独有的**。本模块用监督学习训一个外推器去填它。

实测（`train_predictor.py`，测试集归一化 MAE）:

    零阶保持   0.0955
    两帧线性   0.0793   （现有 delta）
    学习型     0.0512   ← +35.4%

设计
----
* 输入 K=2 帧 × 66 个数值特征（与 harness 的 `_extrapolate(prev, cur)` 传参一致）
* 残差结构：`输出 = 最后一帧 + 网络学出的修正量`
  —— 线性外推就是这个残差等于 (cur − prev) 的特例，
  所以网络**至少不会比线性差**，除非它主动学坏。
* 只预测数值字段；布尔/结构字段原样保留（与 delta 同样的约束，
  否则 "up": true 会变成 1，语义改变——这个坑踩过一次）。
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

MAX_LINKS, MAX_FLOWS = 12, 6
LINK_FIELDS = ("util", "load_mbps", "delay_ms", "loss")
FLOW_FIELDS = ("delivered_mbps", "delay_ms", "loss")
N_FEAT = MAX_LINKS * len(LINK_FIELDS) + MAX_FLOWS * len(FLOW_FIELDS)

_CKPT: dict[str, Any] = {}
_MODEL = None


def _load(path: str | None = None):
    """惰性载入；只用 torch 的推理部分（CPU 或 GPU 均可）。"""
    global _MODEL, _CKPT
    if _MODEL is not None:
        return _MODEL
    import torch
    import torch.nn as nn

    path = path or os.environ.get(
        "AIXNET_LEARNED_PREDICTOR", "out/predictor_data/predictor.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    K, F, H = ck["window"], ck["n_feat"], ck["hidden"]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(K * F, H), nn.SiLU(),
                nn.Linear(H, H), nn.SiLU(),
                nn.Linear(H, F),
            )

        def forward(self, x):
            base = x[:, -1, :]
            return base + self.net(x.reshape(x.shape[0], -1))

    m = Net()
    m.load_state_dict(ck["net"])
    m.eval()
    _CKPT = {"K": K, "scale": np.asarray(ck["scale"], np.float32),
             "nmae": ck.get("nmae"), "linear_nmae": ck.get("linear_nmae")}
    _MODEL = m
    return m


def frame_vec(obs: dict) -> np.ndarray:
    v = np.zeros(N_FEAT, np.float32)
    p = 0
    for L in obs.get("links", [])[:MAX_LINKS]:
        for f in LINK_FIELDS:
            v[p] = float(L.get(f) or 0.0); p += 1
    p = MAX_LINKS * len(LINK_FIELDS)
    for F in obs.get("flows", [])[:MAX_FLOWS]:
        for f in FLOW_FIELDS:
            v[p] = float(F.get(f) or 0.0); p += 1
    return v


def _scatter(obs: dict, vec: np.ndarray) -> None:
    """把预测出的数值写回观测的浅拷贝（就地改，调用方负责拷贝）。"""
    p = 0
    for L in obs.get("links", [])[:MAX_LINKS]:
        for f in LINK_FIELDS:
            if f in L:
                L[f] = float(vec[p])
            p += 1
    p = MAX_LINKS * len(LINK_FIELDS)
    for F in obs.get("flows", [])[:MAX_FLOWS]:
        for f in FLOW_FIELDS:
            if f in F:
                F[f] = float(vec[p])
            p += 1


def predict(prev: dict, cur: dict, steps: int = 1) -> dict:
    """与 harness 的 `_extrapolate(prev, cur, steps)` 同签名。

    返回 `cur` 的一个浅拷贝，其中**数值字段**被替换为学出来的预测。
    布尔与结构字段原样保留。
    """
    import copy

    # ── 恒等检查 ─────────────────────────────────────────────
    # `steps == 0` 表示"不需要外推"（staleness=0 的格）。此时必须**原样返回**。
    # 初版无条件跑网络，于是即便不需要预测也把观测扰动了一遍（nMAE 0.051），
    # 在 c2_s0 / c2_s1 上把已经满分的策略从 1.000 打到 0.87–0.89。
    # 同一类错误在 `_extrapolate` 上犯过一次（布尔字段被算术化），
    # 所以这里显式写成恒等，并保留为回归检查。
    if steps <= 0:
        return cur

    out = copy.deepcopy(cur)

    def isnum(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    try:
        m = _load()
    except Exception:
        return out                     # 模型不可用就退回"什么都没预测"

    x = np.stack([frame_vec(prev), frame_vec(cur)])          # (K=2, F)
    scale = _CKPT["scale"]
    import torch
    with torch.no_grad():
        xn = torch.as_tensor((x / scale)[None, :, :], dtype=torch.float32)
        y = m(xn).numpy()[0] * scale
    if steps > 1:
        # 需要外推多步时，用"预测值 − 当前值"作为一步增量线性叠加
        y = frame_vec(cur) + (y - frame_vec(cur)) * steps
    _scatter(out, y.astype(np.float32))
    return out
