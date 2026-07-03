"""
MatMul-only 近似替代 chunk-wise GatedDeltaNet 里的前向替换三角求逆。

依据: "When Good Enough Is Optimal: Multiplication-Only Matrix Inversion
Approximation for Quantized Gated DeltaNet" (arXiv:2606.06034, Qualcomm AI
Research, 2026-06)。

!!! 重要声明 !!!
这份实现是我根据论文正文/公式(4)-(8)与 Algorithm 1 的描述重新推导编写的，
不是作者发布的官方代码（论文本身没有公开代码仓库）。数值行为大方向应该
一致，但没有拿到官方实现做逐 bit 对比，落地前务必：
  1. 在 CPU/FP32 上跟原始前向替换求逆做数值对比（本文件底部提供了
     self-test），确认相对误差在可接受范围；
  2. 如果精度不达标，优先调大 residual_s（论文里在 INT8 下用到过 S=8~12），
     而不是调大 neumann_n（N 越大越容易在低精度下溢出，这是论文的核心论点）。
"""
from __future__ import annotations

import torch


def _strictly_lower_mask(k: int, device, dtype) -> torch.Tensor:
    return torch.tril(torch.ones(k, k, device=device, dtype=dtype), diagonal=-1)


def neumann_chunk_inverse(
    A: torch.Tensor,
    neumann_n: int = 3,
    residual_s: int = 8,
    chunk_size_check: int | None = 64,
) -> torch.Tensor:
    """
    近似计算 (I - A)^-1，其中 A 是严格下三角矩阵（对角线为 0）。

    Args:
        A: [..., k, k] 严格下三角矩阵（GatedDeltaNet chunk-wise 算法里
           beta * K @ K^T 经 mask 后的结果）。
        neumann_n: 截断阶数 N，论文默认 3。
        residual_s: 残差修正阶数 S，论文按 chunk_size 自适应，64 时约 4~8。
        chunk_size_check: 若提供，会检查 A.shape[-1] 是否超过安全阈值，
           防止在未做 block 拆分的情况下于 FP16/INT16 上溢出。

    Returns:
        T: [..., k, k] 近似的 (I - A)^-1。
    """
    *batch, k, k2 = A.shape
    assert k == k2, f"A 必须是方阵，got {A.shape}"
    if chunk_size_check is not None and k > chunk_size_check:
        raise ValueError(
            f"chunk 维度 k={k} 超过安全阈值 {chunk_size_check}；"
            f"N={neumann_n} 时高阶项在 FP16/INT16 下可能溢出（论文附录 H"
            f"实测 128x128/N=3 最坏情况达到 341376），请先做 block-wise 拆分。"
        )

    I = torch.eye(k, device=A.device, dtype=A.dtype).expand(*batch, k, k)
    M = _strictly_lower_mask(k, A.device, A.dtype)  # 对角局部化掩码

    # ---- 阶段一：截断纽曼级数 T0 = sum_{n=0}^{N} A^n，只用矩阵乘法累加 ----
    T0 = I.clone()
    P = I.clone()
    for _ in range(neumann_n):
        P = P @ A
        T0 = T0 + P
    T0 = T0 * M + torch.diag_embed(torch.ones(*batch, k, device=A.device, dtype=A.dtype))
    # 对角线保留单位阵，非对角部分按严格下三角掩码截断（对应论文的对角局部化结构）

    # ---- 阶段二：并行残差修正 ----
    # E = I - (I - A) @ T0 ；理想情况下 E -> 0，用 sum E^s 再修正一次
    IA = I - A
    E = I - IA @ T0

    T = I.clone()
    P = I.clone()
    for _ in range(residual_s):
        P = P @ E
        T = T + P

    return T @ T0


# ---------------------------------------------------------------------------
# self-test：和精确的前向替换求逆比较相对误差
# ---------------------------------------------------------------------------
def _reference_forward_substitution_inverse(A: torch.Tensor) -> torch.Tensor:
    k = A.shape[-1]
    I = torch.eye(k, device=A.device, dtype=A.dtype).expand_as(A)
    return torch.linalg.solve_triangular(I - A, I, upper=False, unitriangular=True)


def _self_test(chunk_size=64, neumann_n=3, residual_s=8, trials=20, seed=0):
    torch.manual_seed(seed)
    worst = 0.0
    for _ in range(trials):
        raw = torch.randn(chunk_size, chunk_size) * 0.05  # beta*K@K^T 量级通常较小
        A = torch.tril(raw, diagonal=-1)
        ref = _reference_forward_substitution_inverse(A)
        approx = neumann_chunk_inverse(A, neumann_n=neumann_n, residual_s=residual_s,
                                        chunk_size_check=chunk_size)
        rel_err = (approx - ref).norm() / (ref.norm() + 1e-8)
        worst = max(worst, rel_err.item())
    print(f"[self-test] chunk={chunk_size} N={neumann_n} S={residual_s} "
          f"trials={trials} worst_rel_err={worst:.6f}")
    return worst


if __name__ == "__main__":
    err = _self_test()
    assert err < 1e-3, "相对误差偏大，落地前请先调参（见文件顶部说明）后再继续 pipeline"
    print("[self-test] PASS")
