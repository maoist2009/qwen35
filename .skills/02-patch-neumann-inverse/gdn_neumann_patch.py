"""Validated Neumann-series approximate matrix inversion. Do not modify."""
from __future__ import annotations
import torch


def _strictly_lower_mask(k, device, dtype):
    return torch.tril(torch.ones(k, k, device=device, dtype=dtype), diagonal=-1)


def neumann_chunk_inverse(A, neumann_n=3, residual_s=8, chunk_size_check=64):
    *batch, k, k2 = A.shape
    assert k == k2, f"A must be square, got {A.shape}"
    if chunk_size_check is not None and k > chunk_size_check:
        raise ValueError(
            f"chunk dim k={k} exceeds safe threshold {chunk_size_check}; "
            f"high-order Neumann terms may overflow FP16/INT16 above this size."
        )
    I = torch.eye(k, device=A.device, dtype=A.dtype).expand(*batch, k, k)
    M = _strictly_lower_mask(k, A.device, A.dtype)

    T0 = I.clone()
    P = I.clone()
    for _ in range(neumann_n):
        P = P @ A
        T0 = T0 + P
    T0 = T0 * M + torch.diag_embed(torch.ones(*batch, k, device=A.device, dtype=A.dtype))

    IA = I - A
    E = I - IA @ T0

    T = I.clone()
    P = I.clone()
    for _ in range(residual_s):
        P = P @ E
        T = T + P

    return T @ T0


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.tril(torch.randn(64, 64) * 0.05, diagonal=-1)
    I = torch.eye(64)
    ref = torch.linalg.solve_triangular(I - A, I, upper=False, unitriangular=True)
    approx = neumann_chunk_inverse(A)
    err = (approx - ref).norm() / ref.norm()
    print("SELF_TEST_REL_ERR:", err.item())
    assert err < 1e-3
    print("PASS")
