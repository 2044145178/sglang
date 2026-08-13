#!/usr/bin/env python3
"""Standalone NPU test and benchmark for DSpark build_out_tokens_triton.

This file intentionally does not import SGLang.  It contains only the Triton
kernel, its thin wrapper, and an independent Torch oracle.

Examples:

    python3 test_build_out_tokens_triton.py
    python3 test_build_out_tokens_triton.py --bs 32 --replays 100
    python3 test_build_out_tokens_triton.py --bs 1,4,16,64 --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

try:
    import torch_npu  # noqa: F401  # Registers the NPU backend.
except ImportError as exc:
    raise SystemExit(f"torch_npu is required: {exc}") from exc

import triton
import triton.language as tl


@triton.jit
def _build_out_tokens_kernel(
    draft_tokens_ptr,
    correct_len_ptr,
    bonus_ptr,
    out_ptr,
    gamma,
    T,
    n_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    b = offs // T
    k = offs % T

    cl = tl.load(correct_len_ptr + b, mask=mask, other=0).to(tl.int32)
    bonus = tl.load(bonus_ptr + b, mask=mask, other=0)
    draft_mask = mask & (k < gamma)
    draft = tl.load(
        draft_tokens_ptr + b * gamma + k,
        mask=draft_mask,
        other=0,
    )

    # Two stores avoid the nested tl.where predicate-type lowering issue seen
    # with Triton-Ascend.  Every output element belongs to exactly one program;
    # the second store only overwrites the accepted boundary with the bonus.
    tl.store(out_ptr + offs, draft.to(tl.int64), mask=mask)
    tl.store(out_ptr + offs, bonus.to(tl.int64), mask=mask & (k == cl))


def build_out_tokens_triton(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    _validate_inputs(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=verify_num_draft_tokens,
        gamma=gamma,
    )
    bs = draft_tokens.shape[0]
    T = verify_num_draft_tokens
    device = draft_tokens.device

    draft_tokens_i = draft_tokens.to(torch.int64).contiguous()
    correct_len_i = correct_len.to(torch.int64).contiguous()
    bonus_i = bonus.to(torch.int64).contiguous()
    out = torch.empty((bs, T), dtype=torch.int64, device=device)

    n_out = bs * T
    block = 256
    grid = (triton.cdiv(n_out, block),)
    _build_out_tokens_kernel[grid](
        draft_tokens_i,
        correct_len_i,
        bonus_i,
        out,
        gamma,
        T,
        n_out,
        BLOCK=block,
    )
    return out


def build_out_tokens_torch(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    _validate_inputs(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=verify_num_draft_tokens,
        gamma=gamma,
    )
    bs = draft_tokens.shape[0]
    out = torch.empty(
        (bs, verify_num_draft_tokens),
        dtype=torch.int64,
        device=draft_tokens.device,
    )
    out[:, :gamma].copy_(draft_tokens)
    out[:, gamma].fill_(0)
    out.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
    return out


def _validate_inputs(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> None:
    if verify_num_draft_tokens != gamma + 1:
        raise ValueError(
            "build_out_tokens requires verify_num_draft_tokens == gamma + 1; "
            f"got T={verify_num_draft_tokens}, gamma={gamma}"
        )
    if draft_tokens.ndim != 2 or tuple(draft_tokens.shape[1:]) != (gamma,):
        raise ValueError(
            f"draft_tokens must have shape [bs, {gamma}], got "
            f"{tuple(draft_tokens.shape)}"
        )
    bs = draft_tokens.shape[0]
    if tuple(correct_len.shape) != (bs,) or tuple(bonus.shape) != (bs,):
        raise ValueError(
            f"correct_len and bonus must have shape [{bs}], got "
            f"{tuple(correct_len.shape)} and {tuple(bonus.shape)}"
        )
    if not (draft_tokens.device == correct_len.device == bonus.device):
        raise ValueError("all inputs must be on the same device")


def _sync() -> None:
    torch.npu.synchronize()


def _compact_error(exc: BaseException, limit: int = 2000) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= limit:
        return text
    marker = f"\n... <truncated {len(text) - limit} chars> ...\n"
    keep = limit - len(marker)
    return text[: keep * 2 // 3] + marker + text[-keep // 3 :]


def _make_inputs(
    *, device: torch.device, bs: int, gamma: int, seed: int, replay: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + replay)

    # Construct on CPU so input generation is outside the tested NPU kernels.
    draft = torch.randint(
        0, 129_280, (bs, gamma), dtype=torch.int64, generator=generator
    )
    if replay == 0:
        # Deterministically cover correct_len=0 and correct_len=gamma.
        correct_len = torch.arange(bs, dtype=torch.int64) % (gamma + 1)
    else:
        correct_len = torch.randint(
            0, gamma + 1, (bs,), dtype=torch.int64, generator=generator
        )
    bonus = torch.randint(
        0, 129_280, (bs,), dtype=torch.int64, generator=generator
    )
    return draft.to(device), correct_len.to(device), bonus.to(device)


def _run_parity(
    *, device: torch.device, bs: int, gamma: int, replays: int, seed: int
) -> None:
    T = gamma + 1
    for replay in range(replays):
        draft, correct_len, bonus = _make_inputs(
            device=device, bs=bs, gamma=gamma, seed=seed, replay=replay
        )
        expected = build_out_tokens_torch(
            draft_tokens=draft,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=T,
            gamma=gamma,
        )
        actual = build_out_tokens_triton(
            draft_tokens=draft,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=T,
            gamma=gamma,
        )
        _sync()
        try:
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        except AssertionError as exc:
            print(f"[FAIL] bs={bs} replay={replay}", file=sys.stderr)
            print(f"correct_len={correct_len.cpu().tolist()}", file=sys.stderr)
            print(f"draft={draft.cpu().tolist()}", file=sys.stderr)
            print(f"bonus={bonus.cpu().tolist()}", file=sys.stderr)
            print(f"expected={expected.cpu().tolist()}", file=sys.stderr)
            print(f"actual={actual.cpu().tolist()}", file=sys.stderr)
            raise RuntimeError("Torch/Triton parity mismatch") from exc
    print(f"[PASS] parity bs={bs} gamma={gamma} replays={replays}")


def _benchmark(
    *, device: torch.device, bs: int, gamma: int, warmup: int, iters: int, seed: int
) -> None:
    T = gamma + 1
    draft, correct_len, bonus = _make_inputs(
        device=device, bs=bs, gamma=gamma, seed=seed, replay=10_000
    )
    kwargs = dict(
        draft_tokens=draft,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=T,
        gamma=gamma,
    )

    def measure(fn) -> float:
        for _ in range(warmup):
            fn(**kwargs)
        _sync()
        begin = time.perf_counter()
        for _ in range(iters):
            fn(**kwargs)
        _sync()
        return (time.perf_counter() - begin) * 1e6 / iters

    torch_us = measure(build_out_tokens_torch)
    triton_us = measure(build_out_tokens_triton)
    speedup = torch_us / triton_us if triton_us else float("inf")
    print(
        f"[BENCH] bs={bs} gamma={gamma} torch={torch_us:.3f} us "
        f"triton={triton_us:.3f} us speedup={speedup:.3f}x"
    )


def _parse_bs(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("--bs must contain positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--bs", type=_parse_bs, default=[1, 4, 16, 64])
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gamma <= 0 or args.replays <= 0:
        raise SystemExit("--gamma and --replays must be positive")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("NPU is not available")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    print(
        f"[CONFIG] device={device} bs={args.bs} gamma={args.gamma} "
        f"T={args.gamma + 1}"
    )

    try:
        for bs in args.bs:
            _run_parity(
                device=device,
                bs=bs,
                gamma=args.gamma,
                replays=args.replays,
                seed=args.seed,
            )
            if args.benchmark:
                _benchmark(
                    device=device,
                    bs=bs,
                    gamma=args.gamma,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                )
    except BaseException as exc:
        print(f"[ERROR] {_compact_error(exc)}", file=sys.stderr)
        return 1

    print("[SUMMARY] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
