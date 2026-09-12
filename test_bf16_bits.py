#!/usr/bin/env python3
"""验证指定 BF16 位模式转换为 FP32 后的位模式。"""

import torch


BF16_BITS = [0x7FFF, 0x7FF1, 0x7F81, 0xFFFF, 0x0FFF, 0xFF81]
EXPECTED_FP32_BITS = [value << 16 for value in BF16_BITS]


def unsigned_hex32(tensor: torch.Tensor) -> list[str]:
    """以无符号 32 位十六进制显示 int32 tensor。"""
    return [f"0x{value & 0xFFFFFFFF:08x}" for value in tensor.cpu().tolist()]


def test_on(device: str) -> None:
    # PyTorch 没有从整数数值直接构造任意 BF16 位模式的接口，因此先构造
    # uint16，再用 view 按位重解释为 bfloat16（不是数值类型转换）。
    bf16 = torch.tensor(BF16_BITS, dtype=torch.uint16, device=device).view(
        torch.bfloat16
    )
    fp32 = bf16.to(torch.float32)
    actual_bits = [value & 0xFFFFFFFF for value in fp32.view(torch.int32).cpu().tolist()]

    print(f"device={device}")
    for src, value, actual, expected in zip(
        BF16_BITS, fp32.cpu().tolist(), actual_bits, EXPECTED_FP32_BITS
    ):
        print(
            f"bf16=0x{src:04x} -> float={value!r:<12} "
            f"fp32_bits=0x{actual:08x}"
        )
        assert actual == expected, (
            f"0x{src:04x}: got 0x{actual:08x}, expected 0x{expected:08x}"
        )

    print("result:", unsigned_hex32(fp32.view(torch.int32)))


if __name__ == "__main__":
    test_on("cpu")
    if torch.cuda.is_available():
        test_on("cuda")

