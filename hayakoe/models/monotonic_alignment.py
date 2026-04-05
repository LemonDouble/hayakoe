"""
아래 함수의 주석은 리팩토링 시 GPT-4로 생성한 것으로, 코드와 완전히 일치하지 않을 수 있다.
"""

from typing import Any

import numba
import torch
from numpy import float32, int32, zeros


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    주어진 음의 중심값과 마스크를 사용하여 최대 경로를 계산한다

    Args:
        neg_cent (torch.Tensor): 음의 중심값을 나타내는 텐서
        mask (torch.Tensor): 마스크를 나타내는 텐서

    Returns:
        Tensor: 계산된 최대 경로를 나타내는 텐서
    """

    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)
    path = zeros(neg_cent.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    __maximum_path_jit(path, neg_cent, t_t_max, t_s_max)

    return torch.from_numpy(path).to(device=device, dtype=dtype)


@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)  # type: ignore
def __maximum_path_jit(paths: Any, values: Any, t_ys: Any, t_xs: Any) -> None:
    """
    주어진 경로, 값, 그리고 타겟 y/x 좌표를 사용하여 JIT으로 최대 경로를 계산한다

    Args:
        paths: 계산된 경로를 저장하기 위한 정수형 3차원 배열
        values: 값을 저장하기 위한 부동소수점형 3차원 배열
        t_ys: 타겟 y 좌표를 저장하기 위한 정수형 1차원 배열
        t_xs: 타겟 x 좌표를 저장하기 위한 정수형 1차원 배열
    """

    b = paths.shape[0]
    max_neg_val = -1e9
    for i in range(int(b)):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]

        v_prev = v_cur = 0.0
        index = t_x - 1

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index = index - 1
