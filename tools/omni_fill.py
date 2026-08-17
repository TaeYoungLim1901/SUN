"""OMNI 계열 채움값 → NaN.

제품마다 센티널이 다르다 (hourly vs HRO). 기본 목록은 합집합.
로드 직후 data_probe 히스토그램으로 실제 값을 보고 fills 를 덮어라.

사용:
    from tools.omni_fill import to_nan, selftest
    v = to_nan(array)
    python tools/omni_fill.py --selftest
"""
from __future__ import annotations

import argparse

import numpy as np

# hourly OMNI2 + HRO 1-min 에서 자주 나오는 센티널. 완전하지 않음.
FILLS = (
    999.9,
    999.99,
    9999.9,
    9999.99,
    99999.9,
    9999999.0,
    99.99,
)


def to_nan(a, fills=FILLS, abs_gt=None):
    """fills 에 가까운 값, 또는 abs_gt 이상 → NaN. 입력 dtype 유지하지 않고 float."""
    x = np.asarray(a, float).copy()
    for f in fills:
        x[np.isfinite(x) & (np.abs(x - f) < max(1e-3, 1e-6 * abs(f)))] = np.nan
    if abs_gt is not None:
        x[np.isfinite(x) & (np.abs(x) >= abs_gt)] = np.nan
    return x


def selftest():
    a = np.array([400.0, 9999.9, 999.9, 9999999.0, 99999.9, 350.0])
    b = to_nan(a)
    assert np.isfinite(b[0]) and np.isfinite(b[-1])
    assert np.isnan(b[1:5]).all()
    assert np.isclose(to_nan(np.array([400.0])), 400.0).all()
    print("omni_fill selftest 통과")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        p.print_help()
