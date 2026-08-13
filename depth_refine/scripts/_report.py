"""손목(refine_wrist.py)·헤드(stereo_head.py) 비교 리포트 스크립트가 공유하는 헬퍼.

두 CLI 모두 "데이터셋을 순회 → 방법(들)을 실행 → GT 있으면 메트릭, 없으면 hole_ratio만
→ 프레임별 side-by-side PNG + metrics.csv(frame, method, mae, rmse, hole_ratio_pred,
runtime_ms) + 콘솔 요약표"라는 동일한 리포트 구조를 따른다(Task 8에서 refine_wrist.py로
먼저 구현되었고, Task 11에서 stereo_head.py와 공유하기 위해 이 모듈로 추출됨 — Task 8이
계약한 관측 가능한 출력, 특히 콘솔 skip 메시지 문구 `[skip] <이름>: <사유>`와
metrics.csv 스키마는 그대로 유지한다; 내부 구현(예: CSV를 프레임마다 스트리밍
기록하지 않고 다 모아 한 번에 쓰는 것)은 동일한 관측 가능 출력을 내는 한도 내에서
바뀌었다). 이름 그대로 두 스크립트의 내부 구현 세부사항이라 밑줄 프리픽스 모듈이며,
외부에서 임포트해 쓰라고 만든 공개 API가 아니다.
"""
from __future__ import annotations

import csv
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..common.depth_utils import depth_metrics, hole_ratio, valid_mask

METRICS_HEADER = ["frame", "method", "mae", "rmse", "hole_ratio_pred", "runtime_ms"]
SUMMARY_FIELDS = ("mae", "rmse", "hole_ratio_pred", "runtime_ms")


def select_methods(requested: Sequence[str],
                    available_fn: Callable[[], List[str]],
                    get_fn: Callable[[str], Any]) -> List[Tuple[str, Any]]:
    """요청된 이름들을 ``get_fn``으로 인스턴스화해 ``(이름, 인스턴스)`` 목록으로 해석.

    미등록 이름(``get_fn``이 던지는 KeyError) 또는 등록됐지만 ``available_fn()`` 결과에
    없는(비가용 — 무거운 의존성 미설치 등) 이름은 건너뛰고 ``[skip] <이름>: <사유>``를
    출력한다 — Task 8이 계약한 정확한 문구(``refine_wrist.py``의 원래 구현 그대로)이며
    ``stereo_head.py``도 동일하게 따른다(형식은 ``test_skip_message_matches_task8_
    contract``로 고정 — 이 함수를 고칠 때 그 형식만은 절대 깨면 안 된다). 예외는 던지지
    않는다. "선택 결과가 비면 실패로 볼지"는 호출부가 결정한다(refine_wrist는 요청된 여러
    방법 중 일부만 비어도 계속 진행, stereo_head는 매처/refiner 각각 단일 필수
    항목이라 비면 그 자리에서 실패 처리).

    비가용 사유 텍스트: 인스턴스(``obj``)의 ``unavailable_reason`` 속성(Task 14의 무거운
    모델 어댑터들이 ``is_available()``에서 채워두는, "왜 안 되는지" 한 줄 사유)이 있고
    비어있지 않으면 그것을 사유로 쓴다 — 없으면(고전/일반 refiner·matcher처럼 애초에 이
    속성이 없거나, 어떤 이유로 비어 있으면) 기존 고정 문구로 폴백한다. ``[skip] <이름>:
    <사유>`` 형식 자체는 그대로이고 사유 텍스트만 더 구체적으로 바뀐다.
    """
    avail = set(available_fn())
    selected: List[Tuple[str, Any]] = []
    for name in requested:
        try:
            obj = get_fn(name)
        except KeyError as e:
            reason = e.args[0] if e.args else str(e)
            print("[skip] {}: {}".format(name, reason))
            continue
        if name not in avail:
            reason = getattr(obj, "unavailable_reason", None) or \
                "registered but not available (missing dependencies)"
            print("[skip] {}: {}".format(name, reason))
            continue
        selected.append((name, obj))
    return selected


def frame_vmin_vmax(gt: Optional[np.ndarray], fallback: np.ndarray) -> Tuple[float, float]:
    """GT의 유효 픽셀(없거나 유효 픽셀이 하나도 없으면 fallback의 유효 픽셀)에서
    1/99 백분위수로 vmin/vmax 산출.

    프레임 내 모든 패널(입력/방법별 출력/GT)이 이 값을 공유해야 시각적으로 비교
    가능하므로, 프레임마다 한 번만 계산한다. 두 소스 모두 유효 픽셀이 전혀 없는
    (극단적) 경우에만 임의의 기본 범위로 폴백한다.
    """
    for candidate in (gt, fallback):
        if candidate is None:
            continue
        valid_vals = candidate[valid_mask(candidate)]
        if valid_vals.size > 0:
            return float(np.percentile(valid_vals, 1)), float(np.percentile(valid_vals, 99))
    return 0.0, 1.0


def metrics_row(frame_idx: int, method: str, pred_depth_m: np.ndarray,
                 gt_depth_m: Optional[np.ndarray], runtime_ms: float) -> Dict[str, Any]:
    """예측 깊이 1건에 대한 METRICS_HEADER 스키마 행 딕셔너리.

    GT가 있으면 depth_metrics()로 mae/rmse/hole_ratio_pred를 채운다. GT가 없으면
    mae/rmse는 NaN으로 기록하고(비교 불가), hole_ratio_pred는 GT 없이도 예측 자체
    에서 계산 가능하므로 hole_ratio()로 항상 채운다.
    """
    if gt_depth_m is not None:
        m = depth_metrics(pred_depth_m, gt_depth_m)
        mae, rmse, hrp = m["mae"], m["rmse"], m["hole_ratio_pred"]
    else:
        mae, rmse = float("nan"), float("nan")
        hrp = hole_ratio(pred_depth_m)
    return {
        "frame": frame_idx, "method": method, "mae": mae, "rmse": rmse,
        "hole_ratio_pred": hrp, "runtime_ms": runtime_ms,
    }


def print_summary(rows_by_method: Dict[str, List[Dict[str, Any]]]) -> None:
    """방법별 평균 mae/rmse/hole_ratio_pred/runtime_ms 콘솔 요약표.

    ``rows_by_method``의 키 순서대로 한 줄씩 출력한다 — 표시 순서는 호출부가
    딕셔너리를 채운 순서로 결정한다(예: 사용자가 지정한 --methods 순서).
    """
    print("{:<15}{:>12}{:>12}{:>22}{:>16}".format(
        "method", "mean_mae", "mean_rmse", "mean_hole_ratio_pred", "mean_runtime_ms"))
    with warnings.catch_warnings():
        # 프레임이 하나도 처리되지 않았거나 전 프레임 GT가 없어 mae/rmse가 전부 NaN인
        # 경우 nanmean이 내는 "Mean of empty slice"/all-NaN 경고를 억제한다 — 결과
        # 자체(NaN)는 그대로 두고 콘솔 노이즈만 없앤다.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for name, rows in rows_by_method.items():
            means = {
                field: (float(np.nanmean([r[field] for r in rows])) if rows else float("nan"))
                for field in SUMMARY_FIELDS
            }
            print("{:<15}{:>12.4f}{:>12.4f}{:>22.4f}{:>16.2f}".format(
                name, means["mae"], means["rmse"], means["hole_ratio_pred"],
                means["runtime_ms"]))


def write_metrics_csv(path, rows: List[Dict[str, Any]]) -> None:
    """rows(각 원소는 METRICS_HEADER 키를 가진 dict)를 헤더 포함 CSV로 저장."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def imwrite_or_raise(path, img: np.ndarray) -> None:
    """cv2.imwrite 실패를 조용히 무시하지 않고 IOError로 표면화.

    두 호출부 모두 프레임 패널 PNG(`frame_{idx:06d}.png`) 저장 전용이므로, 메시지
    문구는 refine_wrist.py의 원래 구현 그대로 "프레임 이미지 저장 실패"를 쓴다.
    """
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError("프레임 이미지 저장 실패: {}".format(path))
