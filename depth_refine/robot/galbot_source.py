"""Galbot SDK 1.9.0 어댑터 — 손목(D405)·헤드 스테레오 프레임을 FrameSource 계약으로 노출.

SDK는 로봇에서만 설치되어 있다. 이 모듈은 SDK를 지연 임포트
(`importlib.import_module(os.environ.get("GALBOT_SDK_MODULE", "galbot_sdk"))`)하므로
SDK가 없는 개발 PC에서도 이 파일을 import하는 것 자체는 항상 안전하다 — 실패는
`GalbotSource()`를 실제로 생성할 때만 명확한 RuntimeError로 드러난다(§7 우아한 비활성화).

이 파일의 모든 SDK 호출은 공식 문서 기반으로 작성했고 아직 로봇에서 실행해보지
못했다. 실 SDK 메시지 필드명은 로봇에서 record.py --dry-run으로 1회 검증 필요 —
어긋나는 부분이 있으면 아래 `_decode_rgb`/`_decode_depth`/`_to_intrinsics`/
`_synced_pair`/`_acquire_robot` 중 해당 메서드 하나만 고치면 된다(온-로봇 단일 수정
지점으로 격리한 설계).
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from .interface import FrameSource, HeadPair, WristFrame

_SIDES = ("left", "right")
_DEFAULT_DEPTH_SCALE = 1000.0  # raw uint16 -> m 환산 스케일. record.py --depth-scale로 조정 가능.


def _msg_field(msg: Any, key: str) -> Any:
    """메시지에서 key 필드를 읽는다 — dict형(구독 접근)과 속성 접근 메시지를 모두 지원한다.

    실 SDK가 dict를 주는지 객체를 주는지는 로봇에서 검증하기 전까지 확정할 수 없어
    (§ 모듈 독스트링) 양쪽을 다 시도하는 관용적 접근을 쓴다.
    """
    try:
        return msg[key]
    except (TypeError, KeyError, IndexError):
        return getattr(msg, key)


def _pose_vec_to_list(pose_vec: Any) -> Any:
    """포즈 벡터를 JSON 직렬화 가능한 리스트로 변환. extrinsics_sdk.json은 참고값일 뿐이므로
    변환이 실패해도 레코딩을 중단시키지 않고 문자열로 대체해 기록한다."""
    try:
        return np.asarray(pose_vec, dtype=np.float64).reshape(-1).tolist()
    except Exception:
        return str(pose_vec)


class GalbotSource(FrameSource):
    """Galbot SDK로 손목(side 팔)·헤드 프레임을 공급하는 FrameSource 구현.

    Args:
        side: 손목 카메라로 쓸 팔 ("left" 또는 "right", 기본 "left").
        depth_scale: 손목 깊이 raw uint16 payload -> 미터 환산 나눗셈 값
            (SDK 문서: "divided by depth_scale". 기본 1000.0 = mm 단위 가정,
            record.py --depth-scale로 조정 가능).
    """

    def __init__(self, side: str = "left", depth_scale: float = _DEFAULT_DEPTH_SCALE) -> None:
        if side not in _SIDES:
            raise ValueError("side must be 'left' or 'right', got {!r}".format(side))
        self.side = side
        self._depth_scale = float(depth_scale)

        module_name = os.environ.get("GALBOT_SDK_MODULE", "galbot_sdk")
        try:
            self._sdk = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                "Galbot SDK를 찾을 수 없습니다. 로봇에서 실행 중인지, GALBOT_SDK_MODULE "
                "환경변수가 맞는지 확인하세요. (module={!r}: {})".format(module_name, exc)
            ) from exc

        self._robot = self._acquire_robot()

    # ---------------- SDK 접근 지점 (로봇 검증 후 수정할 땐 이 메서드들만 보면 됨) ----------------
    def _acquire_robot(self) -> Any:
        """SDK 문서의 'GalbotRobot.get_instance() 후 initialize()' 획득 패턴."""
        try:
            robot = self._sdk.GalbotRobot.get_instance()
            initialize = getattr(robot, "initialize", None)
            if callable(initialize):
                initialize()
            return robot
        except Exception as exc:
            raise RuntimeError(
                "Galbot 로봇 인스턴스 초기화에 실패했습니다 (GalbotRobot.get_instance()/"
                "initialize() 절차를 로봇에서 확인하세요): {}".format(exc)
            ) from exc

    def _decode_rgb(self, msg: Any) -> np.ndarray:
        """압축 RGB 메시지(msg['data']) -> BGR uint8 이미지 (cv2 관례, 이 저장소 전체와 일관)."""
        data = _msg_field(msg, "data")
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(
                "RGB 메시지 디코딩 실패 — msg['data']가 유효한 압축 이미지가 아닙니다. "
                "record.py --dry-run으로 실제 메시지 구조를 확인하세요."
            )
        return img

    def _decode_depth(self, msg: Any, width: int, height: int) -> np.ndarray:
        """16UC1 깊이 메시지(msg['data']) -> float32 미터 깊이 (self._depth_scale로 환산)."""
        data = _msg_field(msg, "data")
        raw = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
        return raw.astype(np.float32) / self._depth_scale

    def _to_intrinsics(self, raw: Any) -> CameraIntrinsics:
        """get_camera_intrinsic() 반환값 -> CameraIntrinsics. 속성 접근/딕셔너리 키 둘 다 지원."""
        fx = float(_msg_field(raw, "fx"))
        fy = float(_msg_field(raw, "fy"))
        cx = float(_msg_field(raw, "cx"))
        cy = float(_msg_field(raw, "cy"))
        width = int(_msg_field(raw, "width"))
        height = int(_msg_field(raw, "height"))
        return CameraIntrinsics(fx, fy, cx, cy, width, height)

    def _synced_pair(self, sensor_types: List[Any]) -> Any:
        """동기화 관측 API 호출. 메서드명 자체가 문서상 확정적이지 않아(get_synced_observation
        으로 가정) 이 메서드 하나로 격리한다 — 첫 항목이 anchor, 나머지는 최근접 시각 매칭."""
        get_synced = getattr(self._robot, "get_synced_observation", None)
        if get_synced is None:
            raise NotImplementedError(
                "Galbot SDK에서 동기화 관측 API를 찾지 못했습니다(get_synced_observation으로 "
                "가정) — 로봇에서 실제 메서드명을 확인해 GalbotSource._synced_pair()만 수정하면 "
                "됩니다."
            )
        return get_synced(sensor_types)

    def _wrist_sensor_types(self) -> Tuple[Any, Any]:
        """self.side에 따른 (rgb SensorType, depth SensorType)."""
        SensorType = self._sdk.SensorType
        if self.side == "left":
            return SensorType.LEFT_ARM_CAMERA, SensorType.LEFT_ARM_DEPTH_CAMERA
        return SensorType.RIGHT_ARM_CAMERA, SensorType.RIGHT_ARM_DEPTH_CAMERA

    # ---------------- FrameSource 계약 ----------------
    def get_wrist_frame(self) -> WristFrame:
        rgb_type, depth_type = self._wrist_sensor_types()
        msg_rgb = self._robot.get_rgb_data(rgb_type)
        msg_depth = self._robot.get_depth_data(depth_type)
        intr = self._to_intrinsics(self._robot.get_camera_intrinsic(rgb_type))

        rgb = self._decode_rgb(msg_rgb)
        depth = self._decode_depth(msg_depth, intr.width, intr.height)
        ts_rgb = int(_msg_field(msg_rgb, "header"))
        ts_depth = int(_msg_field(msg_depth, "header"))

        return WristFrame(rgb=rgb, depth_m=depth, intrinsics=intr,
                           ts_rgb_ns=ts_rgb, ts_depth_ns=ts_depth, gt_depth_m=None)

    def get_head_pair(self) -> HeadPair:
        SensorType = self._sdk.SensorType
        left_type = SensorType.HEAD_LEFT_CAMERA
        right_type = SensorType.HEAD_RIGHT_CAMERA

        obs = self._synced_pair([left_type, right_type])
        rgb_map = _msg_field(obs, "rgb_data_map")
        msg_l = rgb_map[left_type]
        msg_r = rgb_map[right_type]

        left = self._decode_rgb(msg_l)
        right = self._decode_rgb(msg_r)
        ts_l = int(_msg_field(msg_l, "header"))
        ts_r = int(_msg_field(msg_r, "header"))

        return HeadPair(left=left, right=right, ts_left_ns=ts_l, ts_right_ns=ts_r,
                         gt_depth_left_m=None)

    def head_intrinsics(self) -> Tuple[CameraIntrinsics, CameraIntrinsics]:
        SensorType = self._sdk.SensorType
        intr_l = self._to_intrinsics(self._robot.get_camera_intrinsic(SensorType.HEAD_LEFT_CAMERA))
        intr_r = self._to_intrinsics(self._robot.get_camera_intrinsic(SensorType.HEAD_RIGHT_CAMERA))
        return intr_l, intr_r

    def get_head_extrinsics_sdk(self) -> Dict:
        """head/extrinsics_sdk.json 참고값(§4) — record.py가 그대로 저장한다. 스테레오
        캘리브레이션(calibrate_head.py)은 이 값과 무관하게 체커보드로 별도 수행한다."""
        SensorType = self._sdk.SensorType
        pose_l, ts_l = self._robot.get_sensor_extrinsic(SensorType.HEAD_LEFT_CAMERA)
        pose_r, ts_r = self._robot.get_sensor_extrinsic(SensorType.HEAD_RIGHT_CAMERA)
        return {
            "left": {"pose_vec": _pose_vec_to_list(pose_l), "ts_ns": int(ts_l)},
            "right": {"pose_vec": _pose_vec_to_list(pose_r), "ts_ns": int(ts_r)},
        }

    # ---------------- dry-run 진단 전용 (디코드 전 raw 메시지 노출) ----------------
    def get_wrist_raw(self) -> Dict[str, Any]:
        """record.py --dry-run 전용: 디코드하지 않은 원본 SDK 메시지를 반환한다."""
        rgb_type, depth_type = self._wrist_sensor_types()
        return {
            "rgb": self._robot.get_rgb_data(rgb_type),
            "depth": self._robot.get_depth_data(depth_type),
        }

    def get_head_raw(self) -> Dict[str, Any]:
        """record.py --dry-run 전용: 디코드하지 않은 원본 SDK 메시지를 반환한다."""
        SensorType = self._sdk.SensorType
        left_type = SensorType.HEAD_LEFT_CAMERA
        right_type = SensorType.HEAD_RIGHT_CAMERA
        obs = self._synced_pair([left_type, right_type])
        rgb_map = _msg_field(obs, "rgb_data_map")
        return {"left": rgb_map[left_type], "right": rgb_map[right_type]}

    def close(self) -> None:
        """SDK 세션 정리. 실제 종료 메서드명 미검증 — 존재하는 후보만 best-effort 호출."""
        for name in ("close", "shutdown", "disconnect", "release"):
            fn = getattr(self._robot, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return
