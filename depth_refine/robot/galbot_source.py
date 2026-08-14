"""Galbot SDK 1.9.x 어댑터 — 손목(D405)·헤드 스테레오 프레임을 FrameSource 계약으로 노출.

SDK는 로봇에서만 설치되어 있다. 이 모듈은 SDK를 지연 임포트
(`importlib.import_module(os.environ.get("GALBOT_SDK_MODULE", "galbot_sdk"))`)하므로
SDK가 없는 개발 PC에서도 이 파일을 import하는 것 자체는 항상 안전하다 — 실패는
`GalbotSource()`를 실제로 생성할 때만 명확한 RuntimeError로 드러난다(§7 우아한 비활성화).

**2026-08-14 실물 로봇(G1 "galbot-echo", SDK 1.9.1)에서 검증된 사실** (reports/deep_eval/
ANALYSIS.md — 이전 버전의 문서 기반 가정에서 어긋났던 지점을 전부 실측값으로 교체):

- 로봇 획득: ``GalbotRobot.get_instance(MachineType.G1)`` 후 ``robot.init(sensor_set)``
  (bool 반환). **init에 지정한 센서만 켜지므로** 손목 RGB/깊이 + 헤드 좌우를 명시한다.
  init 직후 첫 프레임이 도착할 때까지 짧은 워밍업이 필요하다(빈 dict 반환).
- 모든 get_*_data는 실패/미도착 시 **빈 dict**를 돌려준다(예외 아님) — 각 호출부에서
  빈 메시지를 검사해 명확한 RuntimeError로 바꾼다.
- ``header``는 ``{"frame_id", "timestamp_ns"}`` dict (속성형 객체 대비 양쪽 지원).
- RGB ``data``는 압축 스트림(손목은 format="rgb8"로 표기되지만 실제 46KB 압축) —
  cv2.imdecode로 디코드한다.
- 손목 깊이는 **raw uint16**(1280x720, format="16UC1", 압축 아님)이고 메시지에
  ``depth_scale``(실측 10000 = 0.1mm 단위)·``width``·``height``가 들어 있다.
  **depth_m = raw / depth_scale** — 메시지 필드를 우선 사용하고, 없을 때만 생성자
  인자(depth_scale)로 폴백한다.
- intrinsics는 fx/fy/cx/cy 필드가 아니라 **K 행렬(9-float)** + width/height로 온다.
- ``get_synced_observation``은 현재 None을 반환한다(원인 미상) — 헤드 좌우는
  ``get_rgb_data`` 연속 호출로 충분하다: 헤드는 하드웨어 동기 페어라 좌우 타임스탬프가
  동일하고(실측 스큐 0.0ms), 불일치 시 1회 재시도한다.
- 종료 시퀀스: ``request_shutdown() -> wait_for_shutdown() -> destroy()``.
  destroy() 이후 같은 프로세스에서 SDK 재초기화는 불가능하다.
"""
from __future__ import annotations

import importlib
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..common.camera import CameraIntrinsics
from .interface import FrameSource, HeadPair, WristFrame

_SIDES = ("left", "right")
_DEFAULT_DEPTH_SCALE = 10000.0  # raw -> m 나눗셈 값. 메시지의 depth_scale 필드가 우선이며
                                # (실측 10000 = 0.1mm 단위), 이 값은 필드가 없을 때의 폴백.
_WARMUP_TIMEOUT_S = 10.0        # init 후 첫 프레임 대기 상한
_WARMUP_POLL_S = 0.3


def _msg_field(msg: Any, key: str, default: Any = None) -> Any:
    """메시지에서 key 필드를 읽는다 — dict형(구독 접근)과 속성 접근 메시지를 모두 지원."""
    try:
        return msg[key]
    except (TypeError, KeyError, IndexError):
        return getattr(msg, key, default)


def _msg_empty(msg: Any) -> bool:
    """SDK get_*_data의 '실패 = 빈 dict' 관례 감지 (실측 확인)."""
    if msg is None:
        return True
    try:
        return len(msg) == 0
    except TypeError:
        return False


def _header_ts_ns(msg: Any) -> int:
    """메시지 header(dict 또는 객체)에서 timestamp_ns를 꺼낸다 (실측: dict)."""
    header = _msg_field(msg, "header")
    if header is None:
        raise RuntimeError(
            "메시지에 header 필드가 없습니다 — record.py --dry-run으로 실제 메시지 구조를 "
            "확인하세요."
        )
    ts = _msg_field(header, "timestamp_ns")
    if ts is None:
        raise RuntimeError(
            "header에 timestamp_ns가 없습니다 (header={!r})".format(header)
        )
    return int(ts)


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
        depth_scale: 손목 깊이 raw -> 미터 환산 나눗셈 폴백 값. **메시지에 depth_scale
            필드가 있으면 그 값이 우선한다**(실측 10000 = 0.1mm 단위). record.py
            --depth-scale로 조정 가능하지만 실 SDK에서는 사실상 쓰이지 않는다.
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

    # ---------------- SDK 접근 지점 ----------------
    def _acquire_robot(self) -> Any:
        """실측 획득 패턴: GalbotRobot.get_instance(MachineType.G1) -> init(sensor_set).

        init에 지정한 센서만 활성화되므로 이 소스가 쓰는 4개(손목 RGB/깊이 + 헤드 좌우)를
        명시한다. init은 bool을 반환하며(예외 아님), 성공 후에도 첫 프레임 도착까지
        수백 ms~수 초가 걸리므로 손목 RGB가 나올 때까지 폴링으로 워밍업한다.
        """
        try:
            robot = self._sdk.GalbotRobot.get_instance(self._sdk.MachineType.G1)
            rgb_type, depth_type = self._wrist_sensor_types_from(self._sdk, self.side)
            SensorType = self._sdk.SensorType
            sensors = {
                rgb_type,
                depth_type,
                SensorType.HEAD_LEFT_CAMERA,
                SensorType.HEAD_RIGHT_CAMERA,
            }
            if not robot.init(sensors):
                raise RuntimeError("robot.init()이 False를 반환 — 센서 연결 상태를 확인하세요")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Galbot 로봇 인스턴스 초기화에 실패했습니다 (GalbotRobot.get_instance(G1)/"
                "init(sensors) 절차를 로봇에서 확인하세요): {}".format(exc)
            ) from exc

        # 워밍업: 첫 손목 RGB가 도착할 때까지 대기 (빈 dict = 아직 미도착, 실측 관례)
        deadline = time.monotonic() + _WARMUP_TIMEOUT_S
        rgb_type, _ = self._wrist_sensor_types_from(self._sdk, self.side)
        while _msg_empty(robot.get_rgb_data(rgb_type)):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "센서 워밍업 실패 — init 후 {:.0f}초 동안 {} RGB 프레임이 도착하지 "
                    "않았습니다. 카메라 연결/캡처 데몬 상태를 확인하세요.".format(
                        _WARMUP_TIMEOUT_S, self.side
                    )
                )
            time.sleep(_WARMUP_POLL_S)
        return robot

    @staticmethod
    def _wrist_sensor_types_from(sdk: Any, side: str) -> Tuple[Any, Any]:
        SensorType = sdk.SensorType
        if side == "left":
            return SensorType.LEFT_ARM_CAMERA, SensorType.LEFT_ARM_DEPTH_CAMERA
        return SensorType.RIGHT_ARM_CAMERA, SensorType.RIGHT_ARM_DEPTH_CAMERA

    def _wrist_sensor_types(self) -> Tuple[Any, Any]:
        return self._wrist_sensor_types_from(self._sdk, self.side)

    def _sdk_rgb(self, sensor_type: Any) -> Any:
        """robot.get_rgb_data() 호출 단일 지점 — 빈 dict(실패/미도착)를 즉시 진단한다."""
        msg = self._robot.get_rgb_data(sensor_type)
        if _msg_empty(msg):
            raise RuntimeError(
                "get_rgb_data({})가 빈 메시지를 반환 — 해당 카메라가 robot.init() 센서 "
                "집합에 포함됐는지, 캡처 데몬이 살아있는지 확인하세요.".format(sensor_type)
            )
        return msg

    def _sdk_depth(self, sensor_type: Any) -> Any:
        """robot.get_depth_data() 호출 단일 지점."""
        msg = self._robot.get_depth_data(sensor_type)
        if _msg_empty(msg):
            raise RuntimeError(
                "get_depth_data({})가 빈 메시지를 반환 — 해당 깊이 카메라가 robot.init() "
                "센서 집합에 포함됐는지 확인하세요.".format(sensor_type)
            )
        return msg

    def _sdk_intrinsic(self, sensor_type: Any) -> Any:
        """robot.get_camera_intrinsic() 호출 단일 지점 (손목·헤드 공용)."""
        msg = self._robot.get_camera_intrinsic(sensor_type)
        if _msg_empty(msg):
            raise RuntimeError(
                "get_camera_intrinsic({})가 빈 dict를 반환했습니다.".format(sensor_type)
            )
        return msg

    def _sdk_extrinsic(self, sensor_type: Any) -> Any:
        """robot.get_sensor_extrinsic() 호출 단일 지점 -> ([x,y,z,qx,qy,qz,qw], ts_ns).
        실패 시 빈 리스트를 반환한다(SDK 문서) — 참고값이므로 예외 대신 그대로 전달."""
        return self._robot.get_sensor_extrinsic(sensor_type)

    def _decode_rgb(self, msg: Any) -> np.ndarray:
        """압축 RGB 메시지(msg['data']) -> BGR uint8 이미지 (cv2 관례, 저장소 전체와 일관)."""
        data = _msg_field(msg, "data")
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(
                "RGB 메시지 디코딩 실패 — msg['data']가 유효한 압축 이미지가 아닙니다. "
                "record.py --dry-run으로 실제 메시지 구조를 확인하세요."
            )
        return img

    def _decode_depth(self, msg: Any) -> np.ndarray:
        """손목 깊이 메시지 -> float32 미터 깊이.

        실측 포맷: raw uint16(비압축), 메시지에 width/height/depth_scale 포함.
        depth_m = raw / depth_scale (depth_scale 실측 10000 = 0.1mm 단위).
        방어적으로 float32 raw·압축(imdecode) 페이로드도 지원한다.
        """
        data = _msg_field(msg, "data")
        width = int(_msg_field(msg, "width", 0) or 0)
        height = int(_msg_field(msg, "height", 0) or 0)
        scale = float(_msg_field(msg, "depth_scale", 0) or 0) or self._depth_scale

        raw = None
        n = len(data)
        if width > 0 and height > 0:
            if n == width * height * 2:
                raw = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            elif n == width * height * 4:
                raw = np.frombuffer(data, dtype=np.float32).reshape(height, width)
        if raw is None:
            arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if arr is None or arr.ndim != 2:
                raise RuntimeError(
                    "깊이 메시지 디코딩 실패 — {} bytes를 raw({}x{} uint16/float32)로도 "
                    "압축 이미지로도 해석할 수 없습니다. record.py --dry-run으로 실제 "
                    "메시지 구조를 확인하세요.".format(n, width, height)
                )
            raw = arr
        return raw.astype(np.float32) / scale

    def _to_intrinsics(self, raw: Any) -> CameraIntrinsics:
        """get_camera_intrinsic() 반환값 -> CameraIntrinsics.

        실측 포맷: K 행렬(9-float 리스트) + width/height. fx=K[0], cx=K[2], fy=K[4],
        cy=K[5]. 방어적으로 구식 fx/fy/cx/cy 필드도 지원한다.
        """
        K = _msg_field(raw, "K")
        if K is not None and len(K) == 9:
            fx, cx, fy, cy = float(K[0]), float(K[2]), float(K[4]), float(K[5])
        else:
            fx = float(_msg_field(raw, "fx"))
            fy = float(_msg_field(raw, "fy"))
            cx = float(_msg_field(raw, "cx"))
            cy = float(_msg_field(raw, "cy"))
        width = int(_msg_field(raw, "width"))
        height = int(_msg_field(raw, "height"))
        return CameraIntrinsics(fx, fy, cx, cy, width, height)

    def _head_pair_msgs(self) -> Tuple[Any, Any]:
        """헤드 좌우 RGB 메시지 획득.

        get_synced_observation은 실측에서 None만 반환해(원인 미상) 사용하지 않는다.
        대신 get_rgb_data를 좌우 연속 호출한다 — 헤드는 하드웨어 동기 페어라 좌우
        타임스탬프가 동일하고(실측 스큐 0.0ms), 호출 사이에 프레임 경계를 넘은 경우
        (타임스탬프 불일치) 1회 재획득한다.
        """
        SensorType = self._sdk.SensorType
        msg_l = self._sdk_rgb(SensorType.HEAD_LEFT_CAMERA)
        msg_r = self._sdk_rgb(SensorType.HEAD_RIGHT_CAMERA)
        if _header_ts_ns(msg_l) != _header_ts_ns(msg_r):
            msg_l = self._sdk_rgb(SensorType.HEAD_LEFT_CAMERA)
            msg_r = self._sdk_rgb(SensorType.HEAD_RIGHT_CAMERA)
        return msg_l, msg_r

    # ---------------- FrameSource 계약 ----------------
    def get_wrist_frame(self) -> WristFrame:
        rgb_type, depth_type = self._wrist_sensor_types()
        msg_rgb = self._sdk_rgb(rgb_type)
        msg_depth = self._sdk_depth(depth_type)
        intr = self._to_intrinsics(self._sdk_intrinsic(rgb_type))

        rgb = self._decode_rgb(msg_rgb)
        depth = self._decode_depth(msg_depth)
        ts_rgb = _header_ts_ns(msg_rgb)
        ts_depth = _header_ts_ns(msg_depth)

        return WristFrame(rgb=rgb, depth_m=depth, intrinsics=intr,
                           ts_rgb_ns=ts_rgb, ts_depth_ns=ts_depth, gt_depth_m=None)

    def get_head_pair(self) -> HeadPair:
        msg_l, msg_r = self._head_pair_msgs()
        left = self._decode_rgb(msg_l)
        right = self._decode_rgb(msg_r)
        return HeadPair(left=left, right=right,
                         ts_left_ns=_header_ts_ns(msg_l), ts_right_ns=_header_ts_ns(msg_r),
                         gt_depth_left_m=None)

    def head_intrinsics(self) -> Tuple[CameraIntrinsics, CameraIntrinsics]:
        SensorType = self._sdk.SensorType
        intr_l = self._to_intrinsics(self._sdk_intrinsic(SensorType.HEAD_LEFT_CAMERA))
        intr_r = self._to_intrinsics(self._sdk_intrinsic(SensorType.HEAD_RIGHT_CAMERA))
        return intr_l, intr_r

    def get_head_extrinsics_sdk(self) -> Dict:
        """head/extrinsics_sdk.json — record.py가 그대로 저장한다.

        실측(2026-08-14, reports/deep_eval/extrinsic_probe.json): SDK TF가 헤드 좌우
        포즈(base_link 기준)와 **좌<-우 상대 변환까지 전부 제공**한다. 상대 변환은
        [0.0596635, 0, 0] + 단위 쿼터니언 — 공장 렉티파이 페어, baseline 59.6635mm.
        즉 이 로봇에서는 체커보드 캘리브레이션(calibrate_head.py) 없이도 스테레오 기하가
        완결된다. init 직후에는 TF 버퍼가 비어 빈 리스트가 올 수 있어 짧게 재시도한다.
        """
        SensorType = self._sdk.SensorType
        out = {}
        for name, st in (("left", SensorType.HEAD_LEFT_CAMERA),
                         ("right", SensorType.HEAD_RIGHT_CAMERA)):
            try:
                pose, ts = self._sdk_extrinsic(st)
                for _ in range(6):  # TF 워밍업 재시도 (~3초)
                    if len(pose):
                        break
                    time.sleep(0.5)
                    pose, ts = self._sdk_extrinsic(st)
            except Exception as exc:
                out[name] = {"error": str(exc)}
                continue
            out[name] = {"pose_vec": _pose_vec_to_list(pose), "ts_ns": int(ts)}

        # 좌<-우 상대 변환(baseline 포함, [x,y,z,qx,qy,qz,qw]). 프레임 이름은 intrinsic
        # header의 frame_id 실측값이다.
        try:
            mat, ts = self._robot.get_transform(
                "head_left_camera_color_optical_frame",
                "head_right_camera_color_optical_frame", 0, 500)
            out["left_from_right"] = {"pose_vec": _pose_vec_to_list(mat), "ts_ns": int(ts)}
        except Exception as exc:
            out["left_from_right"] = {"error": str(exc)}
        return out

    # ---------------- dry-run 진단 전용 (디코드 전 raw 메시지 노출) ----------------
    def get_wrist_raw(self) -> Dict[str, Any]:
        """record.py --dry-run 전용: 디코드하지 않은 원본 SDK 메시지를 반환한다."""
        rgb_type, depth_type = self._wrist_sensor_types()
        return {
            "rgb": self._sdk_rgb(rgb_type),
            "depth": self._sdk_depth(depth_type),
        }

    def get_head_raw(self) -> Dict[str, Any]:
        """record.py --dry-run 전용: 디코드하지 않은 원본 SDK 메시지를 반환한다."""
        msg_l, msg_r = self._head_pair_msgs()
        return {"left": msg_l, "right": msg_r}

    def close(self) -> None:
        """실측 종료 시퀀스: request_shutdown -> wait_for_shutdown -> destroy (best-effort).

        destroy() 이후 같은 프로세스에서 SDK 재초기화는 불가능하다(SDK 문서) — record.py
        처럼 프로세스 종료 직전에만 부른다.
        """
        for name in ("request_shutdown", "wait_for_shutdown", "destroy"):
            fn = getattr(self._robot, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
