"""깊이 정제기(DepthRefiner) 공통 인터페이스 + 이름 기반 레지스트리.

모든 정제 방법(고전적 인페인팅, 단안 스케일 정렬, 파운데이션 모델 어댑터 등)은
이 모듈의 ``DepthRefiner``를 상속하고 ``@register``로 등록한다. 호출부는 구체
클래스를 몰라도 ``get_refiner(name)``/``available_refiners()``만으로 조립 가능.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import numpy as np

from ..common.camera import CameraIntrinsics


class DepthRefiner(ABC):
    """깊이 정제기 공통 인터페이스.

    구현체는 클래스 속성 ``name``(레지스트리 키)을 지정하고 ``refine``을 구현한다.
    """

    name: str = ""

    @abstractmethod
    def refine(self, rgb: np.ndarray, depth_m: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
        """rgb(H,W,3 uint8)와 depth_m(H,W float32 meters, 무효=0)을 받아 정제된
        depth_m(H,W float32 meters, 무효=0)을 반환한다."""
        raise NotImplementedError

    @classmethod
    def is_available(cls) -> bool:
        """무거운 의존성(모델 등)이 설치/로드 가능한지. 기본은 항상 사용 가능(True).

        의존성이 없는 환경에서도 예외를 던지지 말고 False를 반환해야 한다.
        """
        return True


REGISTRY: Dict[str, Type[DepthRefiner]] = {}


def register(cls: Type[DepthRefiner]) -> Type[DepthRefiner]:
    """``DepthRefiner`` 서브클래스를 ``cls.name`` 키로 REGISTRY에 등록하는 클래스 데코레이터."""
    REGISTRY[cls.name] = cls
    return cls


def get_refiner(name: str) -> DepthRefiner:
    """등록된 이름으로 정제기를 인스턴스화한다. 미등록 이름이면 KeyError."""
    try:
        cls = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "(none registered)"
        raise KeyError(
            "Unknown refiner {!r}. Available refiners: {}".format(name, known)
        )
    return cls()


def available_refiners() -> List[str]:
    """``is_available()``이 True인 등록된 정제기 이름 목록."""
    return [name for name, cls in REGISTRY.items() if cls.is_available()]
