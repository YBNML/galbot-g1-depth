"""스테레오 매처(StereoMatcher) 공통 인터페이스 + 이름 기반 레지스트리.

``depth_refine.refiners.base``의 ``DepthRefiner``/``REGISTRY`` 패턴을 스테레오
매칭에도 그대로 적용한다 — 구체 매칭 알고리즘(OpenCV SGBM, 학습 기반 스테레오 등)은
이 모듈의 ``StereoMatcher``를 상속하고 ``@register_matcher``로 등록한다. 호출부는
구체 클래스를 몰라도 ``get_matcher(name)``/``available_matchers()``만으로 조립 가능.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import numpy as np


class StereoMatcher(ABC):
    """스테레오 매처 공통 인터페이스.

    구현체는 클래스 속성 ``name``(레지스트리 키)을 지정하고 ``compute``를 구현한다.
    """

    name: str = ""

    @abstractmethod
    def compute(self, rect_left_bgr: np.ndarray, rect_right_bgr: np.ndarray) -> np.ndarray:
        """렉티파이된 좌/우 BGR 영상(H,W,3 uint8)을 받아 disparity_px(H,W float32)를
        반환한다. 매칭 실패/무효 픽셀은 ``<=0``."""
        raise NotImplementedError

    @classmethod
    def is_available(cls) -> bool:
        """무거운 의존성(모델 등)이 설치/로드 가능한지. 기본은 항상 사용 가능(True).

        의존성이 없는 환경에서도 예외를 던지지 말고 False를 반환해야 한다.
        """
        return True


MATCHER_REGISTRY: Dict[str, Type[StereoMatcher]] = {}


def register_matcher(cls: Type[StereoMatcher]) -> Type[StereoMatcher]:
    """``StereoMatcher`` 서브클래스를 ``cls.name`` 키로 MATCHER_REGISTRY에 등록하는 클래스 데코레이터.

    ``name``이 비어있으면(서브클래스가 지정을 잊은 경우) 즉시 ``ValueError``.
    이미 다른 클래스가 같은 이름으로 등록되어 있으면 조용히 덮어쓰지 않고
    ``ValueError`` — 이름 충돌은 한쪽 매처를 소리없이 접근 불가능하게 만들기
    때문. 동일 클래스 객체를 같은 이름으로 다시 등록하는 것(모듈 재로드 등)은
    멱등 — 예외 없이 통과.
    """
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(
            "Cannot register {}: missing or empty 'name' attribute".format(cls.__qualname__)
        )
    existing = MATCHER_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            "Matcher name {!r} is already registered to {}; cannot register {} under "
            "the same name (choose a distinct 'name' or fix the collision)".format(
                name, existing.__qualname__, cls.__qualname__
            )
        )
    MATCHER_REGISTRY[name] = cls
    return cls


def get_matcher(name: str) -> StereoMatcher:
    """등록된 이름으로 매처를 인스턴스화한다. 미등록 이름이면 KeyError."""
    try:
        cls = MATCHER_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(MATCHER_REGISTRY)) or "(none registered)"
        raise KeyError(
            "Unknown matcher {!r}. Available matchers: {}".format(name, known)
        )
    return cls()


def available_matchers() -> List[str]:
    """``is_available()``이 True인 등록된 매처 이름 목록."""
    return [name for name, cls in MATCHER_REGISTRY.items() if cls.is_available()]
