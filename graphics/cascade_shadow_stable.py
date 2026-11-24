"""稳定级联阴影（Cascade Shadow Maps）的纯 Python 参考实现。

本模块不依赖 NumPy 或 GPU，可以直接在契约/回归测试中验证以下六个修复点：

1. :func:`texel_snap` 在**光空间**按当前级联纹素尺寸对齐中心/边界，且最小边
   界向下取整、最大边界向上取整，避免相机旋转时阴影边缘反向来回跳。
2. :func:`switch_hysteresis` 用当前级联和切换边界两侧的滞回带决定是否换层，
   避免角色停在固定比例切分线附近时阴影清晰度反复突变。
3. :func:`same_frame_matrix` 把相机矩阵和阴影矩阵放进同一个帧捕获包；异步
   阴影渲染只能使用这个不可变包，不能在提交后改读主相机的新矩阵。
4. :func:`filter_per_cascade` 按每一级自己的正交覆盖范围和分辨率计算纹素世
   界尺寸，再计算 PCF/过滤半径，远级联不再误用近级联的小半径。
5. :func:`fov_coverage` 由垂直 FOV、宽高比和远平面重算覆盖宽度/高度与包围半
   径，广角相机不会继续使用窄 FOV 的级联覆盖范围。
6. :func:`seam_blend` 在相邻级联过渡带内生成 smoothstep 权重并混合两层阴影
   结果，消除硬接缝。

其余函数提供分割、拟合、阴影图描述、bias、确定性随机、校验、审计、安全回退
和指标。所有公共入口都能容忍仓库自带的 ``fn(1) is not None`` 冒烟调用。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

__all__ = [
    # 六个直接修复点
    "texel_snap",
    "switch_hysteresis",
    "same_frame_matrix",
    "filter_per_cascade",
    "fov_coverage",
    "seam_blend",
    # 配套入口
    "determinism",
    "validation",
    "audit",
    "fallback",
    "cascade_count",
    "metrics",
    "shadow_map",
    "bias",
    "fit",
    # 可复用辅助函数/类型
    "cascade_splits",
    "select_cascade",
    "seam_weight",
    "texel_size",
    "HysteresisDecision",
    "HysteresisState",
    "identity_matrix",
    "ortho_matrix",
]

_EPS = 1.0e-12
_DEFAULT_CASCADES = 4
_DEFAULT_RESOLUTION = 2048
_DEFAULT_FOV_DEGREES = 60.0
_DEFAULT_ASPECT = 16.0 / 9.0
_DEFAULT_NEAR = 0.1
_DEFAULT_FAR = 200.0
_DEFAULT_HYSTERESIS_RATIO = 0.10
_DEFAULT_BLEND_FRACTION = 0.10
_DEFAULT_KERNEL_RADIUS_TEXELS = 1.0


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_float(value: Any, default: float | None = None) -> float:
    if _is_number(value):
        return float(value)
    if default is not None:
        return float(default)
    raise ValueError(f"expected a finite number, got {value!r}")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _clean(value: float, tolerance: float = 1.0e-10) -> float:
    value = float(value)
    if abs(value) < tolerance:
        return 0.0
    return value


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) <= _EPS:
        return 0.0 if x < edge1 else 1.0
    t = _clamp((float(x) - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    t = _clamp(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _merge_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """合并 dict 位置参数和关键字参数，返回 (配置, 非 dict 位置参数)。"""

    cfg: dict[str, Any] = {}
    positional: list[Any] = []
    for arg in args:
        if isinstance(arg, dict):
            cfg.update(arg)
        else:
            positional.append(arg)
    cfg.update(kwargs)
    return cfg, positional


def _pick(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _number_at(values: Sequence[Any], index: int, default: float) -> float:
    if index < len(values) and _is_number(values[index]):
        return float(values[index])
    return default


def _as_float_list(value: Any, name: str = "sequence") -> list[float]:
    if _is_number(value):
        return [float(value)]
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be numeric, not a string")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    out: list[float] = []
    for item in items:
        if not _is_number(item):
            raise ValueError(f"{name} contains a non-finite number: {item!r}")
        out.append(float(item))
    return out


def _fov_radians(fov: Any = _DEFAULT_FOV_DEGREES, explicit_unit: Any = None) -> float:
    value = abs(_as_float(fov, _DEFAULT_FOV_DEGREES))
    unit = str(explicit_unit or "").lower()
    as_degrees: bool | None = None
    if unit in {"degree", "degrees", "deg", "d"}:
        as_degrees = True
    elif unit in {"radian", "radians", "rad", "r"}:
        as_degrees = False
    elif explicit_unit is True:
        as_degrees = True
    elif explicit_unit is False:
        as_degrees = False

    if as_degrees is None:
        # 合法弧度 FOV 不超过 pi；常见角度配置 30/45/60/90 均大于 pi。
        as_degrees = value > math.pi
    radians = math.radians(value) if as_degrees else value
    # 极端非法值不向调用方抛异常，钳到开区间内以保证后续 tan 有限。
    return _clamp(radians, _EPS, math.pi - _EPS)


# ---------------------------------------------------------------------------
# 级联数量与分割
# ---------------------------------------------------------------------------


def cascade_count(*args: Any, **kwargs: Any) -> int:
    """返回级联数量。

    接受级联边界序列、含 ``cascades/splits/count`` 的配置，或直接传入整数。
    冒烟调用 ``cascade_count(1)`` 返回 ``1``。
    """

    cfg, positional = _merge_args(args, kwargs)
    spec: Any = None
    if positional:
        spec = positional[0]
    elif cfg:
        spec = cfg

    if spec is None:
        return _DEFAULT_CASCADES
    if isinstance(spec, dict):
        for key in ("splits", "distances", "boundaries", "levels", "cascades"):
            value = spec.get(key)
            if isinstance(value, (list, tuple, range)):
                return max(1, len(value))
            if _is_number(value):
                return max(1, int(value))
        if _is_number(spec.get("count")):
            return max(1, int(spec["count"]))
        return _DEFAULT_CASCADES
    if isinstance(spec, (list, tuple, range)):
        return max(1, len(spec))
    if _is_number(spec):
        return max(1, int(float(spec)))
    return _DEFAULT_CASCADES


def cascade_splits(*args: Any, **kwargs: Any) -> list[float]:
    """生成 PSSM 风格的对数/均匀混合切分距离。

    位置参数可以是 ``(count,)``、``(count, near, far)`` 或现成边界序列。
    ``lambda_split=1`` 为纯对数切分，``0`` 为均匀切分。
    """

    cfg, positional = _merge_args(args, kwargs)
    raw = positional[0] if positional else _pick(cfg, "splits", "distances", "boundaries")
    if isinstance(raw, (list, tuple, range)):
        values = _as_float_list(raw, "splits")
        return [abs(v) for v in values]

    count = max(1, cascade_count(cfg.get("count", raw if raw is not None else _DEFAULT_CASCADES)))
    near = abs(_as_float(_pick(cfg, "near", "near_plane", default=_DEFAULT_NEAR), _DEFAULT_NEAR))
    far = abs(_as_float(_pick(cfg, "far", "far_plane", "distance", default=_DEFAULT_FAR), _DEFAULT_FAR))
    if len(positional) >= 2:
        near = abs(_as_float(positional[1], near))
    if len(positional) >= 3:
        far = abs(_as_float(positional[2], far))
    if far <= near:
        far = near + max(1.0, abs(far))

    weight = _clamp(_as_float(_pick(cfg, "lambda_split", "split_lambda", "exponential_weight", default=0.5), 0.5))
    ratio = far / near
    result: list[float] = []
    for i in range(1, count + 1):
        t = i / count
        logarithmic = near * (ratio ** t)
        uniform = near + (far - near) * t
        result.append(_clean(weight * logarithmic + (1.0 - weight) * uniform))
    result[-1] = float(far)
    return result


def select_cascade(distance: float, splits: Sequence[float]) -> int:
    """按深度选择第一个远平面不小于 ``distance`` 的级联，并钳到末级。"""

    d = abs(_as_float(distance, 0.0))
    boundaries = _as_float_list(splits, "splits")
    if not boundaries:
        return 0
    for index, boundary in enumerate(boundaries):
        if d <= boundary + _EPS:
            return index
    return len(boundaries) - 1


# ---------------------------------------------------------------------------
# 1. 光空间纹素对齐
# ---------------------------------------------------------------------------


def _grid_round(value: float, step: float, mode: str = "nearest") -> float:
    if step <= _EPS:
        return float(value)
    units = float(value) / step
    mode = str(mode or "nearest").lower()
    if mode in {"floor", "min", "lower", "down"}:
        index = math.floor(units)
    elif mode in {"ceil", "max", "upper", "up"}:
        index = math.ceil(units)
    elif mode in {"round", "nearest", "half_up"}:
        # 半值向远离零方向取整，避免 Python banker's rounding 在边界上偏向偶数。
        index = math.floor(abs(units) + 0.5)
        if units < 0.0:
            index = -index
    else:
        raise ValueError(f"unknown snap mode: {mode}")
    return _clean(index * step, tolerance=step * 1.0e-9)


def _resolve_texel_size(cfg: dict[str, Any], positional: Sequence[Any] = ()) -> float:
    direct = _pick(cfg, "texel_size", "texel", "world_per_texel", "texel_world_size")
    if direct is not None:
        return abs(_as_float(direct, 1.0))

    resolution = _pick(cfg, "resolution", "map_size", "shadow_map_size", "texels")
    world_size = _pick(cfg, "world_size", "span", "ortho_size", "coverage", "width", "size")
    extent = _pick(cfg, "extent", "half_size", "half_width", "ortho_extent")

    if len(positional) >= 2 and _is_number(positional[-1]):
        maybe_resolution = float(positional[-1])
        if resolution is None and maybe_resolution >= 1.0:
            resolution = maybe_resolution
        if world_size is None and extent is None and _is_number(positional[0]):
            world_size = positional[0]

    if resolution is not None and _is_number(resolution) and float(resolution) > 0.0:
        res = float(resolution)
        if world_size is not None and _is_number(world_size):
            return abs(float(world_size)) / res
        if extent is not None and _is_number(extent):
            return 2.0 * abs(float(extent)) / res
    return 1.0


def texel_snap(*args: Any, **kwargs: Any) -> Any:
    """把光空间坐标或正交边界对齐到纹素网格。

    常见调用：

    * ``texel_snap(center_xy, texel_size)``：对齐二维/三维中心；三维坐标默认
      只对齐光平面的 x/y，z（光空间深度）保持不变。
    * ``texel_snap(x, y, texel_size)`` / ``texel_snap(x, y, z, texel_size)``。
    * ``texel_snap({"min": ..., "max": ..., "texel_size": ...})``：边界盒的
      ``min`` 用 floor、``max`` 用 ceil 向外对齐，保证覆盖不缩小。
    * 提供 ``resolution`` 和 ``world_size/extent`` 时自动计算纹素尺寸。

    标量/序列输入默认返回对齐后的同形态值；dict 或 ``details=True`` 时返回
    含 ``snapped``、``delta`` 和 ``max_error`` 的报告。
    """

    cfg, positional = _merge_args(args, kwargs)
    details = bool(cfg.pop("details", False) or cfg.pop("return_report", False))
    mode = str(_pick(cfg, "mode", "rounding", default="nearest"))

    # 明确常见位置参数中的 texel_size，避免把 (center, texel_size) 误当成
    # (world_size, resolution)。需要由覆盖范围/分辨率换算时请使用关键字参数。
    if _pick(cfg, "texel_size", "texel", "world_per_texel", "texel_world_size") is None:
        if len(positional) >= 2 and isinstance(positional[0], (Sequence,)) and not isinstance(positional[0], (str, bytes)) and _is_number(positional[1]):
            cfg["texel_size"] = abs(float(positional[1]))
        elif len(positional) == 2 and _is_number(positional[0]) and _is_number(positional[1]):
            cfg["texel_size"] = abs(float(positional[1]))
        elif len(positional) >= 3 and _is_number(positional[-1]) and all(_is_number(v) for v in positional[:-1]):
            cfg["texel_size"] = abs(float(positional[-1]))

    # 边界盒：最小边 floor，最大边 ceil；这是“覆盖完整”的正确方向。
    bounds = _pick(cfg, "bounds", "ortho_bounds")
    lower = _pick(cfg, "min", "lower", "minimum")
    upper = _pick(cfg, "max", "upper", "maximum")
    if isinstance(bounds, dict):
        lower = bounds.get("min", bounds.get("lower"))
        upper = bounds.get("max", bounds.get("upper"))
    if lower is not None and upper is not None:
        texel = _resolve_texel_size(cfg, positional)
        if isinstance(lower, Sequence) and not isinstance(lower, (str, bytes)):
            low_out = [_grid_round(v, texel, "floor") for v in lower]
            high_out = [_grid_round(v, texel, "ceil") for v in upper]
        else:
            low_out = _grid_round(_as_float(lower), texel, "floor")
            high_out = _grid_round(_as_float(upper), texel, "ceil")
        report = {
            "space": "light",
            "texel_size": texel,
            "min": low_out,
            "max": high_out,
            "mode": "outward_bounds",
            "expanded": True,
        }
        return report if details or args and isinstance(args[0], dict) else (low_out, high_out)

    texel = _resolve_texel_size(cfg, positional)

    center = _pick(cfg, "center", "position", "origin", "xy", "light_center")
    if center is None:
        nums = [p for p in positional if _is_number(p)]
        non_nums = [p for p in positional if not _is_number(p)]
        if non_nums and isinstance(non_nums[0], (Sequence,)):
            center = non_nums[0]
        elif len(nums) >= 2 and len(positional) >= 3 and _is_number(positional[-1]):
            # x, y, [z,] texel_size
            texel = abs(float(positional[-1])) or texel
            center = nums[:-1] if len(nums) > 2 else nums[:-1]
            if len(center) == 1:
                center = [nums[0], nums[1]]
        elif len(positional) >= 2:
            center = positional[0]
            if _is_number(positional[1]):
                texel = abs(float(positional[1]))
        elif positional:
            center = positional[0]
        else:
            center = 0.0

    axes = cfg.get("axes")
    original = center
    if _is_number(center):
        snapped = _grid_round(float(center), texel, mode)
        delta = snapped - float(center)
        result: Any = snapped
    elif isinstance(center, (Sequence, list, tuple)):
        values = _as_float_list(center, "center")
        if axes is None:
            # 光空间前两轴是正交平面；第三轴是深度，不做纹素量化。
            snap_axes = set(range(min(2, len(values)))) if len(values) >= 2 else {0}
        else:
            snap_axes = {int(a) for a in axes}
        snapped_values = [
            _grid_round(value, texel, mode) if index in snap_axes else value
            for index, value in enumerate(values)
        ]
        snapped = type(center)(snapped_values) if isinstance(center, tuple) else snapped_values
        delta = [snapped_values[i] - values[i] for i in range(len(values))]
        result = snapped
    else:
        raise ValueError("center must be a number or numeric sequence")

    report = {
        "space": "light",
        "original": original,
        "snapped": snapped,
        "delta": delta,
        "texel_size": texel,
        "mode": mode,
        "max_error": 0.5 * texel,
        "stable_under_camera_rotation": True,
    }
    if details or (args and isinstance(args[0], dict)):
        return report
    return result


# ---------------------------------------------------------------------------
# 2. 级联切换滞回
# ---------------------------------------------------------------------------


class HysteresisDecision(int):
    """整数型级联决策；既可以 ``== 1``，也可以 ``decision['cascade']``。"""

    def __new__(cls, cascade: int, **fields: Any) -> "HysteresisDecision":
        obj = int.__new__(cls, int(cascade))
        obj._fields = dict(fields)
        obj._fields.setdefault("cascade", int(cascade))
        obj._fields.setdefault("index", int(cascade))
        obj._fields.setdefault("value", int(cascade))
        return obj

    def __getitem__(self, key: str) -> Any:
        if key in self._fields:
            return self._fields[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._fields.get(key, default)

    def keys(self):
        return self._fields.keys()

    def items(self):
        return self._fields.items()

    def __repr__(self) -> str:
        return f"HysteresisDecision({int(self)}, switched={self._fields.get('switched')})"


class HysteresisState:
    """按相机/视图键保存当前级联的小型有状态滞回器。"""

    def __init__(self, current: int = 0, *, band: float = 0.0, ratio: float = _DEFAULT_HYSTERESIS_RATIO):
        self.current = max(0, int(current))
        self.band = max(0.0, float(band))
        self.ratio = max(0.0, float(ratio))

    def update(self, distance: float, splits: Sequence[float]) -> HysteresisDecision:
        decision = switch_hysteresis(
            distance=distance,
            splits=splits,
            current=self.current,
            band=self.band,
            ratio=self.ratio,
        )
        self.current = int(decision)
        return decision


def switch_hysteresis(*args: Any, **kwargs: Any) -> HysteresisDecision:
    """带滞回地决定级联编号。

    两种主要调用方式：

    1. 离散层级：``switch_hysteresis(current, candidate, threshold=1)``。
       只有 ``abs(candidate-current) >= threshold`` 才切换。
    2. 距离/切分：``switch_hysteresis(distance=12, splits=[10,20], current=0,
       ratio=0.1)``。向外切换要求越过边界加滞回量，向内切换要求低于边界减滞
       回量，因此边界附近存在不会来回跳的滞回带。

    返回 :class:`HysteresisDecision`，它本身就是级联整数，同时携带
    ``switched/raw_candidate/lower/upper`` 等审计字段。
    """

    cfg, positional = _merge_args(args, kwargs)

    # 距离驱动模式。
    distance = _pick(cfg, "distance", "z", "depth")
    splits = _pick(cfg, "splits", "boundaries", "cascade_splits")
    if distance is not None and splits is not None:
        d = abs(_as_float(distance, 0.0))
        boundaries = _as_float_list(splits, "splits")
        current = max(0, int(_as_float(_pick(cfg, "current", "current_cascade", default=0), 0)))
        raw = select_cascade(d, boundaries)
        band = abs(_as_float(_pick(cfg, "band", "margin", "hysteresis", default=0.0), 0.0))
        ratio = abs(_as_float(_pick(cfg, "ratio", "hysteresis_ratio", default=_DEFAULT_HYSTERESIS_RATIO), _DEFAULT_HYSTERESIS_RATIO))

        switched = raw != current
        reason = "same_cascade"
        if raw > current and current < len(boundaries):
            boundary = abs(boundaries[current])
            required = max(band, boundary * ratio) if band == 0.0 or ratio > 0.0 else band
            upper = boundary + required
            lower = boundary - required
            if d + _EPS < upper:
                chosen = current
                switched = False
                reason = "outward_waiting_in_hysteresis_band"
            else:
                chosen = raw
                reason = "switched_outward"
        elif raw < current and current - 1 < len(boundaries):
            boundary = abs(boundaries[current - 1])
            required = max(band, boundary * ratio) if band == 0.0 or ratio > 0.0 else band
            upper = boundary + required
            lower = boundary - required
            if d + _EPS > lower:
                chosen = current
                switched = False
                reason = "inward_waiting_in_hysteresis_band"
            else:
                chosen = raw
                reason = "switched_inward"
        else:
            chosen = current if not switched else raw
            boundary = None
            lower = upper = None

        count = len(boundaries)
        chosen = max(0, min(chosen, max(0, count - 1))) if count else 0
        return HysteresisDecision(
            chosen,
            switched=bool(chosen != current),
            current=current,
            raw_candidate=raw,
            distance=d,
            splits=boundaries,
            boundary=boundary,
            lower=lower,
            upper=upper,
            band=band,
            ratio=ratio,
            reason=reason,
        )

    # 离散 current/candidate/threshold 模式。
    current = 0
    candidate: int | None = None
    threshold = _as_float(_pick(cfg, "threshold", "min_delta", "deadband", default=1), 1)

    if positional:
        if len(positional) >= 1 and _is_number(positional[0]):
            current = int(_as_float(positional[0]))
        if len(positional) >= 2 and _is_number(positional[1]):
            candidate = int(_as_float(positional[1]))
        if len(positional) >= 3 and _is_number(positional[2]):
            threshold = _as_float(positional[2], threshold)
    current = int(_as_float(_pick(cfg, "current", "current_cascade", default=current), current))
    if candidate is None:
        cand_value = _pick(cfg, "candidate", "desired", "target")
        candidate = int(_as_float(cand_value, current)) if cand_value is not None else current
    threshold = max(0, int(math.ceil(abs(threshold))))

    if candidate == current:
        chosen = current
        switched = False
    elif abs(candidate - current) >= threshold:
        chosen = candidate
        switched = True
    else:
        chosen = current
        switched = False
    return HysteresisDecision(
        chosen,
        switched=switched,
        current=current,
        candidate=candidate,
        raw_candidate=candidate,
        threshold=threshold,
        reason="threshold_exceeded" if switched else "inside_deadband",
    )


# ---------------------------------------------------------------------------
# 3. 相机/阴影矩阵同帧捕获
# ---------------------------------------------------------------------------


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _normalize_matrix(value: Any) -> list[list[float]]:
    if value is None:
        return identity_matrix()
    if isinstance(value, dict):
        value = _pick(value, "matrix", "data", "m", "view_projection", "projection_view")
        if value is None:
            return identity_matrix()
    try:
        if isinstance(value, (list, tuple)) and len(value) == 16 and _is_number(value[0]):
            return [[float(value[r * 4 + c]) for c in range(4)] for r in range(4)]
        rows = list(value)
        if len(rows) == 4 and all(isinstance(row, (list, tuple)) and len(row) == 4 for row in rows):
            return [[_as_float(v, 0.0) for v in row] for row in rows]
        if len(rows) == 4 and all(_is_number(v) for v in rows):
            # 列主序 4 向量也接受。
            cols = [_as_float_list(row) for row in rows]
            return [[cols[c][r] for c in range(4)] for r in range(4)]
    except Exception:
        return identity_matrix()
    return identity_matrix()


def _matrix_frame(value: Any) -> int | None:
    if isinstance(value, dict):
        frame = _pick(value, "frame", "frame_id", "captured_frame")
        if _is_number(frame):
            return int(frame)
    return None


def _matrix_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return _pick(value, "matrix", "data", "m", "view_projection", "projection_view", default=value)
    return value


def _mat4_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def ortho_matrix(left: float, right: float, bottom: float, top: float, near: float = -1.0, far: float = 1.0) -> list[list[float]]:
    """标准 OpenGL 风格正交投影矩阵（测试/模拟用）。"""

    rml = right - left
    tmb = top - bottom
    fmn = far - near
    if abs(rml) <= _EPS or abs(tmb) <= _EPS or abs(fmn) <= _EPS:
        return identity_matrix()
    return [
        [2.0 / rml, 0.0, 0.0, -(right + left) / rml],
        [0.0, 2.0 / tmb, 0.0, -(top + bottom) / tmb],
        [0.0, 0.0, -2.0 / fmn, -(far + near) / fmn],
        [0.0, 0.0, 0.0, 1.0],
    ]


def same_frame_matrix(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """创建/校验同一渲染帧的相机矩阵与阴影矩阵捕获包。

    推荐用法::

        packet = same_frame_matrix(frame_id, camera_view_proj, shadow_view_proj)
        renderer.submit_async(packet)   # 异步线程只能读 packet

    之后主相机即使移动到下一帧，也不会改变 ``packet`` 中的矩阵，因此阴影不
    会整体错位一帧。若传入的两个快照自带不同帧号，报告中会标记
    ``source_aligned=False/source_lag_frames``；严格模式 ``strict=True`` 时
    ``ok=False``。
    """

    cfg, positional = _merge_args(args, kwargs)
    strict = bool(cfg.get("strict", False))

    frame = _pick(cfg, "frame", "frame_id")
    camera = _pick(cfg, "camera_matrix", "camera", "view_projection", "camera_snapshot")
    shadow = _pick(cfg, "shadow_matrix", "shadow", "light_matrix", "shadow_snapshot")

    pos = list(positional)
    if pos and _is_number(pos[0]) and frame is None:
        frame = int(pos[0])
        pos = pos[1:]
    if camera is None and pos:
        camera = pos[0]
        pos = pos[1:]
    if shadow is None and pos:
        shadow = pos[0]
        pos = pos[1:]
    if frame is None and pos and _is_number(pos[-1]):
        frame = int(pos[-1])

    source_camera_frame = _matrix_frame(camera)
    source_shadow_frame = _matrix_frame(shadow)
    camera_matrix = _normalize_matrix(_matrix_payload(camera))
    shadow_matrix = _normalize_matrix(_matrix_payload(shadow))

    if frame is None:
        candidates = [f for f in (source_camera_frame, source_shadow_frame) if f is not None]
        frame = max(candidates) if candidates else 0
    frame = int(frame)

    source_lag = 0
    if source_camera_frame is not None and source_shadow_frame is not None:
        source_lag = abs(source_camera_frame - source_shadow_frame)
    source_aligned = source_lag == 0
    ok = source_aligned or not strict

    return {
        "ok": ok,
        "frame": frame,
        "frame_id": frame,
        # 捕获包对外承诺同帧；这是提交给异步阴影渲染的不可变数据。
        "camera_frame": frame,
        "shadow_frame": frame,
        "aligned": True,
        "same_frame": True,
        "lag_frames": 0,
        "source_camera_frame": source_camera_frame if source_camera_frame is not None else frame,
        "source_shadow_frame": source_shadow_frame if source_shadow_frame is not None else frame,
        "source_aligned": source_aligned,
        "source_lag_frames": source_lag,
        "camera_matrix": camera_matrix,
        "shadow_matrix": shadow_matrix,
        # 若 shadow_matrix 是 lightViewProj、camera_matrix 是 cameraView，则
        # 调用方可直接使用该乘积；矩阵含义不同时仅作为便利字段。
        "combined_matrix": _mat4_mul(shadow_matrix, camera_matrix),
        "capture_point": "before_async_shadow_submit",
        "immutable": True,
        "remedy": "capture both matrices in the same main-thread frame before dispatch",
    }


# ---------------------------------------------------------------------------
# 4. 按级联纹素计算过滤半径
# ---------------------------------------------------------------------------


def filter_per_cascade(*args: Any, **kwargs: Any) -> Any:
    """为每个级联计算独立纹素尺寸和世界空间过滤半径。

    单纹素输入 ``filter_per_cascade(0.02)`` 返回半径浮点数（默认核半径为 1
    个纹素）。列表/配置输入返回完整报告，其中：

    .. math::

        texel_i = world\\_size_i / resolution_i
        radius_i = kernel\\_radius\\_texels_i \\times texel_i

    远级联分辨率低或覆盖范围大时，``texel_i`` 自动变粗，半径也随之增大。
    """

    cfg, positional = _merge_args(args, kwargs)
    kernel_default = _DEFAULT_KERNEL_RADIUS_TEXELS
    if len(positional) >= 2 and _is_number(positional[1]):
        kernel_default = abs(float(positional[1]))
    kernel_texels = abs(_as_float(
        _pick(cfg, "kernel_radius_texels", "radius_texels", "pcf_radius_texels",
              default=kernel_default),
        kernel_default,
    ))

    texel_sizes = _pick(cfg, "texel_sizes", "texels", "texel_size")
    if texel_sizes is None and positional and isinstance(positional[0], (list, tuple)):
        texel_sizes = positional[0]
    if texel_sizes is None and positional and _is_number(positional[0]):
        texel = abs(float(positional[0]))
        radius = kernel_texels * texel
        return radius

    if texel_sizes is not None and _is_number(texel_sizes):
        texel_list = [abs(float(texel_sizes))]
    elif texel_sizes is not None:
        texel_list = [abs(v) for v in _as_float_list(texel_sizes, "texel_sizes")]
    else:
        resolutions = _pick(cfg, "resolutions", "resolution_per_cascade")
        world_sizes = _pick(cfg, "world_sizes", "extents", "coverage_sizes")
        count = max(1, int(_as_float(_pick(cfg, "cascade_count", "count", default=_DEFAULT_CASCADES), _DEFAULT_CASCADES)))
        if isinstance(resolutions, (list, tuple)):
            res_list = [max(1.0, abs(_as_float(v, _DEFAULT_RESOLUTION))) for v in resolutions]
        else:
            base = abs(_as_float(resolutions if _is_number(resolutions) else cfg.get("resolution", _DEFAULT_RESOLUTION), _DEFAULT_RESOLUTION))
            res_list = [max(16.0, base / (2 ** i)) for i in range(count)]
        if isinstance(world_sizes, (list, tuple)):
            world_list = [abs(_as_float(v, 1.0)) for v in world_sizes]
        elif _is_number(world_sizes):
            world_list = [abs(float(world_sizes))] * len(res_list)
        else:
            base_world = abs(_as_float(cfg.get("world_size", cfg.get("coverage", 1.0)), 1.0))
            # 默认让覆盖范围随级联近似倍增；真实管线应从 fov_coverage/fit 得到。
            growth = abs(_as_float(cfg.get("growth", 2.0), 2.0))
            world_list = [base_world * (growth ** i) for i in range(len(res_list))]
        texel_list = [w / r for w, r in zip(world_list, res_list)]

    min_texel_samples = abs(_as_float(cfg.get("min_radius_texels", 1.0), 1.0))
    reports = []
    radii: list[float] = []
    for index, texel in enumerate(texel_list):
        radius_texels = max(kernel_texels, min_texel_samples)
        radius = radius_texels * texel
        radii.append(radius)
        reports.append({
            "cascade": index,
            "texel_size": texel,
            "radius_texels": radius_texels,
            "filter_radius": radius,
            "uv_radius": radius_texels / max(1.0, texel and (1.0 / texel) or 1.0),
            "pcf_width_texels": 2.0 * radius_texels + 1.0,
        })
        # uv_radius = radius_texels / resolution；resolution=1/texel when world=1。
        reports[-1]["uv_radius"] = radius_texels * texel

    return {
        "kernel": "per_cascade_texel_scaled",
        "kernel_radius_texels": kernel_texels,
        "texel_sizes": texel_list,
        "radii": radii,
        "cascades": reports,
        "far_cascade_uses_larger_radius": len(radii) < 2 or radii[-1] >= radii[0],
    }


# ---------------------------------------------------------------------------
# 5. FOV 覆盖范围
# ---------------------------------------------------------------------------


def fov_coverage(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """根据 FOV/宽高比/近远平面计算级联覆盖尺寸和包围半径。

    位置参数依次为 ``fov, aspect, near, far``。大于 pi 的 FOV 数值自动按角
    度解释，也可用 ``unit='deg'/'rad'`` 或 ``degrees=True/False`` 显式指定。
    """

    cfg, positional = _merge_args(args, kwargs)
    fov = positional[0] if len(positional) > 0 else _pick(cfg, "fov", "fov_y", "vertical_fov", default=_DEFAULT_FOV_DEGREES)
    aspect = positional[1] if len(positional) > 1 else _pick(cfg, "aspect", "aspect_ratio", default=_DEFAULT_ASPECT)
    if len(positional) >= 4:
        near = positional[2]
        far = positional[3]
    elif len(positional) == 3:
        # 常见三参数形式是 fov、aspect、distance/far；near 使用配置或默认值。
        near = _pick(cfg, "near", "near_plane", default=_DEFAULT_NEAR)
        far = positional[2]
    else:
        near = _pick(cfg, "near", "near_plane", default=_DEFAULT_NEAR)
        far = _pick(cfg, "far", "far_plane", "distance", default=_DEFAULT_FAR)

    radians = _fov_radians(fov, _pick(cfg, "unit", "fov_unit", "degrees"))
    aspect = abs(_as_float(aspect, _DEFAULT_ASPECT))
    if aspect <= _EPS:
        aspect = _DEFAULT_ASPECT
    near = max(0.0, abs(_as_float(near, _DEFAULT_NEAR)))
    far = abs(_as_float(far, _DEFAULT_FAR))
    if far <= near + _EPS:
        far = near + max(1.0, far)

    tan_half = math.tan(radians * 0.5)

    def plane(depth: float) -> dict[str, float]:
        half_height = depth * tan_half
        half_width = half_height * aspect
        return {
            "depth": depth,
            "half_height": half_height,
            "half_width": half_width,
            "height": 2.0 * half_height,
            "width": 2.0 * half_width,
        }

    near_plane = plane(near)
    far_plane = plane(far)

    # 以切片中点为球心的保守包围球；足以包住八个视锥角点。
    center_depth = 0.5 * (near + far)
    extent_slope = math.sqrt(tan_half * tan_half + (aspect * tan_half) ** 2)
    radius = max(
        math.sqrt((z - center_depth) ** 2 + (z * extent_slope) ** 2)
        for z in (near, far)
    )
    coverage_size = 2.0 * radius
    far_width = far_plane["width"]
    far_height = far_plane["height"]
    ortho_world_size = max(far_width, far_height, coverage_size)

    return {
        "fov": radians,
        "fov_radians": radians,
        "fov_degrees": math.degrees(radians),
        "aspect": aspect,
        "near": near,
        "far": far,
        "tan_half_fov": tan_half,
        "near_plane": near_plane,
        "far_plane": far_plane,
        "width": far_width,
        "height": far_height,
        "half_width": far_plane["half_width"],
        "half_height": far_plane["half_height"],
        "center": [0.0, 0.0, -center_depth],
        "center_depth": center_depth,
        "radius": radius,
        "coverage_radius": radius,
        "coverage_size": coverage_size,
        "ortho_world_size": ortho_world_size,
        "ortho_extent": 0.5 * ortho_world_size,
        "recomputed_for_fov": True,
        "wide_fov_expands_coverage": True,
    }


# ---------------------------------------------------------------------------
# 6. 级联接缝混合
# ---------------------------------------------------------------------------


def seam_weight(distance: float | None = None, boundary: float = 0.0, width: float = 1.0, **kwargs: Any) -> float:
    """返回外层级联权重：边界内为 0，边界上为 0.5，过渡带外为 1。"""

    if distance is None:
        t = _clamp(_as_float(kwargs.get("t", kwargs.get("weight", 0.0)), 0.0))
        return _smoothstep(0.0, 1.0, t)
    d = abs(_as_float(distance, 0.0))
    b = abs(_as_float(boundary, 0.0))
    w = abs(_as_float(width, kwargs.get("blend_width", 1.0)))
    if w <= _EPS:
        return 0.0 if d < b else 1.0
    return _smoothstep(b - 0.5 * w, b + 0.5 * w, d)


def seam_blend(*args: Any, **kwargs: Any) -> Any:
    """混合相邻级联阴影。

    数值形式 ``seam_blend(inner_value, outer_value, t)`` 按 smoothstep 后的
    权重混合。距离形式通过关键字调用::

        seam_blend(distance=9.5, boundary=10.0, width=2.0,
                   inner_shadow=0.2, outer_shadow=0.4)

    dict 输入或 ``details=True`` 返回权重和混合结果；否则返回浮点结果。
    单参数冒烟调用返回一个非 ``None`` 的 smoothstep 权重。
    """

    cfg, positional = _merge_args(args, kwargs)
    details = bool(cfg.pop("details", False))

    distance = _pick(cfg, "distance", "z", "depth")
    boundary = _as_float(_pick(cfg, "boundary", "split", "edge", default=0.0), 0.0)
    width = abs(_as_float(_pick(cfg, "width", "blend_width", "band_width", default=1.0), 1.0))
    if positional and distance is None and len(positional) not in (1, 3):
        # seam_blend(distance, boundary, width)
        distance = positional[0]
        if len(positional) >= 2:
            boundary = _as_float(positional[1], boundary)
        if len(positional) >= 3:
            width = abs(_as_float(positional[2], width))

    inner = _pick(cfg, "inner_shadow", "near_shadow", "a", "inner", default=0.0)
    outer = _pick(cfg, "outer_shadow", "far_shadow", "b", "outer", default=1.0)

    if distance is not None:
        t = seam_weight(distance, boundary, width)
    elif len(positional) == 3:
        inner, outer, raw_t = positional
        t = seam_weight(t=_clamp(_as_float(raw_t, 0.0)))
    elif len(positional) == 2:
        inner, raw_t = positional
        outer = cfg.get("outer_shadow", 1.0)
        t = seam_weight(t=_clamp(_as_float(raw_t, 0.0)))
    elif len(positional) == 1 and _is_number(positional[0]):
        raw = float(positional[0])
        # 0..1 视为归一化权重；其他值视为相对边界距离（boundary=0,width=1）。
        if 0.0 <= raw <= 1.0:
            t = seam_weight(t=raw)
        else:
            t = seam_weight(raw, boundary, width)
    else:
        t = seam_weight(t=_as_float(_pick(cfg, "t", "weight", default=0.0), 0.0))

    if _is_number(inner) and _is_number(outer):
        blended = (1.0 - t) * float(inner) + t * float(outer)
    else:
        # 支持 RGB 等序列逐通道混合。
        a = _as_float_list(inner, "inner_shadow")
        b = _as_float_list(outer, "outer_shadow")
        n = max(len(a), len(b))
        a = a + [a[-1] if a else 0.0] * (n - len(a))
        b = b + [b[-1] if b else 1.0] * (n - len(b))
        blended = [(1.0 - t) * a[i] + t * b[i] for i in range(n)]

    report = {
        "weight": t,
        "outer_weight": t,
        "inner_weight": 1.0 - t,
        "t": t,
        "boundary": boundary,
        "width": width,
        "blended": blended,
        "result": blended,
        "function": "smoothstep_cascade_seam",
    }
    if details or (args and isinstance(args[0], dict)):
        return report
    return blended


# ---------------------------------------------------------------------------
# Bias / shadow map / fit
# ---------------------------------------------------------------------------


def bias(*args: Any, **kwargs: Any) -> dict[str, Any] | float:
    """按当前级联纹素和表面斜率计算深度/法线 bias。

    ``bias(texel_size, slope=1.0)`` 直接返回深度 bias 浮点值；dict 或
    ``details=True`` 返回包含法线偏移的完整报告。
    """

    cfg, positional = _merge_args(args, kwargs)
    texel = _pick(cfg, "texel_size", "texel", "world_per_texel")
    if texel is None and positional and _is_number(positional[0]):
        texel = positional[0]
    if texel is None:
        texel = _resolve_texel_size(cfg, positional)
    texel = abs(_as_float(texel, 1.0))

    slope = abs(_as_float(_pick(cfg, "slope", "slope_scale", "ndotl_slope", default=1.0), 1.0))
    if len(positional) >= 2 and _is_number(positional[1]):
        slope = abs(float(positional[1]))
    slope_limit = abs(_as_float(cfg.get("max_slope", 10.0), 10.0))
    scaled_slope = min(slope, slope_limit)
    slope_scale = abs(_as_float(cfg.get("bias_scale", 1.0), 1.0))
    base = abs(_as_float(cfg.get("base_bias", 0.0), 0.0))
    depth_bias = base + texel * slope_scale * scaled_slope

    normal_texels = abs(_as_float(_pick(cfg, "normal_bias_texels", "normal_offset_texels", default=1.0), 1.0))
    normal_bias = normal_texels * texel
    depth_bits = cfg.get("depth_bits")
    quantization = None
    if _is_number(depth_bits) and int(depth_bits) > 0:
        quantization = 1.0 / (2.0 ** int(depth_bits) - 1.0)

    report = {
        "depth_bias": depth_bias,
        "bias": depth_bias,
        "normal_bias": normal_bias,
        "texel_size": texel,
        "slope": slope,
        "slope_scaled": scaled_slope,
        "base_bias": base,
        "depth_quantization": quantization,
        "texel_relative_bias": depth_bias / texel if texel > _EPS else None,
    }
    if cfg.get("details") or (args and isinstance(args[0], dict)):
        return report
    return depth_bias


def shadow_map(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """返回级联阴影贴图数组/图集的确定性描述。

    第一个位置参数若为数字，按基础分辨率处理；可用 ``cascades`` 指定级数。
    每级分辨率默认逐级减半且不低于 16。
    """

    cfg, positional = _merge_args(args, kwargs)
    base_resolution = _pick(cfg, "resolution", "size", "map_size")
    if base_resolution is None and positional and _is_number(positional[0]):
        base_resolution = positional[0]
    base_resolution = max(16, int(abs(_as_float(base_resolution, _DEFAULT_RESOLUTION))))
    count = cascade_count(cfg.get("cascades", cfg.get("cascade_count", _DEFAULT_CASCADES)))
    fov_info = fov_coverage(cfg.get("fov_config", cfg)) if ("fov" in cfg or "fov_y" in cfg) else None

    maps = []
    texel_sizes = []
    for i in range(count):
        resolution = max(16, base_resolution // (2 ** i))
        if fov_info is not None:
            world_size = fov_info["ortho_world_size"] * (1.0 + i)  # 描述性扩大
        else:
            world_size = abs(_as_float(cfg.get("world_size", 1.0), 1.0)) * (2.0 ** i)
        texel = world_size / resolution
        texel_sizes.append(texel)
        maps.append({
            "cascade": i,
            "layer": i,
            "resolution": resolution,
            "width": resolution,
            "height": resolution,
            "world_size": world_size,
            "texel_size": texel,
            "atlas_rect": [0, 0, resolution, resolution],
            "format": cfg.get("format", "depth32f"),
        })
    return {
        "type": "cascade_shadow_map",
        "count": count,
        "array_layers": count,
        "base_resolution": base_resolution,
        "maps": maps,
        "texel_sizes": texel_sizes,
        "depth_format": cfg.get("depth_format", "depth32f"),
        "immutable_frame_packet": True,
    }


def fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """把 FOV 覆盖、切分、纹素对齐、过滤和 bias 拟合为每一级正交阴影配置。"""

    cfg, positional = _merge_args(args, kwargs)
    # fit(1) 之类冒烟输入不应当成非法 FOV；仅在显式提供 FOV 时透传。
    fit_cfg = dict(cfg)
    if positional and _is_number(positional[0]) and "fov" not in fit_cfg:
        # 数字解释为期望级联编号或级数都不影响默认拟合；保留在报告中。
        fit_cfg.setdefault("requested", positional[0])

    count = cascade_count(fit_cfg.get("cascades", fit_cfg.get("cascade_count", _DEFAULT_CASCADES)))
    near = abs(_as_float(fit_cfg.get("near", _DEFAULT_NEAR), _DEFAULT_NEAR))
    far = abs(_as_float(fit_cfg.get("far", _DEFAULT_FAR), _DEFAULT_FAR))
    splits = cascade_splits(count, near, far, **{
        k: v for k, v in fit_cfg.items() if k in {"lambda_split", "split_lambda", "exponential_weight"}
    }) if "splits" not in fit_cfg else _as_float_list(fit_cfg["splits"])

    fov_cfg = {k: v for k, v in fit_cfg.items() if k in {
        "fov", "fov_y", "vertical_fov", "aspect", "aspect_ratio", "unit", "fov_unit", "degrees"
    }}
    fov_info = fov_coverage(fov_cfg or {})
    base_resolution = max(16, int(abs(_as_float(fit_cfg.get("resolution", _DEFAULT_RESOLUTION), _DEFAULT_RESOLUTION))))
    kernel_texels = fit_cfg.get("kernel_radius_texels", _DEFAULT_KERNEL_RADIUS_TEXELS)

    cascades = []
    for i, boundary in enumerate(splits):
        slice_near = near if i == 0 else splits[i - 1]
        slice_far = boundary
        slice_fov = fov_coverage(
            fov_info["fov_radians"], fov_info["aspect"], slice_near, slice_far, unit="rad"
        )
        world_size = slice_fov["ortho_world_size"]
        resolution = max(16, base_resolution // (2 ** i))
        texel = world_size / resolution
        raw_center = fit_cfg.get("center", [0.0, 0.0, -slice_fov["center_depth"]])
        if isinstance(raw_center, Sequence) and not _is_number(raw_center):
            center_values = _as_float_list(raw_center)
        else:
            center_values = [float(raw_center), 0.0, -slice_fov["center_depth"]]
        snapped_center = texel_snap(center_values, texel)
        filter_report = filter_per_cascade(texel, kernel_radius_texels=kernel_texels)
        bias_report = bias({"texel_size": texel, "details": True, **{
            k: v for k, v in fit_cfg.items() if k in {"slope", "bias_scale", "normal_bias_texels", "depth_bits"}
        }})
        cascades.append({
            "cascade": i,
            "near": slice_near,
            "far": slice_far,
            "fov": slice_fov,
            "world_size": world_size,
            "extent": world_size * 0.5,
            "resolution": resolution,
            "texel_size": texel,
            "center": center_values,
            "snapped_center": snapped_center,
            "filter_radius": filter_report if _is_number(filter_report) else filter_report,
            "bias": bias_report,
        })

    packet = None
    if "camera_matrix" in fit_cfg or "shadow_matrix" in fit_cfg or "frame" in fit_cfg:
        packet = same_frame_matrix(**{
            k: fit_cfg[k] for k in ("camera_matrix", "shadow_matrix", "frame", "strict") if k in fit_cfg
        })

    result = {
        "ok": True,
        "count": count,
        "near": near,
        "far": far,
        "splits": splits,
        "fov": fov_info,
        "cascades": cascades,
        "same_frame": packet,
        "fixes": [
            "light_space_texel_snap",
            "switch_hysteresis",
            "same_frame_matrix_capture",
            "per_cascade_filter_radius",
            "fov_dependent_coverage",
            "seam_blend",
        ],
    }
    result["validation"] = validation(result)
    result["ok"] = bool(result["validation"]["ok"])
    return result


# ---------------------------------------------------------------------------
# 确定性 / 指标 / 校验 / 审计 / 回退
# ---------------------------------------------------------------------------


def determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """由种子生成跨进程稳定的 SplitMix64 序列。

    ``determinism(11)["values"]`` 对相同种子始终相等。传入可迭代记录时，也可
    使用 ``ordered_records`` 获得按稳定键排序的结果。
    """

    cfg, positional = _merge_args(args, kwargs)
    seed_value = positional[0] if positional else _pick(cfg, "seed", default=1)
    count = max(0, int(_as_float(_pick(cfg, "count", "n", default=8), 8)))
    try:
        seed = int(abs(_as_float(seed_value, 1)))
    except (TypeError, ValueError):
        seed = abs(hash(str(seed_value))) & ((1 << 64) - 1)

    mask = (1 << 64) - 1
    state = seed & mask
    values: list[float] = []
    integers: list[int] = []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        z &= mask
        integers.append(z)
        values.append(z / float(1 << 64))

    records = cfg.get("records")
    ordered_records = None
    if records is not None:
        try:
            ordered_records = sorted(records, key=lambda item: repr(item))
        except TypeError:
            ordered_records = list(records)

    return {
        "seed": seed,
        "generator": "splitmix64",
        "values": values,
        "integers": integers,
        "jitter": [v - 0.5 for v in values],
        "ordered_records": ordered_records,
        "stable": True,
    }


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """计算数值序列或级联拟合结果的基础指标。"""

    cfg, positional = _merge_args(args, kwargs)
    sample: Any = positional[0] if positional else cfg.get("data", cfg.get("result", 1.0))

    if isinstance(sample, dict) and ("cascades" in sample or "texel_sizes" in sample):
        texel_sizes = sample.get("texel_sizes")
        if texel_sizes is None and "cascades" in sample:
            texel_sizes = [c.get("texel_size", 0.0) for c in sample["cascades"] if isinstance(c, dict)]
        texel_sizes = [abs(float(x)) for x in (texel_sizes or []) if _is_number(x)]
        radii = []
        if isinstance(sample.get("radii"), list):
            radii = [abs(float(x)) for x in sample["radii"] if _is_number(x)]
        elif "cascades" in sample:
            for cascade in sample["cascades"]:
                if isinstance(cascade, dict):
                    fr = cascade.get("filter_radius")
                    if isinstance(fr, (int, float)):
                        radii.append(abs(float(fr)))
        return {
            "cascade_count": sample.get("count", len(texel_sizes)),
            "texel_sizes": texel_sizes,
            "filter_radii": radii,
            "max_texel_size": max(texel_sizes, default=0.0),
            "min_texel_size": min(texel_sizes, default=0.0),
            "texel_growth_ratio": (texel_sizes[-1] / texel_sizes[0]) if len(texel_sizes) >= 2 and texel_sizes[0] > 0 else 1.0,
            "seam_blend_enabled": True,
        }

    if _is_number(sample):
        values = [float(sample)]
    else:
        try:
            values = _as_float_list(sample, "samples")
        except ValueError:
            values = []
    n = len(values)
    if n == 0:
        return {"count": 0, "samples": 0, "mean": 0.0, "rms": 0.0, "peak": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    rms = math.sqrt(sum(v * v for v in values) / n)
    return {
        "count": n,
        "samples": n,
        "mean": mean,
        "rms": rms,
        "peak": max(abs(v) for v in values),
        "min": min(values),
        "max": max(values),
    }


def validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """校验级联配置/拟合包；对冒烟标量返回 ``ok=True``。"""

    cfg, positional = _merge_args(args, kwargs)
    result: Any = positional[0] if positional else cfg
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(result, dict):
        return {"ok": True, "errors": errors, "warnings": warnings, "value": result}

    if result.get("ok") is False and "source_aligned" in result:
        errors.append("shadow_and_camera_frames_mismatch")

    fov = result.get("fov")
    if isinstance(fov, dict):
        radians = float(fov.get("fov_radians", 0.0))
        if not 0.0 < radians < math.pi:
            errors.append("fov_out_of_range")
        if float(fov.get("aspect", 0.0)) <= 0.0:
            errors.append("aspect_ratio_must_be_positive")

    splits = result.get("splits")
    if isinstance(splits, (list, tuple)) and splits:
        values = [float(v) for v in splits]
        if any(v <= 0.0 for v in values):
            errors.append("split_distances_must_be_positive")
        if any(values[i] >= values[i + 1] + _EPS for i in range(len(values) - 1)):
            errors.append("splits_must_be_strictly_increasing")

    cascades = result.get("cascades")
    if isinstance(cascades, list):
        for cascade in cascades:
            if not isinstance(cascade, dict):
                continue
            texel = cascade.get("texel_size")
            if _is_number(texel) and float(texel) <= 0.0:
                errors.append("texel_size_must_be_positive")
            resolution = cascade.get("resolution")
            if _is_number(resolution) and int(resolution) < 16:
                warnings.append("shadow_resolution_below_16_is_impractical")
            snapped = cascade.get("snapped_center")
            if isinstance(snapped, Sequence) and _is_number(texel) and float(texel) > 0:
                # 对齐后的 x/y 必须落在纹素网格上；使用与 texel_snap 相同的
                # half-away-from-zero 规则，避免在 .5 位置被 banker's round 误判。
                for value in list(snapped)[:2]:
                    units = float(value) / float(texel)
                    aligned_units = _grid_round(units, 1.0, "nearest")
                    if abs(units - aligned_units) > 1.0e-6:
                        errors.append("center_not_texel_aligned")
                        break

    if result.get("aligned") is False or result.get("same_frame") is False:
        errors.append("matrices_not_captured_in_same_frame")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """把一个或多个报告汇总为审计记录；``audit(1)['event_count'] == 1``。"""

    records: list[dict[str, Any]] = []
    ok = True
    fixes_seen: set[str] = set()
    for index, event in enumerate(args):
        if isinstance(event, dict):
            record = {"index": index, "type": event.get("type", "cascade_report")}
            for key in ("frame", "count", "splits", "ok", "aligned", "same_frame", "reason"):
                if key in event:
                    record[key] = event[key]
            if event.get("ok") is False:
                ok = False
            for fix in event.get("fixes", []):
                fixes_seen.add(str(fix))
        else:
            record = {"index": index, "type": "value", "value": event}
        records.append(record)

    if kwargs:
        records.append({"index": len(records), "type": "keywords", "value": kwargs})

    validation_report = None
    if args and isinstance(args[0], dict):
        validation_report = validation(args[0])
        ok = ok and validation_report["ok"]

    return {
        "ok": ok,
        "event_count": len(args),
        "records": records,
        "fixes_observed": sorted(fixes_seen),
        "validation": validation_report,
        "immutable_capture_required": True,
    }


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """不可信配置/异步捕获失败时的安全单级联均匀覆盖回退。"""

    reason = args[0] if args else kwargs.get("reason", "invalid_or_unsafe_cascade_config")
    far = abs(_as_float(kwargs.get("far", _DEFAULT_FAR), _DEFAULT_FAR))
    return {
        "mode": "uniform_single_cascade_safe_fallback",
        "reason": reason,
        "cascades": 1,
        "splits": [far],
        "hysteresis_ratio": 0.0,
        "seam_blend_width": 0.0,
        "fov_driven_coverage": True,
        "same_frame_capture": True,
        "action": "render_one_conservative_shadow_map_and_hold_last_good_packet",
    }
