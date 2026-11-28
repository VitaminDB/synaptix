# vllm-project__vllm-omni__h3-sampling-code

Source: https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/diffusion/models/minimax_h3
Fetched: 2026-11-28



---

## FILE: vllm_omni/diffusion/models/minimax_h3/sampling.py (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/diffusion/models/minimax_h3/sampling.py)

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 x0 model adapters for sampler-owned, coupled video/audio integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cancellation import check_request_cancellation
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.sampler import ModelEvaluation, SamplingState

from .latent_mask import MiniMaxH3LatentEdit
from .scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0

if TYPE_CHECKING:
    from .denoise_loop import MiniMaxH3DenoiseBranch

logger = init_logger(__name__)


def _assign_rows(rows: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> None:
    """``rows[mask] = values`` through the mask's precomputed row ``index``, which needs no host sync.

    Every caller passes one value row per target row (sampler latents and
    locked audio rows are both sized to the target rows), so any other shape
    is a bug and raises instead of broadcasting.
    """
    expected = (index.shape[0], *rows.shape[1:])
    if tuple(values.shape) != expected:
        raise ValueError(f"target row values have shape {tuple(values.shape)}, expected {expected}")
    rows.index_copy_(0, index, values)


class MiniMaxH3Model:
    """Translate sampler evaluations into H3 inputs and raw velocity into x0."""

    def __init__(
        self,
        transformer,
        branch: MiniMaxH3DenoiseBranch,
        rows: tuple[torch.Tensor, torch.Tensor],
        edits: tuple[MiniMaxH3LatentEdit | None, MiniMaxH3LatentEdit | None] = (None, None),
        cond_timesteps: tuple[float, float] = (0.999, 1.0),
        model_rows: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_schedule=None,
        attention_sigma_schedule=(),
    ):
        self.transformer, self.branch, self.rows = transformer, branch, rows
        self.attention_schedule = attention_schedule
        self.attention_sigma_schedule = attention_sigma_schedule
        self.edits, self.cond_timesteps = edits, cond_timesteps
        # Fixed condition rows stay in this buffer; each prediction replaces
        # the target rows. Keep it for the entire request, including retries.
        self.model_rows = model_rows if model_rows is not None else tuple(row.clone() for row in rows)
        self._indices: tuple[torch.Tensor, ...] = ()
        self._indices_branch = None

    def rows_for(self, latents):
        """Fill reusable model inputs; valid until this request's next prediction."""
        rows = self.model_rows
        for row, index, latent in zip(rows, self.indices, latents, strict=True):
            _assign_rows(row, index, latent)
        if self.branch.locked_audio_rows is not None:
            _assign_rows(rows[1], self.indices[1], self.branch.locked_audio_rows)
        return tuple(rows)

    @property
    def masks(self):
        return self.branch.update_mask_dev, self.branch.audio_update_mask_dev

    @property
    def indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._indices_branch is not self.branch:
            self._indices = tuple(
                index if index is not None else mask.nonzero().view(-1)
                for index, mask in zip(
                    (getattr(self.branch, "update_idx_dev", None), getattr(self.branch, "audio_update_idx_dev", None)),
                    self.masks,
                    strict=True,
                )
            )
            self._indices_branch = self.branch
        return self._indices

    def prepare(self, evaluation: ModelEvaluation):
        rows = self.rows_for(evaluation.latents)
        times = tuple(1.0 - sigma for sigma in evaluation.sigmas)
        cond = tuple(max(t, anchor) for t, anchor in zip(times, self.cond_timesteps, strict=True))
        prepared = []
        for row, mask, edit, t, c, sigma in zip(
            rows, self.masks, self.edits, times, cond, evaluation.sigmas, strict=True
        ):
            target_timesteps = None
            if edit is not None:
                row[mask], target_timesteps = edit.prepare(row[mask], t, c, sigma=sigma)
            prepared.append((row, target_timesteps))
        return rows, prepared, times, cond

    def to_denoised(self, prepared, velocities):
        rows, model_rows, times, _ = prepared
        return tuple(
            minimax_h3_rf_v_to_x0(
                row.index_select(0, index),
                v.float().index_select(0, index),
                torch.full((), t, dtype=row.dtype, device=row.device),
            )
            if edit is None
            else edit.x0(model_row[0][mask], v.float()[mask], t)
            for row, model_row, mask, index, edit, t, v in zip(
                rows, model_rows, self.masks, self.indices, self.edits, times, velocities, strict=True
            )
        )

    def validate_write_back(self, state: SamplingState):
        """Check both streams before any request's full rows are updated."""
        if self.branch.locked_audio_rows is not None:
            state.latents = (state.latents[0], self.branch.locked_audio_rows)
        for row, index, latent in zip(self.rows, self.indices, state.latents, strict=True):
            shape = (index.shape[0], *row.shape[1:])
            if latent.shape != shape or latent.dtype != row.dtype or latent.device != row.device:
                raise ValueError("Sampled latents must match the target rows' shape, dtype, and device")
        if not bool(torch.stack([torch.isfinite(latent).all() for latent in state.latents]).all().item()):
            raise ValueError("Sampled latents must be finite before writing rows")

    def write_back(self, state: SamplingState):
        """Write checked latents into the existing full video/audio rows."""
        for row, index, latent in zip(self.rows, self.indices, state.latents, strict=True):
            _assign_rows(row, index, latent)
        return self.rows


class MiniMaxH3ModelBatch:
    """Callable x0 model; pack compatible evaluations without advancing samplers."""

    def __init__(self, models: dict[SamplingState, MiniMaxH3Model], packed_supported: Callable = lambda _: False):
        self.models = models
        self.packed_supported = packed_supported

    def _unpacked_reason(
        self, models: list[MiniMaxH3Model], *, pdd: bool, scheduled: bool, uniform_progress: bool
    ) -> tuple[Callable[..., None], str] | None:
        """Why these requests cannot share one packed forward, or None if they can."""
        if pdd:
            # PDD heads take one request-local step index per forward.
            return logger.debug, "MiniMax H3 PDD heads execute requests separately with request-local step indices."
        if len({id(m.transformer) for m in models}) > 1:
            return logger.warning_once, "MiniMax H3 step batch contains requests for different task-specific DiTs."
        if any("reference_kv_state" in m.branch.static_kwargs for m in models):
            # Packed kwargs carry no per-request reference-KV state.
            return logger.debug, "MiniMax H3 reference-KV requests run separately."
        if scheduled and (
            not uniform_progress
            or len({m.attention_schedule for m in models}) > 1
            or len({m.attention_sigma_schedule for m in models}) > 1
        ):
            # A packed forward selects one attention profile from the published progress.
            # Sigma windows split like step ranges. The resolved windows are in the step batch key, and
            # the runner's batch_attention_sigma_schedule() rejects differing ones before denoise_step,
            # so differing windows cannot reach one batch; the check mirrors the step-range one.
            return (
                logger.info_once,
                "MiniMax H3 runs a scheduled step batch whose requests differ in progress or schedule "
                "one request per forward, so each request selects attention at its own step.",
            )
        # Checked under the published progress: a scheduled step's selected
        # profile, not the baseline, must isolate requests.
        if self.packed_supported(models[0].transformer):
            return None
        from .minimax_h3_blocks import MiniMaxH3Attention

        modules = getattr(models[0].transformer, "modules", None)
        if callable(modules) and any(
            getattr(getattr(module, "attention", None), "use_ring", False)
            for module in modules()
            if isinstance(module, MiniMaxH3Attention)
        ):
            return (
                logger.warning_once,
                "MiniMax H3 step batching is disabled when ring attention is active: the ring kernels ignore "
                "packed cu_seqlens and would attend across request boundaries.",
            )
        return (
            logger.warning_once,
            "MiniMax H3 step batching needs every attention on a backend that isolates packed multi-document "
            "cu_seqlens (see AttentionBackend.supports_multi_doc_packed_varlen).",
        )

    def __call__(self, evaluations: Sequence[ModelEvaluation]):
        from .batched_packing import minimax_h3_batched_forward_kwargs
        from .denoise_loop import minimax_h3_publish_denoise_progress

        if not evaluations:
            return []
        check_request_cancellation()
        models = [self.models[e.state] for e in evaluations]
        request_progress = [(e.state.step_index, e.sigmas[0], len(e.state.sigmas[0]) - 1) for e in evaluations]
        uniform_progress = len(set(request_progress)) == 1
        # Requests at different points have nothing common to publish, so
        # step-gated attention stays dense, its safe default.
        minimax_h3_publish_denoise_progress(*(request_progress[0] if uniform_progress else (None, None, None)))
        context = get_forward_context() if is_forward_context_available() else None
        try:
            transformer = models[0].transformer
            pdd = (
                getattr(getattr(transformer, "arch", None), "pdd_steps", None) is not None
                or getattr(getattr(transformer, "final_layer", None), "fused_pdd_heads", None) is not None
            )
            if len(models) > 1:
                scheduled = context is not None and bool(
                    getattr(context, "attention_schedule", None) or getattr(context, "attention_sigma_schedule", None)
                )
                reason = self._unpacked_reason(models, pdd=pdd, scheduled=scheduled, uniform_progress=uniform_progress)
                if reason is not None:
                    level, message = reason
                    level(message + " Running %d requests one forward at a time.", len(models))
                    return [self([e])[0] for e in evaluations]
            prepared = [m.prepare(e) for m, e in zip(models, evaluations, strict=True)]
            if len(models) == 1:
                _, rows, times, cond = prepared[0]
                kwargs = models[0].branch.forward_kwargs(
                    video_rows=rows[0][0],
                    audio_rows=rows[1][0],
                    t_video=times[0],
                    t_audio=times[1],
                    imgvid_cond_timestep=cond[0],
                    audio_ref_cond_timestep=cond[1],
                    video_target_timesteps=rows[0][1],
                    audio_target_timesteps=rows[1][1],
                )
            else:
                kwargs = minimax_h3_batched_forward_kwargs(
                    branches=[m.branch for m in models],
                    video_rows=[p[1][0][0] for p in prepared],
                    audio_rows=[p[1][1][0] for p in prepared],
                    t_video=[p[2][0] for p in prepared],
                    t_audio=[p[2][1] for p in prepared],
                    imgvid_cond_timesteps=[p[3][0] for p in prepared],
                    audio_ref_cond_timesteps=[p[3][1] for p in prepared],
                    video_target_timesteps=[p[1][0][1] for p in prepared],
                    audio_target_timesteps=[p[1][1][1] for p in prepared],
                )
            if pdd:
                kwargs["pdd_step"] = evaluations[0].state.step_index
            reference_kv_state = kwargs.get("reference_kv_state")
            if reference_kv_state is not None:
                reference_kv_state.begin_step(evaluations[0].state.step_index, sigma=float(evaluations[0].sigmas[0]))
            video, audio = models[0].transformer(**kwargs)
            if reference_kv_state is not None:
                reference_kv_state.end_step()
            check_request_cancellation()
            video_parts = video.split([m.rows[0].shape[0] for m in models])
            audio_parts = audio.split([m.rows[1].shape[0] for m in models])
            return [
                m.to_denoised(p, (v, a)) for m, p, v, a in zip(models, prepared, video_parts, audio_parts, strict=True)
            ]
        finally:
            minimax_h3_publish_denoise_progress(None, None, None)

```


---

## FILE: vllm_omni/diffusion/models/minimax_h3/time_request.py (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/diffusion/models/minimax_h3/time_request.py)

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

from collections.abc import Sequence

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule


def _align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    while current % 17 != 5:
        current += 1
    return current


def _video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def _frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def _audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def _time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
    base_schedule: Sequence[float] | None = None,
) -> list[float]:
    """Build N + 1 sigma boundaries for N denoiser evaluations.

    ``base_schedule`` supplies the rectified-flow positions explicitly and takes
    precedence over ``num_steps``. Distilled checkpoints need it because their
    few-step schedule is not the uniform one ``num_steps`` produces; validation
    and the shift itself live in the shared :class:`DMD2SigmaSchedule`.
    """
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")

    if base_schedule is not None:
        return DMD2SigmaSchedule.from_positions(base_schedule).shifted_sigmas(shift_scale)

    import torch

    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    # Requests count denoiser evaluations, not sigma points. Include both
    # endpoints so even a one-step request traverses the full [1.0, 0.0] range.
    base = torch.linspace(
        1.0,
        0.0,
        int(num_steps) + 1,
        device="cpu",
        dtype=torch.float32,
    )
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    # In float32 the first sigma can round one ulp above 1 (flow_shift 2.14, for example), which the
    # sampler rejects. The last sigma is exactly 0 and the others are positive, so only 1 needs a clamp.
    return shifted.clamp(max=1.0).tolist()


class MiniMaxH3ShapePlanner:
    """Compute MiniMax H3 frame, latent-shape, and time-shift values."""

    def align_frame_count(self, frame_count: int) -> int:
        return _align_frame_count(frame_count)

    def video_latent_t(self, frame_count: int) -> int:
        return _video_latent_t(frame_count)

    def frame_count_from_video_latent_t(self, out_t: int) -> int:
        return _frame_count_from_video_latent_t(out_t)

    def audio_latent_t(self, duration_seconds: float) -> int:
        return _audio_latent_t(duration_seconds)

    def time_shift_sigmas(
        self,
        *,
        num_steps: int = 50,
        shift_scale: float = 6.0,
        base_schedule: Sequence[float] | None = None,
    ) -> list[float]:
        return _time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=shift_scale,
            base_schedule=base_schedule,
        )


MINIMAX_H3_SHAPE_PLANNER = MiniMaxH3ShapePlanner()

# Default denoise step count when a request omits ``num_inference_steps``;
# the diffusion-stage resolver and the split-stage intake must agree on it.
MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS = 50


def minimax_h3_align_frame_count(frame_count: int) -> int:
    return MINIMAX_H3_SHAPE_PLANNER.align_frame_count(frame_count)


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    return MINIMAX_H3_SHAPE_PLANNER.frame_count_from_video_latent_t(out_t)


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
    base_schedule: Sequence[float] | None = None,
) -> list[float]:
    return MINIMAX_H3_SHAPE_PLANNER.time_shift_sigmas(
        num_steps=num_steps,
        shift_scale=shift_scale,
        base_schedule=base_schedule,
    )

```


---

## FILE: vllm_omni/diffusion/models/minimax_h3/variant.py (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/diffusion/models/minimax_h3/variant.py)

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What the weights a MiniMax-H3 request samples with imply for the pipeline.

H3 is served from many weight sets: the released checkpoint, full students
(FastH3 V2, FastH3 Omni), adapters fused at load (FastH3 Preview) and
request-scoped adapters (LightX2V Turbo, native distilled LoRAs, ordinary
LoRAs). They differ in how a request is *sampled*: the sigma positions, the
step count, the flow shifts, and whether the output heads are parallel-decoding
(PDD) heads. Those contracts stay with the spec that loads each artifact.

They do not differ in how a request is *conditioned*. ControlNet hints,
latent-tail continuation, latent-mask editing and timeline guides are rows and
masks packed around the target latent; none of them reads the schedule, so the
pipeline serves them on every weight set without asking the variant. The one
conditioning restriction a weight set carries is why it refuses timeline guides.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class H3Variant:
    """The weight set behind a request, as far as the pipeline needs to know.

    ``pdd`` selects the parallel-decoding sampler: sigma nodes on the release's
    own grid and one fused output head per step. ``distilled_adapters`` says
    whether a few-step adapter (Turbo, native distilled LoRA) may be activated
    on top; a full student already pins its schedule, so it takes ordinary
    adapters only. ``guides_unsupported`` is the message a timeline-guide
    request receives, or ``None`` where guides are served.
    """

    name: str
    pdd: bool = False
    distilled_adapters: bool = True
    guides_unsupported: str | None = None


_GUIDES_NEED_REFERENCE_STUDENT = "MiniMax H3 timeline guides require base H3 weights or FastH3 Omni"

BASE_H3 = H3Variant(name="MiniMax H3")
# Ref2VA student sampled on its PDD grid. Its eight forwards are the whole
# schedule, so few-step adapters have nothing to bind to.
FASTH3_OMNI = H3Variant(name="FastH3 Omni", pdd=True, distilled_adapters=False)
# T2VA-only students trained with VSA compression gates. Guide rows change the
# token layout those gates were trained on, and no guided output of either has
# been checked, so guides stay off until one is.
FASTH3_V2 = H3Variant(name="FastH3 V2", guides_unsupported=_GUIDES_NEED_REFERENCE_STUDENT)
FASTH3_PREVIEW = H3Variant(name="FastH3 Preview", guides_unsupported=_GUIDES_NEED_REFERENCE_STUDENT)


def control_task(partition: str) -> str | None:
    """The task a control request must use on a server that serves ``partition``.

    The control branch is bound to the primary DiT. On an FL2VA DiT it hints a
    text-only generation; on a Ref2VA DiT it hints the target rows of a
    reference-conditioned one, and reference rows receive no hint. A combined
    server has two DiTs and one branch, so it serves no control.
    """
    if partition == "ref2va":
        return "ref2va"
    if partition == "fl2va":
        return "t2va"
    return None

```


---

## FILE: vllm_omni/diffusion/models/minimax_h3/quality_policy.py (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/diffusion/models/minimax_h3/quality_policy.py)

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-owned request quality policy for MiniMax H3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

from vllm_omni.diffusion.cache.cachedit import CacheDiTRequestSpec
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.errors import OmniClientError

MINIMAX_H3_GENERIC_CACHE_KEY = "minimax_h3.generic"
MINIMAX_H3_HIGH_CACHE_KEY = "minimax_h3.high"
MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG = "force_refresh_step_hint"
MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG = "force_refresh_step_policy"
MINIMAX_H3_FORCE_REFRESH_POLICIES = ("once", "repeat")


def _high_quality_cache_config() -> DiffusionCacheConfig:
    return DiffusionCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=4,
        residual_diff_threshold=0.04,
        max_continuous_cached_steps=1,
        enable_taylorseer=False,
        scm_steps_mask_policy=None,
    )


def _resolve_force_refresh(
    extra_args: Mapping[str, Any] | None,
    *,
    num_inference_steps: int,
) -> tuple[int, str] | None:
    """Resolve H3's request-scoped Cache-DiT refresh hint.

    These are model-specific ``extra_args`` rather than global sampling fields,
    keeping other diffusion models unchanged.
    """

    if not extra_args:
        return None

    raw_hint = extra_args.get(MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG)
    raw_policy = extra_args.get(MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG)

    if raw_hint is None:
        if raw_policy is not None:
            raise OmniClientError("MiniMax H3 force_refresh_step_policy requires force_refresh_step_hint")
        return None
    if isinstance(raw_hint, bool) or not isinstance(raw_hint, Integral):
        raise OmniClientError("MiniMax H3 force_refresh_step_hint must be a positive integer")

    hint = int(raw_hint)
    if not 1 <= hint <= num_inference_steps:
        raise OmniClientError(
            "MiniMax H3 force_refresh_step_hint must be between 1 and "
            f"num_inference_steps ({num_inference_steps}), got {hint}"
        )

    policy = "once" if raw_policy is None else raw_policy
    if not isinstance(policy, str) or policy not in MINIMAX_H3_FORCE_REFRESH_POLICIES:
        raise OmniClientError(
            "MiniMax H3 force_refresh_step_policy must be one of "
            f"{list(MINIMAX_H3_FORCE_REFRESH_POLICIES)}, got {policy!r}"
        )
    return hint, policy


def _has_force_refresh_args(extra_args: Mapping[str, Any] | None) -> bool:
    if not extra_args:
        return False
    return any(
        extra_args.get(name) is not None
        for name in (
            MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG,
            MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG,
        )
    )


def _with_force_refresh(
    cache_config: DiffusionCacheConfig,
    force_refresh: tuple[int, str] | None,
) -> DiffusionCacheConfig:
    if force_refresh is None:
        return cache_config
    hint, policy = force_refresh
    return replace(
        cache_config,
        force_refresh_step_hint=hint,
        force_refresh_step_policy=policy,
    )


def _cache_installation_key(base_key: str, force_refresh: tuple[int, str] | None) -> str:
    """Make refresh-policy transitions explicit to the request runtime.

    Cache-DiT cannot clear an existing ``force_refresh_step_hint`` through its
    incremental context update API because ``None`` is treated as "keep the
    old value".  Including the request hint in the installation key therefore
    makes a hint change perform a safe hook reinstall, while repeated requests
    with the same hint still only refresh the context.
    """

    if force_refresh is None:
        return base_key
    hint, policy = force_refresh
    return f"{base_key}:force_refresh={hint}:{policy}"


@dataclass(frozen=True)
class MiniMaxH3QualityPlan:
    """Resolved execution choices for one MiniMax H3 request."""

    cache_dit: CacheDiTRequestSpec | None


class MiniMaxH3QualityPolicy:
    """Resolve H3 request quality into a declarative Cache-DiT target.

    When the server starts with Cache-DiT, omitted quality selects the
    server-configured profile. Otherwise, omitted quality selects no cache.
    In either case, ``lossless`` selects no cache and ``high`` selects H3's
    high-quality profile. The policy therefore owns whether a request installs
    Cache-DiT; the startup backend only controls the omitted-quality default.
    H3-specific Cache-DiT refresh hints are read from request ``extra_args``;
    they do not change the global diffusion request contract.
    The pipeline owns applying the resulting target at the request boundary.
    """

    def __init__(self, od_config: Any) -> None:
        self._od_config = od_config
        self._configured_backend = str(getattr(od_config, "cache_backend", "none") or "none").lower()

    def resolve(
        self,
        *,
        quality: str | None,
        num_inference_steps: int,
        extra_args: Mapping[str, Any] | None = None,
    ) -> MiniMaxH3QualityPlan:
        if quality == "high":
            base_key = MINIMAX_H3_HIGH_CACHE_KEY
            base_config = _high_quality_cache_config()
        elif self._configured_backend == "cache_dit" and quality is None:
            base_key = MINIMAX_H3_GENERIC_CACHE_KEY
            base_config = self._od_config.cache_config
        else:
            if _has_force_refresh_args(extra_args):
                raise OmniClientError("MiniMax H3 force-refresh arguments require an active Cache-DiT request target")
            return MiniMaxH3QualityPlan(cache_dit=None)

        force_refresh = _resolve_force_refresh(
            extra_args,
            num_inference_steps=num_inference_steps,
        )
        return MiniMaxH3QualityPlan(
            cache_dit=CacheDiTRequestSpec(
                installation_key=_cache_installation_key(base_key, force_refresh),
                cache_config=_with_force_refresh(base_config, force_refresh),
                num_inference_steps=num_inference_steps,
            ),
        )


__all__ = [
    "MINIMAX_H3_GENERIC_CACHE_KEY",
    "MINIMAX_H3_HIGH_CACHE_KEY",
    "MiniMaxH3QualityPlan",
    "MiniMaxH3QualityPolicy",
]

```


---

## FILE: vllm_omni/diffusion/models/minimax_h3/scheduling_minimax_h3_euler_ancestral.py (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/diffusion/models/minimax_h3/scheduling_minimax_h3_euler_ancestral.py)

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import math
from typing import Any

import torch

from vllm_omni.diffusion.sampler import euler_eta0_step

minimax_h3_euler_eta0_step = euler_eta0_step


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_unit_timestep(timestep: torch.Tensor, name: str) -> None:
    if not isinstance(timestep, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(timestep):
        raise ValueError(f"{name} must be a floating point tensor")
    _require_finite_tensor(timestep, name)
    out_of_range = (timestep < 0) | (timestep > 1)
    if bool(out_of_range.any().item()):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"{name} must be finite")
    if sigma < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return sigma


def _validate_timestep_sigma_pair(
    timestep: torch.Tensor,
    sigma_curr: float,
    name: str,
) -> float:
    _validate_unit_timestep(timestep, f"{name}_timestep")
    sigma = _validate_sigma(sigma_curr, f"{name}_sigma_curr")
    expected = 1.0 - timestep.detach().to(dtype=torch.float32)
    actual = torch.full_like(expected, sigma)
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{name}_sigma_curr must equal 1 - {name}_timestep")
    return sigma


def minimax_h3_rf_v_to_x0(
    xt: torch.Tensor,
    v: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if xt.shape != v.shape:
        raise ValueError(f"xt and v shapes must match, got {xt.shape} vs {v.shape}")
    if not torch.is_floating_point(xt):
        raise ValueError("xt must be a floating point tensor")
    if not torch.is_floating_point(v):
        raise ValueError("v must be a floating point tensor")
    _require_finite_tensor(xt, "xt")
    _require_finite_tensor(v, "v")
    _validate_unit_timestep(timestep, "timestep")
    cond_t = timestep.to(device=xt.device, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    sigma_t = 1 - cond_t
    x0 = xt + sigma_t * v
    _require_finite_tensor(x0, "x0")
    return x0


class MiniMaxH3EulerAncestralEta0SchedulerAdapter:
    def __init__(self, **config: Any) -> None:
        if config:
            raise ValueError(f"{type(self).__name__} does not accept config fields: {sorted(config)}")

    def set_shift(self, _flow_shift: float) -> None:
        """Ignore flow shift, matching the previous loader-specific path."""

    def step_denoising(
        self,
        *,
        input_visual_latent: torch.Tensor,
        input_audio_latent: torch.Tensor,
        timestep: torch.Tensor,
        noise_pred_visual: torch.Tensor,
        noise_pred_audio: torch.Tensor,
        sigma_curr: float,
        sigma_next: float,
        video_timestep: torch.Tensor | None = None,
        audio_timestep: torch.Tensor | None = None,
        video_sigma_curr: float | None = None,
        video_sigma_next: float | None = None,
        audio_sigma_curr: float | None = None,
        audio_sigma_next: float | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_timestep = timestep if video_timestep is None else video_timestep
        audio_timestep = timestep if audio_timestep is None else audio_timestep
        visual_sigma_curr = sigma_curr if video_sigma_curr is None else video_sigma_curr
        visual_sigma_next = sigma_next if video_sigma_next is None else video_sigma_next
        audio_sigma_curr = sigma_curr if audio_sigma_curr is None else audio_sigma_curr
        audio_sigma_next = sigma_next if audio_sigma_next is None else audio_sigma_next
        visual_sigma_curr = _validate_timestep_sigma_pair(
            visual_timestep,
            visual_sigma_curr,
            "video",
        )
        audio_sigma_curr = _validate_timestep_sigma_pair(
            audio_timestep,
            audio_sigma_curr,
            "audio",
        )

        denoised_visual = minimax_h3_rf_v_to_x0(
            input_visual_latent,
            noise_pred_visual,
            visual_timestep,
        )
        denoised_audio = minimax_h3_rf_v_to_x0(
            input_audio_latent,
            noise_pred_audio,
            audio_timestep,
        )
        return {
            "output_visual_latent": minimax_h3_euler_eta0_step(
                input_visual_latent,
                denoised_visual,
                sigma_curr=visual_sigma_curr,
                sigma_next=visual_sigma_next,
            ),
            "output_audio_latent": minimax_h3_euler_eta0_step(
                input_audio_latent,
                denoised_audio,
                sigma_curr=audio_sigma_curr,
                sigma_next=audio_sigma_next,
            ),
        }


EntryClass = MiniMaxH3EulerAncestralEta0SchedulerAdapter

__all__ = [
    "MiniMaxH3EulerAncestralEta0SchedulerAdapter",
    "minimax_h3_euler_eta0_step",
    "minimax_h3_rf_v_to_x0",
]

```


---

## FILE: vllm_omni/diffusion/models/minimax_h3/continuation.py (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/diffusion/models/minimax_h3/continuation.py)

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded H3 windows with synchronized latent-tail guides and global AV RoPE positions.

The guide/discard/append algorithm follows ComfyUI-Minimax-H3-Continuation:
https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.long_video import resolve_long_video_mode

from .packed_tokens import minimax_h3_pack_audio_latent, minimax_h3_patchify_video_latent

logger = init_logger(__name__)


def _video_t(frames: int) -> int:
    return (frames - 5) // 17 * 5 + 2


def _audio_t(frame_boundary: int) -> int:
    return round(frame_boundary * 5 / 3)


@dataclass(frozen=True)
class ContinuationWindow:
    start: int
    end: int
    overlap: int

    @property
    def audio_start(self) -> int:
        return _audio_t(self.start)

    @property
    def audio_end(self) -> int:
        return _audio_t(self.end)

    @property
    def overlap_audio_t(self) -> int:
        return _audio_t(self.start + self.overlap) - self.audio_start


@dataclass(frozen=True)
class LatentPrefix:
    """Clean latents of an earlier clip that the request's first frames repeat.

    ``start_frame`` places the request on that clip's global frame timeline;
    ``video``/``audio`` are its last ``overlap_frames`` frames of latents. They
    ride as a frame-0 AV timeline guide for every task: it anchors the window to
    the clip far more tightly than a Ref2VA ``latent_guide`` reference.
    """

    video: torch.Tensor  # [1, C, _video_t(overlap_frames), H, W]
    audio: torch.Tensor  # [2, C, overlap audio latents]
    start_frame: int
    overlap_frames: int


# Later FL2VA windows no longer carry the first frame. With the Turbo 4-step adapter
# its look survives the join for windows up to 209 frames and is lost at 277.
_DEFAULT_WINDOW_FRAMES = {"fl2va": 141}


def resolve_continuation(extra: Mapping[str, Any], *, task: str, step_execution: bool) -> tuple[int, int] | None:
    mode = resolve_long_video_mode(extra, task)
    if mode == "full":
        return None
    if step_execution:
        raise OmniClientError("MiniMax H3 continuation requires request execution (not step execution)")
    window = extra.get("continuation_window_frames", _DEFAULT_WINDOW_FRAMES.get(task, 277))
    overlap = extra.get("continuation_overlap_frames", 22)
    for name, value in (("window", window), ("overlap", overlap)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 5 or value % 17 != 5:
            raise OmniClientError(f"MiniMax H3 continuation {name} must be an integer on the 17n+5 frame grid")
    if not 107 <= window <= 345 or overlap >= window:
        raise OmniClientError("MiniMax H3 continuation window must be 107..345 frames and exceed its overlap")
    return window, overlap


def continuation_uses_timeline_guides(task: str) -> bool:
    """Whether a task's windows carry the preceding tail as a frame-0 timeline guide.

    Ref2VA appends the tail as a ``latent_guide`` reference block instead. Keeping
    the choice here, rather than inline at the call site, keeps it under test.
    """
    return task != "ref2va"


def later_windows_drop_text_images(task: str) -> bool:
    """Whether a task's later windows need text encoded without the request's images.

    An FL2VA prompt is encoded together with its first-frame image. Later windows
    drop that keyframe, so they take the text-only presentation a ``continue_from``
    request without an image gets; text that still describes a picture the window
    does not carry breaks the join.
    """
    return task == "fl2va"


def plan_continuation_windows(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
    *,
    first_overlap: int = 0,
    origin: int = 0,
) -> list[ContinuationWindow]:
    """Windows over ``total_frames``, offset onto a global timeline by ``origin``.

    ``first_overlap`` hides the first frames of the first window behind a
    :class:`LatentPrefix`; later windows overlap their predecessor.
    """
    if total_frames < 5 or total_frames % 17 != 5:
        raise ValueError("total_frames must satisfy 17n+5")
    if not 5 <= overlap_frames < window_frames or window_frames % 17 != 5 or overlap_frames % 17 != 5:
        raise ValueError("window and overlap must satisfy 17n+5, with 5 <= overlap < window")
    end = min(total_frames, window_frames)
    if first_overlap and (first_overlap % 17 != 5 or first_overlap >= end):
        raise ValueError("first_overlap must satisfy 17n+5 and leave new frames in the first window")
    windows = [ContinuationWindow(origin, origin + end, first_overlap)]
    while end < total_frames:
        start = end - overlap_frames
        end = min(total_frames, start + window_frames)
        windows.append(ContinuationWindow(origin + start, origin + end, overlap_frames))
    return windows


def latent_guide_args(
    kwargs: Mapping[str, Any],
    video_tail: torch.Tensor,
    audio_tail: torch.Tensor,
    *,
    timeline_guide: bool = False,
) -> dict[str, Any]:
    """Add clean video/audio tail latents covering the window's first overlap frames.

    ``video_tail`` is ``[1, C, overlap_v, H, W]`` on the target canvas and
    ``audio_tail`` is ``[2, C, overlap_audio_t]``. Ref2VA appends them as the
    final ``latent_guide`` reference block; other tasks prepend a frame-0 AV
    timeline guide, which shares that origin and packs ahead of references.
    """
    overlap_v, overlap_audio_t = int(video_tail.shape[2]), int(audio_tail.shape[-1])
    if tuple(video_tail.shape[-2:]) != (kwargs["latent_h"], kwargs["latent_w"]):
        raise OmniClientError("MiniMax H3 latent guide must match the target canvas")
    guide_video = minimax_h3_patchify_video_latent(video_tail, patch_size=(1, 2, 2))
    guide_audio = minimax_h3_pack_audio_latent(audio_tail)
    shape = (overlap_v, kwargs["latent_h"], kwargs["latent_w"])
    if timeline_guide:
        block = {"kind": "video_audio", "frame_index": 0, "ref_audio_t": overlap_audio_t}
        return {
            "guide_blocks": [
                {**block, "latent_t": overlap_v, "latent_h": shape[1], "latent_w": shape[2]},
                *(kwargs.get("guide_blocks") or []),
            ],
            "visual_condition": _join(guide_video, kwargs.get("visual_condition"), first=True),
            "audio_condition": _join(guide_audio, kwargs.get("audio_condition"), first=True),
            "visual_condition_shapes": [shape, *(kwargs.get("visual_condition_shapes") or [])],
            "audio_condition_lengths": [overlap_audio_t, *(kwargs.get("audio_condition_lengths") or [])],
        }
    block = {"kind": "latent_guide", "ref_audio_t": overlap_audio_t}
    return {
        "ref_blocks": [
            *(kwargs.get("ref_blocks") or []),
            {**block, "latent_t": overlap_v, "latent_h": shape[1], "latent_w": shape[2]},
        ],
        "visual_condition": _join(guide_video, kwargs.get("visual_condition"), first=False),
        "audio_condition": _join(guide_audio, kwargs.get("audio_condition"), first=False),
        "visual_condition_shapes": [*(kwargs.get("visual_condition_shapes") or []), shape],
        "audio_condition_lengths": [*(kwargs.get("audio_condition_lengths") or []), overlap_audio_t],
    }


def _join(guide: torch.Tensor, existing: torch.Tensor | None, *, first: bool) -> torch.Tensor:
    if existing is None:
        return guide
    return torch.cat((guide, existing) if first else (existing, guide), dim=0)


# Clean conditions a T2VA/FL2VA request pins to frames of its first window: the
# FL2VA first frame, and that keyframe re-expressed as a guide behind a prefix.
_FIRST_WINDOW_CONDITION_KEYS = (
    "visual_condition",
    "visual_condition_shape",
    "visual_condition_shapes",
    "audio_condition",
    "ref_audio_t",
    "audio_condition_lengths",
    "keyframe_frame_indices",
    "guide_blocks",
    # FL2VA lists its keyframe image here as well; the block layout would pack it.
    "ref_blocks",
)


def _without_first_window_conditions(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {key: (None if key in _FIRST_WINDOW_CONDITION_KEYS else value) for key, value in kwargs.items()}


def window_args(
    kwargs: Mapping[str, Any],
    window: ContinuationWindow,
    *,
    video_tail: torch.Tensor | None = None,
    audio_tail: torch.Tensor | None = None,
    timeline_guide: bool = False,
    audio_origin: int = 0,
) -> dict[str, Any]:
    """Denoise arguments for one window: global AV clock, window lengths and any tail guide."""
    frames = window.end - window.start
    args = {
        **kwargs,
        "num_frames": frames,
        "latent_t": _video_t(frames),
        "audio_t": window.audio_end - window.audio_start,
        # Keep fractional RoPE units; audio slice indices alone are rounded.
        "temporal_offset": window.start * (40.0 / 24.0),
    }
    source = kwargs.get("locked_audio_rows")
    if source is not None:
        source = source.reshape(2, kwargs["audio_t"], 32)
        args["locked_audio_rows"] = source[
            :, window.audio_start - audio_origin : window.audio_end - audio_origin
        ].reshape(-1, 32)
    if video_tail is not None and audio_tail is not None:
        args.update(latent_guide_args(kwargs, video_tail, audio_tail, timeline_guide=timeline_guide))
    return args


def diffuse_continuation(
    diffuse: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    kwargs: dict[str, Any],
    *,
    window_frames: int,
    overlap_frames: int,
    text_conditioning: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    prefix: LatentPrefix | None = None,
    timeline_guides: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise fresh windows; retain old latents and append only new suffixes.

    Guides are extra condition rows sharing the new target's temporal origin,
    not a masked target prefix. Audio boundaries refer to the cumulative frame
    timeline, preventing per-window rounding from accumulating A/V drift.
    Each window shifts temporal media positions onto that same global timeline
    before RoPE is evaluated; text and static image references remain fixed.

    A ``prefix`` seeds the cumulative latents with an earlier clip's tail: the
    first window repeats it behind a hidden overlap, and the result starts with
    that tail, so callers trim ``prefix.overlap_frames`` after decoding.

    With ``timeline_guides`` (T2VA and FL2VA) every later window carries its
    predecessor's tail as a frame-0 AV timeline guide, as a ``prefix`` does,
    instead of a Ref2VA ``latent_guide`` reference. The request's own keyframe
    conditions belong to its first window and are dropped from the later ones,
    and per-window text keeps each window's own text origin.
    """
    origin = prefix.start_frame if prefix is not None else 0
    windows = plan_continuation_windows(
        kwargs["num_frames"],
        window_frames,
        overlap_frames,
        first_overlap=prefix.overlap_frames if prefix is not None else 0,
        origin=origin,
    )
    if text_conditioning is not None and len(text_conditioning) != len(windows):
        raise OmniClientError("MiniMax H3 requires one text conditioning per continuation window")
    # Different prompts have different prefix lengths. Keep the media clock
    # anchored after the longest prefix so text length cannot shift AV time.
    # Timeline-guide tasks sample their first window in the task's native
    # layout, where media follows that window's own text, so every window does
    # the same, as the requests of a ``continue_from`` chain do.
    shared_origin = bool(text_conditioning) and not timeline_guides
    media_time_origin = max(item[0].shape[0] for item in text_conditioning) if shared_origin else None
    video, audio = (prefix.video, prefix.audio) if prefix is not None else (None, None)
    for index, window in enumerate(windows):
        overlap_v = _video_t(window.overlap) if window.overlap else 0
        guided = video is not None and audio is not None
        args = window_args(
            _without_first_window_conditions(kwargs) if timeline_guides and index > 0 else kwargs,
            window,
            video_tail=video[:, :, -overlap_v:] if guided else None,
            audio_tail=audio[..., -window.overlap_audio_t :] if guided else None,
            timeline_guide=timeline_guides or (prefix is not None and index == 0),
            audio_origin=_audio_t(origin),
        )
        if text_conditioning is not None:
            args["text_embeddings"], args["text_tags"] = text_conditioning[index]
            if media_time_origin is not None:
                args["media_time_origin"] = media_time_origin
        logger.info(
            "MiniMax H3 continuation window %d/%d: frames [%d, %d), hidden overlap %d, temporal offset %.6f",
            index + 1,
            len(windows),
            window.start,
            window.end,
            window.overlap,
            args["temporal_offset"],
        )
        sampled_video, sampled_audio = diffuse(**args)
        if video is None or audio is None:
            video, audio = sampled_video, sampled_audio
        else:
            video = torch.cat((video, sampled_video[:, :, overlap_v:].to(video.device)), dim=2)
            audio = torch.cat((audio, sampled_audio[..., window.overlap_audio_t :].to(audio.device)), dim=-1)
    assert video is not None and audio is not None
    if video.shape[2] != kwargs["latent_t"] or audio.shape[-1] != kwargs["audio_t"]:
        raise RuntimeError("MiniMax H3 continuation produced inconsistent cumulative AV lengths")
    return video, audio

```


---

## FILE: vllm_omni/diffusion/models/minimax_h3/lora.py (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/diffusion/models/minimax_h3/lora.py)

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Load MiniMax-H3 Turbo and native H3 LoRA adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.layers.linear import LinearBase, MergedColumnParallelLinear, QKVParallelLinear
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.lora.request import LoRARequest

logger = init_logger(__name__)


_TURBO_RANK = 128
_TURBO_HIDDEN_SIZE = 5376
_TURBO_ATTENTION_INNER_SIZE = 7168
_TURBO_FFN_HIDDEN_SIZE = 14336
_TURBO_TIME_EMBED_DIM = 2688
# ``minimax_h3_<task>_turbo_<n>step_v<major.minor>[_768p][_bf16]``. The name is
# the only place the sampler contract is recorded.
_TURBO_NAME_RE = re.compile(
    r"^minimax_h3_(?P<task>fl2v|ref2v)_turbo_(?P<steps>\d+)step"
    r"_v\d+\.\d+(?P<res>_768p)?(?:_bf16)?\.safetensors$"
)
# The 768p retrains moved to a shorter video flow shift; the 544p artifacts keep
# the base model's.
_TURBO_VIDEO_SHIFT_768P = 6.0
_TURBO_VIDEO_SHIFT_544P = 12.0
# The name does not record the shift, and ``8step_v1.2`` breaks the rule above:
# LightX2V's reference configs moved from ``8step_v1.0_768p`` at shift 6 to this
# artifact at the base model's shift in one change
# (``configs/minimax_h3/distill/8steps`` in ModelTC/LightX2V).
_TURBO_V1_2_8STEP_FILENAME = "minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors"
_TURBO_VIDEO_SHIFT_BY_NAME = {
    _TURBO_V1_2_8STEP_FILENAME: 12.0,
}
_TURBO_AUDIO_SHIFT = 3.0
# Only ``4step_v0.1`` declares no alpha. LightX2V's reference script never reads
# the metadata: it applies ``scale * alpha / rank`` with ``--lora-alpha``
# defaulting to 8, and its documented v0.1 command does not override that. So an
# artifact that declares nothing is driven at 8, not at its rank.
_TURBO_DEFAULT_ALPHA = 8.0


@dataclass(frozen=True)
class TurboSpec:
    """The sampler contract one Turbo artifact was distilled for."""

    filename: str
    task_family: str
    """``fl2v`` (serves t2va and fl2va) or ``ref2v`` (serves ref2va)."""
    denoise_steps: int
    video_shift: float
    audio_shift: float
    rank: int
    alpha: float

    @property
    def sigma_points(self) -> int:
        """Sigma points the API contract expects: one more than the forwards."""
        return self.denoise_steps + 1

    @property
    def supported_tasks(self) -> frozenset[str]:
        if self.task_family == "ref2v":
            return frozenset({"ref2va"})
        return frozenset({"t2va", "fl2va"})


def parse_turbo_filename(name: str) -> TurboSpec | None:
    """Return the contract a Turbo filename encodes, or ``None``.

    ``alpha`` is the rank-matched default; the loader replaces it with the
    artifact's declared value.
    """

    match = _TURBO_NAME_RE.match(name)
    if match is None:
        return None
    steps = int(match.group("steps"))
    if steps <= 0:
        return None
    return TurboSpec(
        filename=name,
        task_family=match.group("task"),
        denoise_steps=steps,
        video_shift=_TURBO_VIDEO_SHIFT_BY_NAME.get(
            name, _TURBO_VIDEO_SHIFT_768P if match.group("res") else _TURBO_VIDEO_SHIFT_544P
        ),
        audio_shift=_TURBO_AUDIO_SHIFT,
        rank=_TURBO_RANK,
        alpha=_TURBO_DEFAULT_ALPHA,
    )


# ``pika_h3_flex_turbo_<n>step[_draft|_hd]_<task>[_lite_r32]``: Pika's Flex adapter fused with a few-step
# distill into one native-layout LoRA. The file carries no metadata, so the name is the contract here too.
_PIKA_TURBO_NAME_RE = re.compile(
    r"^pika_h3_flex_turbo_(?P<steps>\d+)step(?:_draft|_hd)?_(?P<task>fl2va|ref2va)(?:_lite_r\d+)?\.safetensors$"
)
# The release's sampling note: video shift 6 on the FL2VA files, 12 on the Ref2VA ones.
_PIKA_TURBO_VIDEO_SHIFT = {"fl2va": 6.0, "ref2va": 12.0}


def parse_pika_turbo_filename(name: str) -> TurboSpec | None:
    """Return the sampling contract a Pika Flex Turbo filename encodes, or ``None``.

    The adapter itself loads as an ordinary native LoRA, so ``rank`` and
    ``alpha`` are placeholders: every pair stores its own.
    """
    match = _PIKA_TURBO_NAME_RE.match(name)
    if match is None or int(match["steps"]) <= 0:
        return None
    return TurboSpec(
        filename=name,
        task_family="ref2v" if match["task"] == "ref2va" else "fl2v",
        denoise_steps=int(match["steps"]),
        video_shift=_PIKA_TURBO_VIDEO_SHIFT[match["task"]],
        audio_shift=_TURBO_AUDIO_SHIFT,
        rank=0,
        alpha=0.0,
    )


_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"
# ``8step_v1.2`` stores one scalar alpha per target next to the metadata value.
_ALPHA_SUFFIX = ".alpha"
_TURBO_TARGETS = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"})
_TURBO_RAW_TARGET_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_TURBO_TARGET_DIMS = {
    "attn.to_q": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_k": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_v": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_out.0": (_TURBO_ATTENTION_INNER_SIZE, _TURBO_HIDDEN_SIZE),
    "ff.net.0.proj": (_TURBO_HIDDEN_SIZE, 2 * _TURBO_FFN_HIDDEN_SIZE),
    "ff.net.2": (_TURBO_FFN_HIDDEN_SIZE, _TURBO_HIDDEN_SIZE),
}
# ``8step_v1.2`` also adapts the timestep modulation: every DiT block's AdaLN
# projection (three modalities, six H-wide vectors each) and the output norm's
# (shift and scale). Their rank is lower than the attention and MLP pairs' and
# is read from the artifact. LightX2V's own native-layout export of the file
# carries these factors unchanged under the native module names, so they need
# no row reordering.
_TURBO_ADALN_TARGET_DIMS = {
    "adaln_proj.linear": (_TURBO_TIME_EMBED_DIM, 18 * _TURBO_HIDDEN_SIZE),
    "norm_out.linear": (_TURBO_TIME_EMBED_DIM, 2 * _TURBO_HIDDEN_SIZE),
}
_TURBO_EXPECTED_RAW_TARGETS = frozenset(
    f"{prefix}.{block_index}.{suffix}"
    for prefix, block_count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    )
    for block_index in range(block_count)
    for suffix in _TURBO_RAW_TARGET_SUFFIXES
)
_TURBO_EXPECTED_ADALN_RAW_TARGETS = frozenset(
    [*(f"transformer_blocks.{block_index}.adaln_proj.linear" for block_index in range(50)), "norm_out.linear"]
)
_TURBO_TARGET_PATTERN = (
    r"^transformer\.(?:token_refiner\.blocks|blocks)\.\d+\."
    r"(?:attn\.(?:to_q|to_k|to_v|out_proj)|mlp\.(?:fc1|fc2))$"
)
# Only an artifact that carries the AdaLN pairs asks the manager to wrap those
# projections; every other artifact leaves them as plain layers.
_TURBO_ADALN_TARGET_PATTERN = (
    rf"(?:{_TURBO_TARGET_PATTERN}|^transformer\.(?:blocks\.\d+|final_layer)\.adaln_proj\.linear$)"
)

_TURBO_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "token_refiner.refiner_blocks.": "token_refiner.blocks.",
        "transformer_blocks.": "blocks.",
        "norm_out.linear.": "final_layer.adaln_proj.linear.",
        ".attn.to_out.0.": ".attn.out_proj.",
        ".ff.net.0.proj.": ".mlp.fc1.",
        ".ff.net.2.": ".mlp.fc2.",
        ".lora_A.default.": ".lora_A.",
        ".lora_B.default.": ".lora_B.",
    }
)


def _select_turbo_file(artifact_path: str | Path) -> Path | None:
    path = Path(artifact_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None

    candidates = sorted(child for child in path.glob("*.safetensors") if parse_turbo_filename(child.name))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{path} holds {len(candidates)} MiniMax-H3 Turbo artifacts "
            f"({[c.name for c in candidates[:4]]}); point --lora-path at one file."
        )
    return None


@dataclass(frozen=True)
class _TurboTensors:
    """A validated Turbo tensor set, renamed for the LoRA manager."""

    tensors: dict[str, torch.Tensor]
    alpha: float | None
    """Alpha the rank-128 pairs store per target; ``None`` when they store none."""
    adaln: dict[str, tuple[int, float]]
    """Native module name to ``(rank, alpha)`` for the AdaLN pairs, if any."""


def _read_turbo_alpha(checkpoint, name: str) -> float:
    tensor = checkpoint.get_tensor(name)
    if tensor.numel() != 1:
        raise ValueError(f"MiniMax-H3 Turbo alpha must be a scalar, got {name}={tuple(tensor.shape)}")
    alpha = float(tensor.float().item())
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"MiniMax-H3 Turbo alpha must be a positive number, got {name}={alpha!r}")
    return alpha


def _validate_and_convert_tensors(checkpoint, *, require_adaln: bool) -> _TurboTensors:
    """Validate a Diffusers-layout Turbo tensor set and pack it for the manager."""

    tensors: dict[str, torch.Tensor] = {}
    pairs: dict[str, set[str]] = {}
    raw_targets: set[str] = set()
    alphas: dict[str, float] = {}
    adaln_ranks: dict[str, dict[str, int]] = {}
    adaln_raw_targets: dict[str, str] = {}
    for name in checkpoint.keys():
        if name.endswith(_LORA_A_SUFFIX):
            raw_target = name[: -len(_LORA_A_SUFFIX)]
            side = "a"
        elif name.endswith(_LORA_B_SUFFIX):
            raw_target = name[: -len(_LORA_B_SUFFIX)]
            side = "b"
        elif name.endswith(_ALPHA_SUFFIX):
            alphas[name[: -len(_ALPHA_SUFFIX)]] = _read_turbo_alpha(checkpoint, name)
            continue
        else:
            raise ValueError(f"Unconsumed MiniMax-H3 Turbo tensor: {name!r}")
        raw_targets.add(raw_target)
        is_adaln = raw_target in _TURBO_EXPECTED_ADALN_RAW_TARGETS

        mapped_name = _TURBO_WEIGHTS_MAPPER.apply_list([name])[0]
        mapped_target = mapped_name.rsplit(".lora_", 1)[0]
        if not is_adaln and mapped_target.rsplit(".", 1)[-1] not in _TURBO_TARGETS:
            raise ValueError(f"Unsupported MiniMax-H3 Turbo target: {raw_target!r}")
        target_sides = pairs.setdefault(mapped_target, set())
        if side in target_sides:
            raise ValueError(f"Duplicate MiniMax-H3 Turbo tensor for {mapped_target}.{side}")
        target_sides.add(side)

        tensor = checkpoint.get_tensor(name)
        if tensor.ndim != 2:
            raise ValueError(f"MiniMax-H3 Turbo LoRA tensors must be matrices, got {name}={tuple(tensor.shape)}")
        if is_adaln:
            suffix = "norm_out.linear" if raw_target == "norm_out.linear" else "adaln_proj.linear"
            input_dim, output_dim = _TURBO_ADALN_TARGET_DIMS[suffix]
            # The pair declares its own rank. The manager's buffers are sized
            # for the attention and MLP pairs, so a rank outside 1..128 is
            # clamped here and then fails the shape check below.
            rank = min(max(tensor.shape[0 if side == "a" else 1], 1), _TURBO_RANK)
            adaln_ranks.setdefault(mapped_target, {})[side] = rank
            adaln_raw_targets[mapped_target] = raw_target
        else:
            suffix = next((suffix for suffix in _TURBO_RAW_TARGET_SUFFIXES if raw_target.endswith(suffix)), None)
            if suffix is None:
                raise ValueError(f"MiniMax-H3 Turbo LoRA contains unsupported target: {raw_target}")
            input_dim, output_dim = _TURBO_TARGET_DIMS[suffix]
            rank = _TURBO_RANK
        expected_shape = (rank, input_dim) if side == "a" else (output_dim, rank)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MiniMax-H3 Turbo tensor has invalid global shape: {name}={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )
        if side == "b" and ".ff.net.0.proj." in name:
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0).contiguous()
        # The LoRA manager re-parses raw tensor names without the weights mapper, so
        # the PEFT adapter-name segment must be dropped here: `parse_fine_tuned_lora_name`
        # rejects `...lora_A.<adapter>.weight` keys.
        tensors[f"{raw_target}.lora_{'A' if side == 'a' else 'B'}.weight"] = tensor

    incomplete = sorted(target for target, sides in pairs.items() if sides != {"a", "b"})
    if incomplete:
        raise ValueError(f"Incomplete MiniMax-H3 Turbo LoRA pairs: {incomplete}")
    # The published v1.2 file requires AdaLN. When present, the group must be complete.
    has_adaln = not raw_targets.isdisjoint(_TURBO_EXPECTED_ADALN_RAW_TARGETS)
    expected = _TURBO_EXPECTED_RAW_TARGETS
    if require_adaln or has_adaln:
        expected |= _TURBO_EXPECTED_ADALN_RAW_TARGETS
    missing = sorted(expected - raw_targets)
    unexpected = sorted(raw_targets - expected)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 Turbo target set does not match the published artifact layout: "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )
    mismatched = sorted(target for target, ranks in adaln_ranks.items() if ranks["a"] != ranks["b"])
    if mismatched:
        raise ValueError(f"MiniMax-H3 Turbo AdaLN pairs disagree on their rank: {mismatched[:5]}")

    # Per-target alphas are all-or-nothing too. Without them a rank-16 AdaLN
    # pair would be driven by the metadata alpha meant for the rank-128 pairs.
    if alphas and set(alphas) != raw_targets:
        without = sorted(raw_targets - set(alphas))
        stray = sorted(set(alphas) - raw_targets)
        raise ValueError(
            "MiniMax-H3 Turbo per-target alphas do not match the LoRA pairs: "
            f"missing={len(without)} {without[:5]}, unexpected={len(stray)} {stray[:5]}"
        )
    if has_adaln and not alphas:
        raise ValueError("MiniMax-H3 Turbo AdaLN pairs need per-target alpha tensors")
    alpha: float | None = None
    if alphas:
        shared = {alphas[target] for target in _TURBO_EXPECTED_RAW_TARGETS}
        if len(shared) != 1:
            raise ValueError(f"MiniMax-H3 Turbo rank-{_TURBO_RANK} pairs must share one alpha, got {sorted(shared)}")
        (alpha,) = shared
    adaln = {target: (ranks["a"], alphas[adaln_raw_targets[target]]) for target, ranks in adaln_ranks.items()}
    return _TurboTensors(tensors=tensors, alpha=alpha, adaln=adaln)


def _pack_h3_turbo_fc1(lora_model: LoRAModel) -> None:
    """Represent H3's fused gate/up projection without generic layout guesses."""

    for module_name, weights in tuple(lora_model.loras.items()):
        if not module_name.endswith(".mlp.fc1"):
            continue
        gate_b, up_b = weights.lora_b.chunk(2, dim=0)
        lora_model.loras[module_name] = PackedLoRALayerWeights(
            module_name=module_name,
            rank=weights.rank,
            lora_alphas=[weights.lora_alpha, weights.lora_alpha],
            lora_a=[weights.lora_a, weights.lora_a],
            lora_b=[gate_b.contiguous(), up_b.contiguous()],
            scaling=[weights.scaling, weights.scaling],
        )


def load_minimax_h3_turbo_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper, TurboSpec] | None:
    """Load any published LightX2V Turbo artifact through the legacy manager."""

    lora_file = _select_turbo_file(lora_path)
    if lora_file is None:
        return None
    spec = parse_turbo_filename(lora_file.name)
    if spec is None:
        # Not a Turbo artifact by name; leave it to the next loader.
        return None
    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        # Three published artifacts declare no key_format, so absence is not an
        # error; a declaration naming a different format is.
        metadata = checkpoint.metadata() or {}
        declared_format = metadata.get("key_format")
        if declared_format not in (None, "minimax-h3-diffusers"):
            raise ValueError(
                f"{lora_file.name} declares key_format={declared_format!r}, expected 'minimax-h3-diffusers'"
            )
        raw_alpha = metadata.get("alpha")
        if raw_alpha is not None:
            try:
                spec = replace(spec, alpha=float(raw_alpha))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"MiniMax-H3 Turbo alpha must be numeric, got {raw_alpha!r}") from exc
        if not math.isfinite(spec.alpha) or spec.alpha <= 0:
            raise ValueError(f"MiniMax-H3 Turbo alpha must be a positive number, got {raw_alpha!r}")

        if partition == "ref2va" and spec.task_family == "fl2v":
            raise ValueError(f"{spec.filename} is an FL2VA/T2VA Turbo artifact; a Ref2VA-only server cannot serve it.")
        # ``combined`` serves ref2va from ``transformers_ref``, but the LoRA
        # target pattern only injects into ``transformer``: the adapter would
        # bind to the stack that never runs.
        if spec.task_family == "ref2v" and partition != "ref2va":
            raise ValueError(
                f"{spec.filename} is a Ref2VA Turbo artifact; start the server with --task-type ref2va "
                f"(task_type={partition!r} serves ref2va from a DiT the adapter cannot bind to)"
            )
        if unsupported_offload_mode is not None:
            raise ValueError(f"MiniMax-H3 Turbo dynamic LoRA does not support {unsupported_offload_mode}")

        converted = _validate_and_convert_tensors(checkpoint, require_adaln=spec.filename == _TURBO_V1_2_8STEP_FILENAME)

    if converted.alpha is not None:
        # The artifact stores alpha per target. Metadata that names a different
        # value leaves no safe reading, so refuse instead of picking one.
        if raw_alpha is not None and converted.alpha != spec.alpha:
            raise ValueError(
                f"{spec.filename} declares alpha={spec.alpha:g} in its metadata but "
                f"alpha={converted.alpha:g} on its rank-{spec.rank} tensors"
            )
        spec = replace(spec, alpha=converted.alpha)
    elif raw_alpha is None:
        logger.warning(
            "MiniMax-H3 Turbo artifact %s declares no alpha; using the LightX2V reference "
            "default alpha=%g (scale %g). Override with the request-level LoRA scale.",
            spec.filename,
            spec.alpha,
            spec.alpha / spec.rank,
        )

    peft_helper = PEFTHelper.from_dict(
        {
            "r": spec.rank,
            "lora_alpha": spec.alpha,
            "target_modules": _TURBO_ADALN_TARGET_PATTERN if converted.adaln else _TURBO_TARGET_PATTERN,
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=converted.tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=_TURBO_WEIGHTS_MAPPER,
    )
    # ``from_lora_tensors`` gave every module the rank-128 pairs' rank and
    # scaling. The AdaLN pairs carry their own.
    for module_name, (rank, alpha) in converted.adaln.items():
        weights = lora_model.loras[module_name]
        weights.rank = rank
        weights.lora_alpha = alpha
        weights.scaling = alpha / rank
    _pack_h3_turbo_fc1(lora_model)
    logger.info(
        "MiniMax-H3 Turbo adapter %s: task=%s, %d denoiser forwards (%d sigma points), "
        "rank=%d alpha=%g, %d AdaLN pairs, flow_shift=%g/%g",
        spec.filename,
        spec.task_family,
        spec.denoise_steps,
        spec.sigma_points,
        spec.rank,
        spec.alpha,
        len(converted.adaln),
        spec.video_shift,
        spec.audio_shift,
    )
    return lora_model, peft_helper, spec


_NATIVE_LORA_KEY = re.compile(
    r"^(?P<module>.+)\.(?P<kind>lora_A|lora_B|lora_down|lora_up|alpha)"
    r"(?:\.default)?(?:\.weight)?$"
)


def load_minimax_h3_native_adapter(
    *,
    transformer: torch.nn.Module,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper] | None:
    """Load native H3 single-file LoRAs with ComfyUI scaling semantics."""
    path = Path(lora_path)
    if path.is_dir():
        if (path / "adapter_config.json").is_file():
            return None
        candidates = sorted(path.glob("*.safetensors"))
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError("Multiple H3 LoRA files found; select one .safetensors file.")
        path = candidates[0]
    if path.suffix != ".safetensors" or not path.is_file():
        return None

    targets: dict[str, tuple[int, tuple[int, ...]]] = {}
    for name, module in transformer.named_modules():
        if name.endswith(".base_layer"):
            continue
        if isinstance(module, BaseLayerWithLoRA):
            module = module.base_layer
        if isinstance(module, LinearBase):
            output_sizes = (
                tuple(module.output_sizes)
                if isinstance(module, (QKVParallelLinear, MergedColumnParallelLinear))
                else (module.output_size,)
            )
            targets[name] = (module.input_size, output_sizes)

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        if not any(_NATIVE_LORA_KEY.fullmatch(key) for key in keys):
            return None
        if unsupported_offload_mode is not None:
            raise ValueError(f"MiniMax-H3 native LoRA does not support {unsupported_offload_mode}")
        for key in keys:
            match = _NATIVE_LORA_KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"Unsupported H3 LoRA tensor: {key}")
            name = match["module"]
            for prefix in ("base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer."):
                name = name.removeprefix(prefix)
            if name not in targets:
                raise ValueError(f"H3 LoRA target does not exist in the model: {key}")
            kind = {"lora_A": "A", "lora_down": "A", "lora_B": "B", "lora_up": "B", "alpha": "alpha"}[match["kind"]]
            parts = grouped.setdefault(name, {})
            if kind in parts:
                raise ValueError(f"Duplicate H3 LoRA weight: {key}")
            parts[kind] = checkpoint.get_tensor(key)

    loras: dict[str, LoRALayerWeights] = {}
    for name, parts in grouped.items():
        if "A" not in parts or "B" not in parts:
            raise ValueError(f"H3 LoRA requires both A and B weights for {name}")
        a, b = parts["A"], parts["B"]
        input_size, output_sizes = targets[name]
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] < 1
            or b.shape != (sum(output_sizes), a.shape[0])
            or a.shape[1] != input_size
        ):
            raise ValueError(f"H3 LoRA shape mismatch for {name}: A={tuple(a.shape)}, B={tuple(b.shape)}")
        rank = a.shape[0]
        alpha_tensor = parts.get("alpha")
        if alpha_tensor is not None and alpha_tensor.numel() != 1:
            raise ValueError(f"H3 LoRA alpha must be scalar for {name}")
        alpha = rank if alpha_tensor is None else float(alpha_tensor.item())
        if not math.isfinite(alpha):
            raise ValueError(f"H3 LoRA alpha must be finite for {name}")
        a, b = a.to(dtype=dtype), b.to(dtype=dtype)
        if len(output_sizes) == 1:
            loras[name] = LoRALayerWeights(name, rank, alpha, a, b, scaling=alpha / rank)
        else:
            loras[name] = PackedLoRALayerWeights(
                module_name=name,
                rank=rank,
                lora_alphas=[alpha] * len(output_sizes),
                lora_a=[a] * len(output_sizes),
                lora_b=[part.contiguous() for part in b.split(output_sizes, dim=0)],
                scaling=[alpha / rank] * len(output_sizes),
            )
    rank = max(weights.rank for weights in loras.values())
    helper = PEFTHelper.from_dict({"r": rank, "lora_alpha": rank, "target_modules": list(loras)})
    return LoRAModel(lora_request.lora_int_id, rank, loras), helper

```


---

## FILE: vllm_omni/deploy/minimax_h3_disaggregated_turbo.yaml (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/deploy/minimax_h3_disaggregated_turbo.yaml)

```
# MiniMax H3 Turbo: five-sigma defaults for the disaggregated topology.
pipeline: minimax_h3_disaggregated
async_chunk: false
trust_remote_code: true
distributed_executor_backend: mp

connectors:
  shared_memory_connector:
    name: SharedMemoryConnector

stages:
  - stage_id: 0
    # Encoder is an LLM-AR stage. Diffusion stays on its own runner.
    model_runner: v2
    devices: "0,1"
    max_num_seqs: 1
    gpu_memory_utilization: 0.6
    tensor_parallel_size: 2
    output_connectors:
      to_stage_1: shared_memory_connector
    hf_overrides:
      minimax_h3_encoder_components:
        text_encoder:
          parallel_mode: tp
        video_vae:
          parallel_mode: patch
        audio_vae:
          parallel_mode: leader
    default_sampling_params:
      max_tokens: 1
      temperature: 0.0
      detokenize: false

  - stage_id: 1
    devices: "2,3,4,5"
    max_num_seqs: 1
    model_loaded:
      text_encoder: false
      vae_encoder: false
    parallel_config:
      tensor_parallel_size: 1
      ulysses_degree: 4
      vae_patch_parallel_size: 4
    input_connectors:
      from_stage_0: shared_memory_connector
    default_sampling_params:
      seed: 42
      num_inference_steps: 5
      extra_args:
        flow_shift: 6.0
        audio_flow_shift: 3.0

```


---

## FILE: vllm_omni/deploy/minimax_h3_disaggregated.yaml (https://github.com/vllm-project/vllm-omni/blob/HEAD/vllm_omni/deploy/minimax_h3_disaggregated.yaml)

```
# MiniMax H3: unified text/video/audio encoder stage + diffusion generation stage.
pipeline: minimax_h3_disaggregated
async_chunk: false
trust_remote_code: true
distributed_executor_backend: mp

connectors:
  shared_memory_connector:
    name: SharedMemoryConnector

stages:
  - stage_id: 0
    # Encoder is an LLM-AR stage. Diffusion stays on its own runner.
    model_runner: v2
    devices: "0,1"
    max_num_seqs: 1
    gpu_memory_utilization: 0.6
    tensor_parallel_size: 2
    output_connectors:
      to_stage_1: shared_memory_connector
    hf_overrides:
      minimax_h3_encoder_components:
        text_encoder:
          parallel_mode: tp
        video_vae:
          parallel_mode: patch
        audio_vae:
          parallel_mode: leader
    default_sampling_params:
      max_tokens: 1
      temperature: 0.0
      detokenize: false

  - stage_id: 1
    devices: "2,3,4,5"
    max_num_seqs: 1
    model_loaded:
      text_encoder: false
      vae_encoder: false
    parallel_config:
      tensor_parallel_size: 1
      ulysses_degree: 4
      vae_patch_parallel_size: 4
    input_connectors:
      from_stage_0: shared_memory_connector
    default_sampling_params:
      seed: 42
      num_inference_steps: 50

```
