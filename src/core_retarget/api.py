"""Thin public API backed by the same contracts used by CLI and web."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core_retarget.config.schema import RunConfig
from core_retarget.exceptions import (
    ConfigurationError,
    ModelVerificationError,
    MotionValidationError,
)
from core_retarget.motion.soma import (
    SomaMotionSummary,
    load_soma_motion,
    validate_soma_npz,
)
from core_retarget.motion.source import (
    SourceMotionSummary,
    load_source_motion,
    validate_source_motion,
)
from core_retarget.native import BackendSelection, resolve_backend
from core_retarget.pipeline.events import EventSink
from core_retarget.pipeline.runner import RetargetRunResult, run_retarget_pipeline
from core_retarget.robots.registry import get_robot
from core_retarget.robots.validation import ModelVerification, verify_robot
from core_retarget.stages.dmr import DmrProgress, DmrResult
from core_retarget.stages.dmr import run_dmr as run_dmr_stage
from core_retarget.stages.initial_collision import (
    InitialCollisionProgress,
    InitialCollisionResult,
)
from core_retarget.stages.initial_collision import (
    run_initial_collision as run_initial_collision_stage,
)
from core_retarget.stages.target_trajectories import (
    TargetTrajectoriesResult,
)
from core_retarget.stages.target_trajectories import (
    run_target_trajectories as run_target_trajectories_stage,
)


@dataclass(frozen=True)
class PreflightResult:
    motion: SomaMotionSummary | SourceMotionSummary
    model: ModelVerification


class Retargeter:
    """Stable facade for verified stages and the shared end-to-end runner."""

    def __init__(self, robot: str, config: RunConfig | None = None) -> None:
        self.robot = get_robot(robot)
        self.config = config or RunConfig(robot=self.robot.robot_id)
        if get_robot(self.config.robot).robot_id != self.robot.robot_id:
            raise ConfigurationError("Retargeter robot and RunConfig robot must match.")
        self._backend = resolve_backend(self.config.backend)

    @property
    def backend(self) -> BackendSelection:
        """Resolved compute backend shared by all stages run through this facade."""

        return self._backend

    def preflight(self, input_path: str | Path) -> PreflightResult:
        if Path(input_path).suffix.lower() == ".npz":
            motion: SomaMotionSummary | SourceMotionSummary = validate_soma_npz(
                input_path, fps_override=self.config.fps
            )
        else:
            motion = validate_source_motion(input_path, fps_override=self.config.fps)
        if motion.frame_count < 2:
            raise MotionValidationError(
                "The complete CoRe pipeline requires at least two motion frames."
            )
        model = verify_robot(self.robot, load_mujoco=True)
        if not model.ok:
            details = "; ".join(issue.message for issue in model.issues)
            raise ModelVerificationError(details)
        return PreflightResult(motion=motion, model=model)

    def run_dmr(
        self,
        input_path: str | Path,
        *,
        progress: DmrProgress | None = None,
    ) -> DmrResult:
        """Run the verified direct-retargeting stage for the selected robot.

        Every robot in the public registry has a validated DMR profile.
        Contact confidences are derived from the SOMA input by the contact
        preprocessing stage, which requires at least two frames even though
        general SOMA validation accepts a one-frame input.
        """

        if Path(input_path).suffix.lower() == ".npz":
            motion = load_soma_motion(input_path, fps_override=self.config.fps)
            return run_dmr_stage(
                motion,
                robot_id=self.robot.robot_id,
                progress=progress,
                backend=self._backend,
            )
        loaded_source = load_source_motion(input_path, fps_override=self.config.fps)
        motion = loaded_source.motion
        contacts = loaded_source.build_contact_schedule()
        return run_dmr_stage(
            motion,
            robot_id=self.robot.robot_id,
            progress=progress,
            backend=self._backend,
            source_provider=loaded_source.summary.provider,
            left_contact_confidence=contacts.left_confidence,
            right_contact_confidence=contacts.right_confidence,
        )

    def run_initial_collision(
        self,
        dmr_result: DmrResult,
        *,
        progress: InitialCollisionProgress | None = None,
    ) -> InitialCollisionResult:
        """Run the faithful first post-DMR collision refinement.

        Passing an explicit :class:`DmrResult` keeps the stage boundary visible
        to scripts and prevents an accidental second DMR solve.  The result
        must belong to the robot selected when this retargeter was created.
        """

        if dmr_result.robot_id != self.robot.robot_id:
            raise ConfigurationError(
                "DMR result robot and Retargeter robot must match "
                f"({dmr_result.robot_id!r} != {self.robot.robot_id!r})."
            )
        return run_initial_collision_stage(
            dmr_result.qpos,
            dmr_result.seconds,
            robot_id=self.robot.robot_id,
            fps=dmr_result.fps,
            progress=progress,
            backend=self._backend,
        )

    def run_target_trajectories(
        self,
        dmr_result: DmrResult,
        collision_result: InitialCollisionResult,
    ) -> TargetTrajectoriesResult:
        """Extract robot-space root, ankle, and toe targets for CoRe."""

        if dmr_result.robot_id != self.robot.robot_id:
            raise ConfigurationError(
                "DMR result robot and Retargeter robot must match "
                f"({dmr_result.robot_id!r} != {self.robot.robot_id!r})."
            )
        if collision_result.robot_id != self.robot.robot_id:
            raise ConfigurationError(
                "collision result robot and Retargeter robot must match "
                f"({collision_result.robot_id!r} != {self.robot.robot_id!r})."
            )
        if dmr_result.fps != collision_result.fps or not np.array_equal(
            dmr_result.seconds,
            collision_result.seconds,
        ):
            raise ConfigurationError("DMR and collision results must share one timeline.")
        return run_target_trajectories_stage(
            dmr_result.qpos,
            collision_result.qpos,
            dmr_result.seconds,
            robot_id=self.robot.robot_id,
            fps=dmr_result.fps,
        )

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        save_stages: bool = True,
        render_video: bool = False,
        render_thumbnail: bool = False,
        width: int = 1280,
        height: int = 720,
        event_sink: EventSink | None = None,
    ) -> RetargetRunResult:
        """Run the complete faithful pipeline and publish an unreviewed bundle."""

        return run_retarget_pipeline(
            input_path,
            self.robot.robot_id,
            output_dir,
            fps_override=self.config.fps,
            save_stages=save_stages,
            render_video=render_video,
            render_thumbnail=render_thumbnail,
            width=width,
            height=height,
            event_sink=event_sink,
            backend=self._backend,
        )
