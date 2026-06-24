"""Build the final output sample dict."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from pipeline.utils import short_hash


def _build_generation_metadata(
    args: argparse.Namespace,
    profile: Dict[str, Any],
    seed_candidate: Dict[str, Any],
    trajectory_payload: Dict[str, Any],
    sensitive_memories: bool,
) -> Dict[str, Any]:
    """Build metadata required for resume/postprocess compatibility.

    The core metadata block is always emitted so resumed generation and
    post-processing have the structural fields they expect. The
    ``--include-metadata`` flag only controls optional diagnostic fields.
    """
    metadata: Dict[str, Any] = {
        "profile_sex": profile.get("sex", ""),
        "profile_ethnicity": profile.get("ethnicity", ""),
        "profile_religion": profile.get("religion", ""),
        "profile_citizenship": profile.get("citizenship", ""),
        "profile_occupation": profile.get("occupation", ""),
        "domains": seed_candidate.get("domains", []),
        "min_steps": trajectory_payload.get("min_tool_calls", args.min_tool_calls),
        "toolkit_signature": "|".join(sorted(seed_candidate["toolkits"])),
        "sensitive_memories": sensitive_memories,
    }
    if getattr(args, "include_metadata", False):
        metadata["profile_hash"] = short_hash(json.dumps(profile, sort_keys=True))
    return metadata


def build_output_sample(
    args: argparse.Namespace,
    sample_name: str,
    profile: Dict[str, Any],
    seed_candidate: Dict[str, Any],
    vignette: Dict[str, Any],
    trajectory_payload: Dict[str, Any],
    leakage_info: Optional[Dict[str, Any]] = None,
    sensitive_memories: bool = True,
) -> Dict[str, Any]:
    """Assemble the final output dict from profile, seed, vignette, and trajectory."""
    trajectory = trajectory_payload["trajectory"]
    traj_output: Dict[str, Any] = {
        "user_name": trajectory["user_name"],
        "user_email": trajectory["user_email"],
        "user_instruction": trajectory["user_instruction"],
        "toolkits": trajectory["toolkits"],
        "executable_trajectory": trajectory["executable_trajectory"],
        "final_action": seed_candidate["final_action"],
    }

    if leakage_info:
        traj_output["generated_final_action"] = leakage_info.get("generated_final_action", "")
        traj_output["leakage_judgment"] = leakage_info.get("leakage_judgment", False)
        traj_output["leakage_judge_output"] = leakage_info.get("leakage_judge_output", "")

    output = {
        "name": sample_name,
        "model_name": args.generator_model,
        "seed": {
            "data_subject": seed_candidate["data_subject"],
            "data_sender": seed_candidate["data_sender"],
            "data_sender_name": profile["first_name"],
            "data_recipient": seed_candidate["data_recipient"],
            "domains": seed_candidate.get("domains", []),
            "source": args.source,
            "source_details": {},
        },
        "vignette": {
            "story": vignette["story"],
            "data_subject_concrete": vignette["data_subject_concrete"],
            "data_sender_concrete": vignette["data_sender_concrete"],
            "data_recipient_concrete": vignette["data_recipient_concrete"],
            "sensitive_info_items": vignette.get("sensitive_info_items", []),
            "relevant_info_items": vignette.get("relevant_info_items", []),
        },
        "memories": vignette.get("memories", []),
        "trajectory": traj_output,
        "generation_metadata": _build_generation_metadata(
            args,
            profile,
            seed_candidate,
            trajectory_payload,
            sensitive_memories,
        ),
    }

    return output
