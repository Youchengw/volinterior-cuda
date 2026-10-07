"""Trajectory-wide method selection; an unsampled leak never creates mixed-method output."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import platform
import shutil
import time
import numpy as np
import pandas as pd
from .directions import make_directions
from .measure import DEFINITIONS, measure_frame


def sample_frames(sequence, count):
    indices = np.rint(np.linspace(0, len(sequence) - 1, min(count, len(sequence)))).astype(int)
    return [sequence[i] for i in indices]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def run_trajectory(source, output, config, *, frames=None, method="auto", check_frames=9,
                   check_only=False, fallback=None, cutoff=None, save_labels=False, directions=None):
    output = Path(output)
    if method not in {"auto", *DEFINITIONS} or fallback not in {None, "fixed", "fuzzy"}:
        raise ValueError("invalid method/fallback")
    if fallback and method != "auto":
        raise ValueError("fallback applies only to auto")
    if cutoff is not None and method != "fuzzy" and fallback != "fuzzy":
        raise ValueError("cutoff applies only to fuzzy")
    if check_frames < 2:
        raise ValueError("check_frames must be at least two")
    if method == "fuzzy" or fallback == "fuzzy":
        from .measure import minimum_blocked
        minimum_blocked(cutoff, config.nrays)
    if check_only and method not in {"auto", "connectivity"}:
        raise ValueError("check_only is for connectivity screening")
    selected = list(source.frames) if frames is None else sorted(set(frames))
    if not selected or set(selected) - set(source.frames):
        raise ValueError("requested frame IDs are not present in the input")
    selected = [int(f) for f in selected]
    reference = int(source.frames[0])
    sequence = sorted(set(selected) | {reference})
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    np.save(output / "atom_radii_A.npy", source.radii, allow_pickle=False)
    np.save(output / "atom_masses_Da.npy", source.masses, allow_pickle=False)
    info = {"status": "running", "requested_method": method, "selected_method": None,
            "fallback_requested": fallback, "fallback_reason": None, "fallback_frame": None,
            "check_only": check_only, "requested_frames": selected, "reference_frame": reference,
            "source": source.provenance, "center": "selected-atom mass-weighted COM",
            "parameters": config.as_dict(), "fuzzy_cutoff": cutoff,
            "label_codes": {"exterior": 5, "interior": 0, "selection": -5},
            "label_axes": ["grid_x", "grid_y", "grid_z"],
            "software": {"python": platform.python_version(), "numpy": np.__version__}}
    if config.backend == "cuda":
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        info["software"].update(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                                gpu=name.decode() if isinstance(name, bytes) else name)

    def finish(status, checks=None):
        info["status"] = status
        info["elapsed_seconds"] = time.perf_counter() - started
        info["created_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "validation.json", {"status": "pass" if status in {"complete", "screen_passed"} else "incomplete",
                                                "checks": checks or {}})
        write_json(output / "metadata.json", info)
        return info

    def evaluate(frame, chosen, labels=False):
        result = measure_frame(source.read(frame), source.radii, source.masses, method=chosen,
                               config=config, cutoff=cutoff, directions=directions, return_labels=labels)
        result.counts.update(full_frame=frame, time_ns=source.time(frame), selected_for_output=frame in selected)
        print(f"frame {frame}: {chosen}, {result.counts['frame_seconds']:.3f} s", flush=True)
        return result

    cache = {}
    chosen = method
    try:
        if method in {"auto", "connectivity"}:
            samples = sample_frames(sequence, check_frames)
            print(f"Screening {len(samples)} uniformly spaced frames: {samples}", flush=True)
            for frame in samples:
                cache[frame] = evaluate(frame, "connectivity").counts
                pd.DataFrame(cache.values()).to_csv(output / "connectivity_check.csv", index=False)
            failed = [f for f, row in cache.items() if not row["closure_pass"]]
            screening = {"frames": samples, "times_ns": [source.time(f) for f in samples],
                         "all_samples_closed": not failed, "failed_frames": failed,
                         "sampling": "uniform positions in ordered frame list, endpoints included"}
            write_json(output / "screening.json", screening)
            info["screening"] = screening
            if failed:
                info.update(fallback_reason="sampled_frame_not_enclosed", fallback_frame=failed[0])
                chosen = fallback if method == "auto" else None
                info["selected_method"] = chosen
                if check_only and chosen is not None:
                    return finish("screen_failed")
                if chosen is None:
                    return finish("needs_dda_choice" if method == "auto" else "connectivity_failed")
            else:
                chosen = "connectivity"
                if check_only:
                    info["selected_method"] = chosen
                    return finish("screen_passed", {"sampled_frames_closed": True})
        while True:
            info["selected_method"] = chosen
            info["definition"] = DEFINITIONS[chosen]
            if chosen != "connectivity":
                if directions is None:
                    directions = make_directions(config.nrays, config.ray_scheme, config.ray_seed, config.ray_candidates)
                np.save(output / "ray_directions.npy", directions, allow_pickle=False)
            rows = []
            pending_labels = output / ".pending_labels"
            if save_labels:
                pending_labels.mkdir()
            partial = output / "counts.partial.csv"
            failure = None
            with partial.open("w") as handle:
                for frame in sequence:
                    if chosen == "connectivity" and frame in cache and not save_labels:
                        row, labels = cache[frame], None
                    else:
                        result = evaluate(frame, chosen, save_labels)
                        row, labels = result.counts, result.labels
                    pd.DataFrame([row]).to_csv(handle, header=not rows, index=False)
                    handle.flush()
                    rows.append(row)
                    if chosen == "connectivity" and not row["closure_pass"]:
                        failure = frame
                        break
                    if save_labels:
                        np.save(pending_labels / f"frame_{frame:06d}.npy", labels, allow_pickle=False)
            if failure is not None:
                os.replace(partial, output / "connectivity_attempt.csv")
                if pending_labels.exists():
                    shutil.rmtree(pending_labels)
                info.update(fallback_reason="unsampled_frame_not_enclosed", fallback_frame=failure)
                if method != "auto" or fallback is None:
                    info["selected_method"] = None
                    info.pop("definition", None)
                    return finish("needs_dda_choice" if method == "auto" else "connectivity_failed")
                chosen = fallback
                print(f"Frame {failure} leaked: restarting the entire requested trajectory with {chosen} DDA", flush=True)
                continue
            data = pd.DataFrame(rows)
            ref = data[data.full_frame == reference].iloc[0]
            plotted = []
            for row in rows:
                if not row["selected_for_output"]:
                    continue
                for compartment in ("lumen", "shell", "outer"):
                    value = row[compartment + "_volume_nm3"]
                    v0 = ref[compartment + "_volume_nm3"]
                    plotted.append({"full_frame": row["full_frame"], "time_ns": row["time_ns"],
                                    "method": chosen, "definition": DEFINITIONS[chosen], "compartment": compartment,
                                    "fuzzy_cutoff": cutoff if chosen == "fuzzy" else None,
                                    "volume_nm3": value, "volume_A3": value * 1000,
                                    "reference_volume_nm3": v0, "delta_volume_nm3": value - v0,
                                    "volume_change_percent": 100 * (value / v0 - 1) if v0 > 0 else None})
            checks = {"frame_coverage": data.full_frame.tolist() == sequence,
                      "single_method": data.method.nunique() == 1,
                      "voxel_conservation": bool((data.total_voxels == data.interior_voxels + data.selection_voxels + data.exterior_voxels).all()),
                      "outer_partition": bool((data.outer_voxels == data.lumen_voxels + data.shell_voxels).all())}
            if chosen == "connectivity":
                checks["every_frame_closed"] = bool(data.closure_pass.all())
            info["completed_output_frames"] = len(selected)
            if not all(checks.values()):
                raise RuntimeError("output invariants failed")
            ref.to_frame().T.to_csv(output / "reference.csv", index=False)
            pd.DataFrame(plotted).to_csv(output / "data.partial.csv", index=False)
            os.replace(partial, output / "counts.csv")
            if save_labels:
                pending_labels.rename(output / "labels")
            os.replace(output / "data.partial.csv", output / "data.csv")
            return finish("complete", checks)
    except Exception as exc:
        info["error"] = str(exc)
        finish("failed")
        raise
