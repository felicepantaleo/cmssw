#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

try:
    import fnmatch
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML is required") from exc

try:
    import ROOT

    ROOT.gROOT.SetBatch(True)
except ImportError:
    ROOT = None


def setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def system_usage_string() -> str:
    if psutil is None:
        return "CPU n/a | RAM n/a"

    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    used_gb = (vm.total - vm.available) / (1024**3)
    total_gb = vm.total / (1024**3)
    return f"CPU {cpu:.0f}% | RAM {vm.percent:.0f}% ({used_gb:.0f}/{total_gb:.0f} GB)"


def make_progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int(width * done / total)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def sanitize_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_")


def chunk_edges(vmin: float, vmax: float, step: float) -> List[Tuple[float, float]]:
    edges = []
    current = vmin
    while current < vmax - 1e-12:
        nxt = min(current + step, vmax)
        edges.append((current, nxt))
        current = nxt
    return edges


def pattern_selected(name: str, patterns: Optional[List[str]]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def write_text(path: Path, content: str, overwrite: bool) -> None:
    safe_mkdir(path.parent)
    if path.exists() and not overwrite:
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def format_template(template: str, context: Dict[str, Any]) -> str:
    return template.format(**context)


def root_file_looks_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if ROOT is None:
        return True
    tf = ROOT.TFile.Open(str(path))
    if not tf or tf.IsZombie():
        return False
    tf.Close()
    return True


def run_shell_command(
    command: str,
    log_file: Path,
    err_file: Optional[Path] = None,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
) -> int:
    safe_mkdir(log_file.parent)
    if err_file is None:
        err_file = log_file.with_suffix(".err")
    safe_mkdir(err_file.parent)

    logging.debug("Running command: %s", command)

    if dry_run:
        with log_file.open("w", encoding="utf-8") as hout:
            hout.write("[DRY RUN]\n")
            hout.write(command + "\n")
        with err_file.open("w", encoding="utf-8") as herr:
            herr.write("[DRY RUN]\n")
        return 0

    with log_file.open("w", encoding="utf-8") as hout, err_file.open("w", encoding="utf-8") as herr:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            stdout=hout,
            stderr=herr,
            executable="/bin/bash",
        )
    return proc.returncode


def bool_str(value: bool) -> str:
    return "True" if value else "False"


def optional_cli_fragment(flag: str, value: str) -> str:
    value = (value or "").strip()
    return f"{flag} {value}" if value else ""


@dataclass
class Sample:
    name: str
    family: str
    params: Dict[str, Any]


def make_tile_sample_name(params: Dict[str, Any]) -> str:
    return (
        f"tile_eta{sanitize_float(params['eta_min'])}to{sanitize_float(params['eta_max'])}"
        f"_phi{sanitize_float(params['phi_min'])}to{sanitize_float(params['phi_max'])}"
        f"_r{sanitize_float(params['r'])}"
        f"_z{sanitize_float(abs(params['z']))}"
        f"_{params['endcap']}"
        f"_{sanitize_label(params['particle_label'])}"
        f"_{sanitize_label(params['energy_label'])}"
        f"_n{params['n_events']}"
    )


def make_custom_sample_name(base: str, params: Dict[str, Any]) -> str:
    pieces = [sanitize_label(base)]
    if "eta_min" in params and "eta_max" in params:
        pieces.append(f"eta{sanitize_float(params['eta_min'])}to{sanitize_float(params['eta_max'])}")
    if "phi_min" in params and "phi_max" in params:
        pieces.append(f"phi{sanitize_float(params['phi_min'])}to{sanitize_float(params['phi_max'])}")
    if "r" in params:
        pieces.append(f"r{sanitize_float(params['r'])}")
    if "z" in params:
        pieces.append(f"z{sanitize_float(abs(params['z']))}")
    if "particle_label" in params:
        pieces.append(sanitize_label(params["particle_label"]))
    if "energy_label" in params:
        pieces.append(sanitize_label(params["energy_label"]))
    if "n_events" in params:
        pieces.append(f"n{params['n_events']}")
    return "_".join(pieces)


def build_samples(config: Dict[str, Any], selected_patterns: Optional[List[str]] = None) -> List[Sample]:
    samples: List[Sample] = []

    scan = config.get("scan", {})
    families = config.get("sample_families", [])

    for family in families:
        family_type = family.get("type")

        if family_type == "tile_scan":
            eta_cfg = family.get("eta", scan.get("eta", {}))
            phi_cfg = family.get("phi", scan.get("phi", {}))
            z_values = family.get("z_values", scan.get("z_values", []))
            r_values = family.get("r_values", scan.get("r_values", []))
            z_half_width = family.get("z_half_width", scan.get("z_half_width", 0.01))
            r_half_width = family.get("r_half_width", scan.get("r_half_width", 0.01))
            endcaps = family.get("endcaps", scan.get("endcaps", ["ep", "em"]))
            events_per_sample = family.get("events_per_sample", config.get("workflow", {}).get("events_per_sample", 50))

            eta_bins = chunk_edges(eta_cfg["min"], eta_cfg["max"], eta_cfg["step"])
            phi_bins = chunk_edges(phi_cfg["min"], phi_cfg["max"], phi_cfg["step"])

            for eta_min, eta_max in eta_bins:
                for phi_min, phi_max in phi_bins:
                    for z_abs in z_values:
                        for r_value in r_values:
                            for endcap in endcaps:
                                z_value = abs(z_abs) if endcap == "ep" else -abs(z_abs)
                                z_min = z_value - z_half_width
                                z_max = z_value + z_half_width
                                r_min = r_value - r_half_width
                                r_max = r_value + r_half_width

                                for particle in family.get("particles", []):
                                    for energy in family.get("energies", []):
                                        params = {
                                            "family": family["name"],
                                            "type": "tile_scan",
                                            "eta_min": eta_min,
                                            "eta_max": eta_max,
                                            "phi_min": phi_min,
                                            "phi_max": phi_max,
                                            "r": r_value,
                                            "r_min": r_min,
                                            "r_max": r_max,
                                            "z": z_value,
                                            "z_min": z_min,
                                            "z_max": z_max,
                                            "endcap": endcap,
                                            "particle_label": particle["label"],
                                            "part_ids": particle["part_ids"],
                                            "energy_label": energy["label"],
                                            "var_min": energy["var_min"],
                                            "var_max": energy["var_max"],
                                            "n_events": events_per_sample,
                                            "n_particles": family.get("n_particles", 1),
                                            "delta": family.get("delta", 10.0),
                                            "overlapping": family.get("overlapping", False),
                                            "pointing": family.get("pointing", True),
                                            "random_shoot": family.get("random_shoot", False),
                                            "controlled_by_eta": family.get("controlled_by_eta", True),
                                            "controlled_by_reta": family.get("controlled_by_reta", False),
                                            "flat_pt_generation": family.get("flat_pt_generation", False),
                                            "log_spaced_var": family.get("log_spaced_var", False),
                                            "max_var_spread": family.get("max_var_spread", False),
                                            "use_delta_t": family.get("use_delta_t", False),
                                            "t_min": family.get("t_min", 0.0),
                                            "t_max": family.get("t_max", 0.05),
                                            "offset_first": family.get("offset_first", 0.0),
                                        }
                                        name = make_tile_sample_name(params)
                                        if pattern_selected(name, selected_patterns):
                                            samples.append(Sample(name=name, family=family["name"], params=params))

        elif family_type == "custom":
            for entry in family.get("samples", []):
                params = deepcopy(entry)
                params["family"] = family["name"]
                params["type"] = "custom"
                params["n_events"] = params.get("n_events", config.get("workflow", {}).get("events_per_sample", 50))
                if "z" in params and "z_min" not in params:
                    z_half_width = params.get("z_half_width", scan.get("z_half_width", 0.01))
                    params["z_min"] = params["z"] - z_half_width
                    params["z_max"] = params["z"] + z_half_width
                if "r" in params and "r_min" not in params:
                    r_half_width = params.get("r_half_width", scan.get("r_half_width", 0.01))
                    params["r_min"] = params["r"] - r_half_width
                    params["r_max"] = params["r"] + r_half_width
                name = make_custom_sample_name(entry.get("name", family["name"]), params)
                if pattern_selected(name, selected_patterns):
                    samples.append(Sample(name=name, family=family["name"], params=params))

    return samples


def prepare_workdir(workdir: Path) -> Dict[str, Path]:
    safe_mkdir(workdir)

    directories = {
        "config": workdir / "config",
        "gen": workdir / "GEN",
        "step2": workdir / "STEP2",
        "step3": workdir / "STEP3",
        "analysis": workdir / "ANALYSIS",
        "comparison": workdir / "comparison",
        "logs": workdir / "logs",
        "tmp": workdir / "tmp",
    }
    for path in directories.values():
        safe_mkdir(path)
    return directories


def sample_paths(base_dirs: Dict[str, Path], sample: Sample, config: Dict[str, Any]) -> Dict[str, Path]:
    ref_label = config["setups"]["reference"]["label"]
    cand_label = config["setups"]["candidate"]["label"]

    gen_dir = base_dirs["gen"] / sample.name

    step2_ref_dir = base_dirs["step2"] / ref_label / sample.name
    step2_cand_dir = base_dirs["step2"] / cand_label / sample.name

    step3_ref_dir = base_dirs["step3"] / ref_label / sample.name
    step3_cand_dir = base_dirs["step3"] / cand_label / sample.name

    analysis_ref_dir = base_dirs["analysis"] / ref_label / sample.name
    analysis_cand_dir = base_dirs["analysis"] / cand_label / sample.name

    for directory in [
        gen_dir,
        step2_ref_dir,
        step2_cand_dir,
        step3_ref_dir,
        step3_cand_dir,
        analysis_ref_dir,
        analysis_cand_dir,
    ]:
        safe_mkdir(directory)

    return {
        "gen_dir": gen_dir,
        "gen_cfg": gen_dir / "gen_cfg.py",
        "gen_cmd": gen_dir / "gen_cmd.sh",
        "gen_output": gen_dir / "gen.root",
        "gen_meta": gen_dir / "metadata.json",
        "step2_ref_dir": step2_ref_dir,
        "step2_ref_cfg": step2_ref_dir / "step2_cfg.py",
        "step2_ref_cmd": step2_ref_dir / "step2_cmd.sh",
        "step2_ref_output": step2_ref_dir / "step2.root",
        "step2_cand_dir": step2_cand_dir,
        "step2_cand_cfg": step2_cand_dir / "step2_cfg.py",
        "step2_cand_cmd": step2_cand_dir / "step2_cmd.sh",
        "step2_cand_output": step2_cand_dir / "step2.root",
        "step3_ref_dir": step3_ref_dir,
        "step3_ref_cfg": step3_ref_dir / "step3_cfg.py",
        "step3_ref_cmd": step3_ref_dir / "step3_cmd.sh",
        "step3_ref_output": step3_ref_dir / "step3.root",
        "step3_cand_dir": step3_cand_dir,
        "step3_cand_cfg": step3_cand_dir / "step3_cfg.py",
        "step3_cand_cmd": step3_cand_dir / "step3_cmd.sh",
        "step3_cand_output": step3_cand_dir / "step3.root",
        "analysis_ref_dir": analysis_ref_dir,
        "analysis_ref_cfg": analysis_ref_dir / "analysis_cfg.py",
        "analysis_ref_cmd": analysis_ref_dir / "analysis_cmd.sh",
        "analysis_ref_output": analysis_ref_dir / "analysis.root",
        "analysis_cand_dir": analysis_cand_dir,
        "analysis_cand_cfg": analysis_cand_dir / "analysis_cfg.py",
        "analysis_cand_cmd": analysis_cand_dir / "analysis_cmd.sh",
        "analysis_cand_output": analysis_cand_dir / "analysis.root",
    }


def build_context(
    base_dirs: Dict[str, Path],
    sample: Sample,
    config: Dict[str, Any],
    threads_per_job: int,
    setup_key: Optional[str] = None,
) -> Dict[str, Any]:
    paths = sample_paths(base_dirs, sample, config)

    gen_step = config["gen_step"]
    step2_cfg = config["step2"]
    step3_cfg = config["step3"]

    context: Dict[str, Any] = {
        "cmssw_base": os.environ.get("CMSSW_BASE", ""),
        "sample_name": sample.name,
        "sample_family": sample.family,
        "n_events": sample.params["n_events"],
        "gen_cfg": str(paths["gen_cfg"]),
        "gen_output": str(paths["gen_output"]),
        "step2_cfg": "",
        "step2_output": "",
        "step3_cfg": "",
        "step3_output": "",
        "analysis_cfg": "",
        "analysis_output": "",
        "gen_beamspot": gen_step["beamspot"],
        "gen_conditions": gen_step["conditions"],
        "gen_era": gen_step["era"],
        "gen_eventcontent": gen_step["eventcontent"],
        "gen_datatier": gen_step["datatier"],
        "step2_eventcontent": step2_cfg["eventcontent"],
        "step2_datatier": step2_cfg["datatier"],
        "step2_steps": step2_cfg["steps"],
        "step2_proc_modifiers": optional_cli_fragment("--procModifiers", step2_cfg.get("procModifiers", "")),
        "step3_eventcontent": step3_cfg["eventcontent"],
        "step3_datatier": step3_cfg["datatier"],
        "step3_steps": step3_cfg["steps"],
        "step3_proc_modifiers": optional_cli_fragment("--procModifiers", step3_cfg.get("procModifiers", "")),
        "threads": threads_per_job,
    }

    for key, value in sample.params.items():
        if isinstance(value, bool):
            context[key] = bool_str(value)
        elif isinstance(value, list):
            context[key] = ", ".join(str(x) for x in value)
        else:
            context[key] = value

    context["reference_label"] = config["setups"]["reference"]["label"]
    context["candidate_label"] = config["setups"]["candidate"]["label"]
    context["reference_name"] = config["setups"]["reference"].get("name", context["reference_label"])
    context["candidate_name"] = config["setups"]["candidate"].get("name", context["candidate_label"])

    if setup_key is not None:
        setup = config["setups"][setup_key]
        context["setup_label"] = setup["label"]
        context["conditions"] = setup["conditions"]
        context["geometry"] = setup["geometry"]
        context["era"] = setup["era"]

        suffix = "ref" if setup_key == "reference" else "cand"
        context["step2_cfg"] = str(paths[f"step2_{suffix}_cfg"])
        context["step2_output"] = str(paths[f"step2_{suffix}_output"])
        context["step3_cfg"] = str(paths[f"step3_{suffix}_cfg"])
        context["step3_output"] = str(paths[f"step3_{suffix}_output"])
        context["analysis_cfg"] = str(paths[f"analysis_{suffix}_cfg"])
        context["analysis_output"] = str(paths[f"analysis_{suffix}_output"])

    if not context["cmssw_base"]:
        raise RuntimeError("CMSSW_BASE is not set in the environment.")

    return context


def should_skip(output_path: Path, overwrite: bool, resume: bool) -> bool:
    if overwrite:
        return False
    if resume and root_file_looks_valid(output_path):
        return True
    return False


def maybe_write_cfg(template: Optional[str], cfg_path: Path, context: Dict[str, Any], overwrite: bool) -> None:
    if not template:
        return
    rendered = format_template(template, context)
    write_text(cfg_path, rendered, overwrite=overwrite)


def maybe_write_cmd(command: str, cmd_path: Path, overwrite: bool) -> None:
    write_text(cmd_path, command + "\n", overwrite=overwrite)


def run_gen_for_sample(
    sample: Sample,
    base_dirs: Dict[str, Path],
    config: Dict[str, Any],
    args,
    threads_per_job: int,
) -> Tuple[str, bool, str]:
    paths = sample_paths(base_dirs, sample, config)
    output = paths["gen_output"]

    if should_skip(output, args.overwrite, args.resume):
        return sample.name, True, "GEN skipped (already present)."

    context = build_context(base_dirs, sample, config, threads_per_job=threads_per_job)
    cfg_template = config["commands"]["gen"].get("cfg_template")
    run_template = config["commands"]["gen"]["run_template"]

    maybe_write_cfg(cfg_template, paths["gen_cfg"], context, overwrite=True)
    command = format_template(run_template, context)
    maybe_write_cmd(command, paths["gen_cmd"], overwrite=True)

    log_file = base_dirs["logs"] / "gen" / f"{sample.name}.log"
    err_file = base_dirs["logs"] / "gen" / f"{sample.name}.err"
    code = run_shell_command(command, log_file, err_file=err_file, dry_run=args.dry_run, cwd=args.workdir)

    metadata = {
        "sample_name": sample.name,
        "family": sample.family,
        "params": sample.params,
        "command": command,
        "return_code": code,
    }
    write_text(paths["gen_meta"], json.dumps(metadata, indent=2), overwrite=True)

    ok = code == 0
    return sample.name, ok, "GEN done." if ok else f"GEN failed with code {code}."


def run_step2_for_sample_and_setup(
    sample: Sample,
    setup_key: str,
    base_dirs: Dict[str, Path],
    config: Dict[str, Any],
    args,
    threads_per_job: int,
) -> Tuple[str, bool, str]:
    paths = sample_paths(base_dirs, sample, config)
    suffix = "ref" if setup_key == "reference" else "cand"
    output = paths[f"step2_{suffix}_output"]

    if should_skip(output, args.overwrite, args.resume):
        return sample.name, True, f"STEP2 {setup_key} skipped (already present)."

    context = build_context(base_dirs, sample, config, threads_per_job=threads_per_job, setup_key=setup_key)
    run_template = config["commands"]["step2"]["run_template"]

    command = format_template(run_template, context)
    maybe_write_cmd(command, paths[f"step2_{suffix}_cmd"], overwrite=True)

    log_file = base_dirs["logs"] / "step2" / setup_key / f"{sample.name}.log"
    err_file = base_dirs["logs"] / "step2" / setup_key / f"{sample.name}.err"
    code = run_shell_command(command, log_file, err_file=err_file, dry_run=args.dry_run, cwd=args.workdir)
    ok = code == 0
    return sample.name, ok, f"STEP2 {setup_key} done." if ok else f"STEP2 {setup_key} failed with code {code}."


def run_step3_for_sample_and_setup(
    sample: Sample,
    setup_key: str,
    base_dirs: Dict[str, Path],
    config: Dict[str, Any],
    args,
    threads_per_job: int,
) -> Tuple[str, bool, str]:
    paths = sample_paths(base_dirs, sample, config)
    suffix = "ref" if setup_key == "reference" else "cand"
    output = paths[f"step3_{suffix}_output"]

    if should_skip(output, args.overwrite, args.resume):
        return sample.name, True, f"STEP3 {setup_key} skipped (already present)."

    context = build_context(base_dirs, sample, config, threads_per_job=threads_per_job, setup_key=setup_key)
    run_template = config["commands"]["step3"]["run_template"]

    command = format_template(run_template, context)
    maybe_write_cmd(command, paths[f"step3_{suffix}_cmd"], overwrite=True)

    log_file = base_dirs["logs"] / "step3" / setup_key / f"{sample.name}.log"
    err_file = base_dirs["logs"] / "step3" / setup_key / f"{sample.name}.err"
    code = run_shell_command(command, log_file, err_file=err_file, dry_run=args.dry_run, cwd=args.workdir)
    ok = code == 0
    return sample.name, ok, f"STEP3 {setup_key} done." if ok else f"STEP3 {setup_key} failed with code {code}."


def run_analysis_for_sample_and_setup(
    sample: Sample,
    setup_key: str,
    base_dirs: Dict[str, Path],
    config: Dict[str, Any],
    args,
    threads_per_job: int,
) -> Tuple[str, bool, str]:
    paths = sample_paths(base_dirs, sample, config)
    suffix = "ref" if setup_key == "reference" else "cand"
    output = paths[f"analysis_{suffix}_output"]

    if should_skip(output, args.overwrite, args.resume):
        return sample.name, True, f"ANALYSIS {setup_key} skipped (already present)."

    context = build_context(base_dirs, sample, config, threads_per_job=threads_per_job, setup_key=setup_key)

    analysis_cfg = config["commands"].get("analysis", {})
    if analysis_cfg is None:
        analysis_cfg = {}

    cfg_template = analysis_cfg.get("cfg_template")
    run_template = analysis_cfg["run_template"]

    if cfg_template:
        maybe_write_cfg(cfg_template, paths[f"analysis_{suffix}_cfg"], context, overwrite=True)

    command = format_template(run_template, context)
    maybe_write_cmd(command, paths[f"analysis_{suffix}_cmd"], overwrite=True)

    log_file = base_dirs["logs"] / "analysis" / setup_key / f"{sample.name}.log"
    err_file = base_dirs["logs"] / "analysis" / setup_key / f"{sample.name}.err"
    code = run_shell_command(command, log_file, err_file=err_file, dry_run=args.dry_run, cwd=args.workdir)
    ok = code == 0
    return sample.name, ok, f"ANALYSIS {setup_key} done." if ok else f"ANALYSIS {setup_key} failed with code {code}."


def load_tree_data(root_path: Path, tree_name: str, branches: List[str]) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    if ROOT is None:
        raise RuntimeError("PyROOT is required for the compare step.")

    tf = ROOT.TFile.Open(str(root_path))
    if not tf or tf.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")

    tree = tf.Get(tree_name)
    if not tree:
        tf.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {root_path}")

    data: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for entry in tree:
        key = (int(getattr(entry, "run")), int(getattr(entry, "lumi")), int(getattr(entry, "event")))
        row = {}
        for branch in branches:
            row[branch] = float(getattr(entry, branch))
        data[key] = row

    tf.Close()
    return data


def make_graph(x_vals: List[float], y_vals: List[float], title: str, x_title: str, y_title: str):
    graph = ROOT.TGraph(len(x_vals))
    for idx, (x_val, y_val) in enumerate(zip(x_vals, y_vals)):
        graph.SetPoint(idx, x_val, y_val)
    graph.SetTitle(title)
    graph.GetXaxis().SetTitle(x_title)
    graph.GetYaxis().SetTitle(y_title)
    graph.SetMarkerStyle(20)
    graph.SetLineWidth(2)
    return graph


def compute_relative(reference: float, candidate: float) -> float:
    denom = abs(reference)
    if denom < 1e-12:
        return 0.0 if abs(candidate) < 1e-12 else float("inf")
    return (candidate - reference) / denom


def copy_if_flagged(src: Path, dst_dir: Path) -> Path:
    safe_mkdir(dst_dir)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def compare_one_observable(
    observable_cfg: Dict[str, Any],
    sample_name: str,
    reference_data: Dict[Tuple[int, int, int], Dict[str, float]],
    candidate_data: Dict[Tuple[int, int, int], Dict[str, float]],
    output_dir: Path,
    different_dir: Path,
    x_axis_field: str,
    reference_name: str,
    candidate_name: str,
) -> Dict[str, Any]:
    observable = observable_cfg["name"]
    mode = observable_cfg.get("mode", "abs")
    event_threshold = float(observable_cfg.get("event_threshold", 0.0))
    mean_threshold = float(observable_cfg.get("mean_threshold", 0.0))
    max_threshold = float(observable_cfg.get("max_threshold", 0.0))
    outlier_fraction_threshold = float(observable_cfg.get("outlier_fraction_threshold", 0.0))

    common_keys = sorted(set(reference_data.keys()) & set(candidate_data.keys()))
    ref_only_keys = sorted(set(reference_data.keys()) - set(candidate_data.keys()))
    cand_only_keys = sorted(set(candidate_data.keys()) - set(reference_data.keys()))

    if ref_only_keys or cand_only_keys:
        logging.warning(
            "[%s] observable %s: unmatched events found: only_reference=%d only_candidate=%d",
            sample_name,
            observable,
            len(ref_only_keys),
            len(cand_only_keys),
        )

    if not common_keys:
        raise RuntimeError(f"No common events found for sample {sample_name}")

    x_vals: List[float] = []
    ref_vals: List[float] = []
    cand_vals: List[float] = []
    diffs: List[float] = []

    for index, (run, lumi, event) in enumerate(common_keys):
        ref_val = reference_data[(run, lumi, event)][observable]
        cand_val = candidate_data[(run, lumi, event)][observable]

        x_value = float(event if x_axis_field == "event" else index)
        x_vals.append(x_value)
        ref_vals.append(ref_val)
        cand_vals.append(cand_val)

        diff = compute_relative(ref_val, cand_val) if mode == "rel" else (cand_val - ref_val)
        diffs.append(diff)

    abs_diffs = [abs(v) for v in diffs]
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
    max_abs_diff = max(abs_diffs)
    outlier_fraction = sum(1 for v in abs_diffs if v > event_threshold) / len(abs_diffs)

    flagged_reasons = []
    if max_abs_diff > max_threshold:
        flagged_reasons.append("max_diff")
    if mean_abs_diff > mean_threshold:
        flagged_reasons.append("mean_diff")
    if outlier_fraction > outlier_fraction_threshold:
        flagged_reasons.append("outlier_fraction")

    safe_mkdir(output_dir)

    canvas = ROOT.TCanvas(f"c_{sanitize_label(observable)}", "", 1200, 800)
    overlay_title = f"{sample_name} - {observable}"
    ref_graph = make_graph(x_vals, ref_vals, overlay_title, x_axis_field, observable)
    cand_graph = make_graph(x_vals, cand_vals, overlay_title, x_axis_field, observable)
    ref_graph.SetLineColor(ROOT.kBlue + 1)
    ref_graph.SetMarkerColor(ROOT.kBlue + 1)
    cand_graph.SetLineColor(ROOT.kRed + 1)
    cand_graph.SetMarkerColor(ROOT.kRed + 1)

    all_overlay_vals = ref_vals + cand_vals
    ymin = min(all_overlay_vals)
    ymax = max(all_overlay_vals)

    if ymin == ymax:
        eps = 1.0 if ymin == 0.0 else 0.05 * abs(ymin)
        ymin -= eps
        ymax += eps
    else:
        pad = 0.05 * (ymax - ymin)
        ymin -= pad
        ymax += pad

    ref_graph.Draw("ALP")
    hist = ref_graph.GetHistogram()
    if hist:
        hist.SetMinimum(ymin)
        hist.SetMaximum(ymax)

    cand_graph.Draw("LP SAME")
    legend = ROOT.TLegend(0.62, 0.78, 0.90, 0.90)
    legend.AddEntry(ref_graph, reference_name, "lp")
    legend.AddEntry(cand_graph, candidate_name, "lp")
    legend.Draw()
    overlay_plot = output_dir / f"{observable}__{sample_name}.png"
    canvas.SaveAs(str(overlay_plot))
    canvas.Close()

    canvas_delta = ROOT.TCanvas(f"cd_{sanitize_label(observable)}", "", 1200, 800)
    delta_graph = make_graph(x_vals, diffs, f"{sample_name} - delta {observable}", x_axis_field, f"delta({observable})")
    delta_graph.SetLineColor(ROOT.kBlack)
    delta_graph.SetMarkerColor(ROOT.kBlack)

    delta_ymin = min(diffs)
    delta_ymax = max(diffs)

    if delta_ymin == delta_ymax:
        eps = 1.0 if delta_ymin == 0.0 else 0.05 * abs(delta_ymin)
        delta_ymin -= eps
        delta_ymax += eps
    else:
        pad = 0.05 * (delta_ymax - delta_ymin)
        delta_ymin -= pad
        delta_ymax += pad

    delta_graph.Draw("ALP")
    delta_hist = delta_graph.GetHistogram()
    if delta_hist:
        delta_hist.SetMinimum(delta_ymin)
        delta_hist.SetMaximum(delta_ymax)

    delta_plot = output_dir / f"delta_{observable}__{sample_name}.png"
    canvas_delta.SaveAs(str(delta_plot))
    canvas_delta.Close()

    copied_files = []
    if flagged_reasons:
        copied_files.append(str(copy_if_flagged(overlay_plot, different_dir)))
        copied_files.append(str(copy_if_flagged(delta_plot, different_dir)))

    return {
        "sample_name": sample_name,
        "observable": observable,
        "mode": mode,
        "n_common_events": len(common_keys),
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "outlier_fraction": outlier_fraction,
        "flagged": bool(flagged_reasons),
        "flagged_reasons": ",".join(flagged_reasons),
        "plot": str(overlay_plot),
        "delta_plot": str(delta_plot),
        "copied_files": ",".join(copied_files),
    }


def run_compare_for_sample(sample: Sample, base_dirs: Dict[str, Path], config: Dict[str, Any], args) -> Tuple[str, bool, str, List[Dict[str, Any]]]:
    if ROOT is None:
        return sample.name, False, "PyROOT not available.", []

    paths = sample_paths(base_dirs, sample, config)
    ref_file = paths["analysis_ref_output"]
    cand_file = paths["analysis_cand_output"]

    if not root_file_looks_valid(ref_file):
        return sample.name, False, f"Missing reference analysis file: {ref_file}", []
    if not root_file_looks_valid(cand_file):
        return sample.name, False, f"Missing candidate analysis file: {cand_file}", []

    compare_cfg = config["comparison"]
    tree_name = compare_cfg.get("tree_name", "events")
    observables = compare_cfg.get("observables", [])
    branch_names = [entry["name"] for entry in observables]

    reference_data = load_tree_data(ref_file, tree_name, branch_names)
    candidate_data = load_tree_data(cand_file, tree_name, branch_names)

    pair_dir = base_dirs["comparison"] / f"{config['setups']['reference']['label']}__vs__{config['setups']['candidate']['label']}"
    plot_dir = pair_dir / "plots" / sample.name
    different_dir = pair_dir / "different"
    safe_mkdir(plot_dir)
    safe_mkdir(different_dir)

    rows = []
    for observable_cfg in observables:
        rows.append(
            compare_one_observable(
                observable_cfg=observable_cfg,
                sample_name=sample.name,
                reference_data=reference_data,
                candidate_data=candidate_data,
                output_dir=plot_dir,
                different_dir=different_dir,
                x_axis_field=compare_cfg.get("x_axis_field", "event"),
                reference_name=config["setups"]["reference"].get("name", config["setups"]["reference"]["label"]),
                candidate_name=config["setups"]["candidate"].get("name", config["setups"]["candidate"]["label"]),
            )
        )

    return sample.name, True, "COMPARE done.", rows


def write_compare_summaries(base_dirs: Dict[str, Path], config: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    pair_dir = base_dirs["comparison"] / f"{config['setups']['reference']['label']}__vs__{config['setups']['candidate']['label']}"
    safe_mkdir(pair_dir)

    summary_json = pair_dir / "summary.json"
    summary_csv = pair_dir / "summary.csv"
    different_csv = pair_dir / "different_index.csv"

    write_text(summary_json, json.dumps(rows, indent=2), overwrite=True)

    if rows:
        fieldnames = list(rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        flagged = [row for row in rows if row["flagged"]]
        with different_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flagged)


def execute_tasks(step_name: str, tasks, max_concurrent_jobs: int, continue_on_error: bool):
    results = []
    total = len(tasks)
    done = 0
    ok_count = 0
    fail_count = 0
    first_failure_message = None
    heartbeat_seconds = 30

    if total == 0:
        return results

    def log_progress(running: int, heartbeat: bool = False):
        pct = 100.0 * done / total if total else 100.0
        bar = make_progress_bar(done, total)
        prefix = f"{step_name}:heartbeat" if heartbeat else step_name
        logging.info(
            "[%s] %s %d/%d (%.1f%%) | running=%d done=%d ok=%d fail=%d | %s",
            prefix,
            bar,
            done,
            total,
            pct,
            running,
            done,
            ok_count,
            fail_count,
            system_usage_string(),
        )

    def normalize_exception(exc: Exception):
        nonlocal first_failure_message
        msg = f"{type(exc).__name__}: {exc}"
        if first_failure_message is None:
            first_failure_message = msg
        logging.exception("[%s] task raised an exception", step_name)
        return ("<exception>", False, msg)

    if max_concurrent_jobs <= 1:
        last_heartbeat = time.time()
        log_progress(running=1 if total > 0 else 0)
        for task in tasks:
            try:
                result = task()
            except Exception as exc:
                result = normalize_exception(exc)

            results.append(result)
            done += 1
            if len(result) >= 2 and result[1]:
                ok_count += 1
            else:
                fail_count += 1

            log_progress(running=0)

            if not continue_on_error and len(result) >= 2 and not result[1]:
                raise RuntimeError(result[2] if len(result) > 2 else "Task failed.")

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                log_progress(running=0, heartbeat=True)
                last_heartbeat = now
        return results

    with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        pending = {executor.submit(task) for task in tasks}
        last_heartbeat = time.time()
        log_progress(running=min(max_concurrent_jobs, total))

        while pending:
            done_now, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

            for future in done_now:
                try:
                    result = future.result()
                except Exception as exc:
                    result = normalize_exception(exc)

                results.append(result)
                done += 1
                if len(result) >= 2 and result[1]:
                    ok_count += 1
                else:
                    fail_count += 1

                running = min(len(pending), max_concurrent_jobs)
                log_progress(running=running)

                if not continue_on_error and len(result) >= 2 and not result[1]:
                    for f in pending:
                        f.cancel()
                    raise RuntimeError(result[2] if len(result) > 2 else "Task failed.")

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                running = min(len(pending), max_concurrent_jobs)
                log_progress(running=running, heartbeat=True)
                last_heartbeat = now

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Compare HGCAL setups with common GEN, step2, step3, analysis and compare.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--workdir", default=".", help="Working directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--resume", action="store_true", help="Skip outputs already present and valid.")
    parser.add_argument("--steps", default="gen,step2,step3,analysis,compare", help="Comma-separated list of steps.")
    parser.add_argument("--samples", default="", help="Comma-separated glob patterns to select samples.")
    parser.add_argument("--max-concurrent-jobs", type=int, default=1, help="Maximum number of concurrent independent jobs.")
    parser.add_argument("--threads-per-job", type=int, default=1, help="Number of threads used by each cmsRun job.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands but do not execute them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if a task fails.")
    parser.add_argument("--list-samples", action="store_true", help="Print resolved samples and exit.")
    parser.add_argument("--dump-resolved-config", action="store_true", help="Dump resolved config under workdir/config.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if psutil is not None:
        psutil.cpu_percent(interval=None)

    if args.max_concurrent_jobs < 1:
        raise RuntimeError("--max-concurrent-jobs must be >= 1")
    if args.threads_per_job is not None and args.threads_per_job < 1:
        raise RuntimeError("--threads-per-job must be >= 1")

    args.workdir = Path(args.workdir).resolve()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    workflow = config.get("workflow", {})
    threads_per_job = args.threads_per_job if args.threads_per_job is not None else int(workflow.get("threads", 1))

    available_cpus = os.cpu_count() or 1
    total_requested_threads = args.max_concurrent_jobs * threads_per_job
    if total_requested_threads > available_cpus:
        logging.warning(
            "Requested concurrency exceeds available CPUs: %d jobs x %d threads/job = %d threads, system reports %d CPUs.",
            args.max_concurrent_jobs,
            threads_per_job,
            total_requested_threads,
            available_cpus,
        )

    base_dirs = prepare_workdir(args.workdir)

    safe_mkdir(base_dirs["logs"] / "gen")
    safe_mkdir(base_dirs["logs"] / "step2" / "reference")
    safe_mkdir(base_dirs["logs"] / "step2" / "candidate")
    safe_mkdir(base_dirs["logs"] / "step3" / "reference")
    safe_mkdir(base_dirs["logs"] / "step3" / "candidate")
    safe_mkdir(base_dirs["logs"] / "analysis" / "reference")
    safe_mkdir(base_dirs["logs"] / "analysis" / "candidate")
    safe_mkdir(base_dirs["logs"] / "compare")

    selected_patterns = [item.strip() for item in args.samples.split(",") if item.strip()]
    samples = build_samples(config, selected_patterns)

    if args.dump_resolved_config:
        dump_yaml(base_dirs["config"] / "resolved_config.yaml", config)

    if args.list_samples:
        for sample in samples:
            print(sample.name)
        return 0

    logging.info("Resolved %d samples.", len(samples))
    logging.info("Using max_concurrent_jobs=%d, threads_per_job=%d", args.max_concurrent_jobs, threads_per_job)

    if not samples:
        logging.warning("No samples found.")
        return 0

    steps = {item.strip() for item in args.steps.split(",") if item.strip()}

    if "gen" in steps:
        tasks = [
            lambda sample=sample: run_gen_for_sample(sample, base_dirs, config, args, threads_per_job)
            for sample in samples
        ]
        results = execute_tasks("gen", tasks, max_concurrent_jobs=args.max_concurrent_jobs, continue_on_error=args.continue_on_error)
        for name, ok, message in results:
            logging.debug("[%s] %s | %s", "gen", name, message)
            if not ok and not args.continue_on_error:
                return 1

    if "step2" in steps:
        tasks = []
        for sample in samples:
            tasks.append(
                lambda sample=sample: run_step2_for_sample_and_setup(
                    sample, "reference", base_dirs, config, args, threads_per_job
                )
            )
            tasks.append(
                lambda sample=sample: run_step2_for_sample_and_setup(
                    sample, "candidate", base_dirs, config, args, threads_per_job
                )
            )
        results = execute_tasks("step2", tasks, max_concurrent_jobs=args.max_concurrent_jobs, continue_on_error=args.continue_on_error)
        for name, ok, message in results:
            logging.debug("[%s] %s | %s", "step2", name, message)
            if not ok and not args.continue_on_error:
                return 1

    if "step3" in steps:
        tasks = []
        for sample in samples:
            tasks.append(
                lambda sample=sample: run_step3_for_sample_and_setup(
                    sample, "reference", base_dirs, config, args, threads_per_job
                )
            )
            tasks.append(
                lambda sample=sample: run_step3_for_sample_and_setup(
                    sample, "candidate", base_dirs, config, args, threads_per_job
                )
            )
        results = execute_tasks("step3", tasks, max_concurrent_jobs=args.max_concurrent_jobs, continue_on_error=args.continue_on_error)
        for name, ok, message in results:
            logging.debug("[%s] %s | %s", "step3", name, message)
            if not ok and not args.continue_on_error:
                return 1

    if "analysis" in steps:
        tasks = []
        for sample in samples:
            tasks.append(
                lambda sample=sample: run_analysis_for_sample_and_setup(
                    sample, "reference", base_dirs, config, args, threads_per_job
                )
            )
            tasks.append(
                lambda sample=sample: run_analysis_for_sample_and_setup(
                    sample, "candidate", base_dirs, config, args, threads_per_job
                )
            )
        results = execute_tasks("analysis", tasks, max_concurrent_jobs=args.max_concurrent_jobs, continue_on_error=args.continue_on_error)
        for name, ok, message in results:
            logging.debug("[%s] %s | %s", "analysis", name, message)
            if not ok and not args.continue_on_error:
                return 1

    if "compare" in steps:
        tasks = [lambda sample=sample: run_compare_for_sample(sample, base_dirs, config, args) for sample in samples]
        results = execute_tasks("compare", tasks, max_concurrent_jobs=args.max_concurrent_jobs, continue_on_error=args.continue_on_error)

        all_rows = []
        for name, ok, message, rows in results:
            logging.debug("[%s] %s | %s", "compare", name, message)
            all_rows.extend(rows)
            if not ok and not args.continue_on_error:
                return 1

        write_compare_summaries(base_dirs, config, all_rows)

    logging.info("Workflow completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())