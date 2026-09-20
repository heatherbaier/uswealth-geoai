"""
Cross-quarter generalization: for one already-trained model
(--model-year/--model-quarter), generate a `task: validate` config (+ SLURM
job) evaluating it against EACH quarter in --imagery-quarters, restricted
in every case to that MODEL's own held-out test set -- never the imagery
quarter's train/val chips, and never a different held-out set per imagery
quarter.

Why this is safe to say "restricted to the MODEL's own held-out test set"
even when the imagery is a totally different quarter's chips: this
project's stable/spatial split (splitting.py in sail, split_strategy:
stable) makes an item's train/val/test bucket a pure function of its own
identity (or spatial block) + the seed -- not of which other items are
present. So as long as every quarter for a state was trained with the same
seed/split/spatial_block_deg (state_registry.yml enforces that -- one
value per state, not per quarter), a GEOID's test-set membership is
IDENTICAL across every quarter's dataset. That's what makes "model trained
on Q1's imagery, evaluated on Q3's imagery for the same held-out
locations" a meaningful, leak-free cross-domain generalization test
instead of an accident of two different random splits happening to
overlap.

Mechanism (no sail code changes needed beyond PR #23's output-filename
fix): sail's run_validation, with dataset.new=False, loads
<model's ckpt_dir>/test_indices.txt (written at train time) and
intersects it with whatever items are actually present at
dataset.data_root/prefix. Point output_dir/experiment_name at the MODEL
(so it loads that model's checkpoint and its own test_indices.txt) and
dataset.data_root/prefix at the IMAGERY quarter, and the intersection
naturally becomes "chips from the imagery quarter whose GEOID is in the
model's own held-out set" -- exactly the target set, without needing the
imagery quarter's own test_indices.txt at all. The diagonal case
(--imagery-quarters includes the model's own quarter) falls out of this
same logic with no special-casing: it's just generate_validate_config.py's
normal validate config.

sail PR #23 (include dataset.prefix in the output CSV filename) is a
prerequisite: without it, running this for several imagery quarters
against the same model would silently overwrite each run's predictions in
that model's ckpt_dir, since they'd all write the plain
epoch<N>_valset_preds.csv name.

Usage:
    # AZ wealth_index_sat, Q1 2016's model, evaluated against all 4 quarters
    # of 2016 (including its own -- the diagonal entry):
    python pipeline_configs/generate_cross_validate_config.py \
        --state az --variable wealth_index_sat \
        --model-year 2016 --model-quarter 1 \
        --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4 --launch

Run this once per already-trained model (looping --model-quarter/--model-year
yourself, e.g. from a shell loop) to build up a full grid; then use
4_analysis/build_cross_quarter_r2_matrix.py to assemble the resulting CSVs
into the R^2 matrix.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_train_config import (  # noqa: E402
    PROJECT_ROOT, TARGET_CHOICES,
    load_registry, resolve_state_settings, write_job_file,
)
from generate_validate_config import latest_existing_version  # noqa: E402

IMAGERY_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$", re.IGNORECASE)

# Subfolder (under the model's own ckpt_dir) every cross-quarter prediction
# CSV lands in -- see the validator.output_subdir comment in build_config.
# build_cross_quarter_r2_matrix.py imports this rather than hardcoding it a
# second time, so the two scripts can't drift apart on where to look.
CROSS_QUARTER_SUBDIR = "cross_quarter"


def parse_imagery_quarter(token: str):
    m = IMAGERY_QUARTER_RE.match(token.strip())
    if not m:
        raise SystemExit(f"--imagery-quarters entry {token!r} isn't YYYYQN, e.g. 2016Q1")
    return int(m.group(1)), int(m.group(2))


def build_config(state, variable, model_year, model_quarter,
                  imagery_year, imagery_quarter, version, device, registry):
    st = resolve_state_settings(state, registry)

    model_data_root = st["data_root_template"].format(state=state, year=model_year, quarter=model_quarter)
    model_base_prefix = st["base_prefix_template"].format(state=state, year=model_year, quarter=model_quarter)
    model_data_root = model_data_root if model_data_root.endswith("/") else model_data_root + "/"
    model_output_dir = model_data_root + "artifacts/"

    exp_base = f"{state}_q{model_quarter}_{model_year}_{variable}"
    version = version or latest_existing_version(model_output_dir, exp_base)
    experiment_name = f"{exp_base}_{version}"

    imagery_data_root = st["data_root_template"].format(state=state, year=imagery_year, quarter=imagery_quarter)
    imagery_base_prefix = st["base_prefix_template"].format(state=state, year=imagery_year, quarter=imagery_quarter)
    imagery_data_root = imagery_data_root if imagery_data_root.endswith("/") else imagery_data_root + "/"
    imagery_prefix = f"{imagery_base_prefix}_{variable}"

    cfg = {
        "task": "validate",
        # output_dir/experiment_name identify the MODEL: which checkpoint
        # gets loaded (highest_epoch(ckpt_dir)), which test_indices.txt
        # gets read, and where the output CSV is written.
        "experiment_name": experiment_name,
        "output_dir": model_output_dir,
        "dataset": {
            "type": "json",
            # data_root/prefix identify the IMAGERY being evaluated --
            # can differ from the model's own quarter. new: False makes
            # run_validation intersect this quarter's available items with
            # the MODEL's own test_indices.txt (see module docstring).
            "data_root": imagery_data_root,
            "prefix": imagery_prefix,
            "img_size": [256, 256],
            "num_workers": 0,
            "seed": 1337,
            "temporal": False,
            "new": False,
            "band_mean": st["band_mean"],
            "band_std": st["band_std"],
        },
        "model": {
            "name": "swin",
            "params": {"in_channels": st["in_channels"]},
        },
        "validator": {
            "device": device,
            # Keep every cross-quarter prediction CSV (including the
            # diagonal entry, if --imagery-quarters includes the model's
            # own quarter) in its own subfolder under the model's
            # ckpt_dir, separate from that model's normal single-quarter
            # validate output (generate_validate_config.py, no
            # output_subdir set) -- needs sail PR #25. Without this,
            # once a model has been both normally validated AND
            # cross-validated, 3_validation/validate.py's "exactly one
            # epoch*_valset_*_preds.csv per quarter directory" glob would
            # match more than one file.
            "output_subdir": CROSS_QUARTER_SUBDIR,
        },
    }
    return cfg, experiment_name


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", required=True, choices=TARGET_CHOICES)
    p.add_argument("--model-year", required=True, type=int,
                    help="Year of the already-trained model being evaluated")
    p.add_argument("--model-quarter", required=True, type=int, choices=[1, 2, 3, 4],
                    help="Quarter of the already-trained model being evaluated")
    p.add_argument("--imagery-quarters", required=True, nargs="+",
                    help="One or more YYYYQN tokens (e.g. 2016Q1 2016Q2 2016Q3 2016Q4) -- "
                         "the model is evaluated against each of these quarters' imagery, "
                         "restricted to its own held-out test set. Include the model's own "
                         "quarter/year here too if you want the diagonal entry generated "
                         "in the same batch.")
    p.add_argument("--version", default=None,
                    help="Which trained v<N> of the model to evaluate. Default: latest existing")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=None,
                    help="Default: configs/sail/tlags/<state>/cross/")
    p.add_argument("--no-job-file", action="store_true", help="Skip writing SLURM job files")
    p.add_argument("--nodes", default="1")
    p.add_argument("--gpus", default="1")
    p.add_argument("--cpus", default="8")
    p.add_argument("--walltime", default="24:00:00")
    p.add_argument("--partition", default="general")
    p.add_argument("--qos", default="grp_hbaier")
    p.add_argument("--conda-env", default="geomain")
    p.add_argument("--repo-dir", default="/home/hbaier/packages/uswealth-geoai",
                    help="Working directory the SLURM job cd's into -- this project repo, "
                         "not sail (sail is used as an installed command, not by path)")
    p.add_argument("--launch", action="store_true",
                    help="sbatch each job file immediately after writing it (requires job files)")
    args = p.parse_args()

    registry = load_registry()
    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "configs", "sail", "tlags", args.state, "cross"
    )
    os.makedirs(out_dir, exist_ok=True)

    for token in args.imagery_quarters:
        imagery_year, imagery_quarter = parse_imagery_quarter(token)
        cfg, experiment_name = build_config(
            args.state, args.variable, args.model_year, args.model_quarter,
            imagery_year, imagery_quarter, args.version, args.device, registry,
        )

        tag = f"{experiment_name}_on_{imagery_year}q{imagery_quarter}"
        out_path = os.path.join(out_dir, f"{tag}_validate.yml")
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"Wrote {out_path}")
        print(f"  model: {experiment_name}   imagery: {imagery_year} Q{imagery_quarter}")

        job_path = None
        if not args.no_job_file:
            job_path = os.path.join(out_dir, f"{tag}.sh")
            write_job_file(job_path, tag, out_path, args.nodes, args.gpus,
                            args.cpus, args.walltime, args.partition, args.qos,
                            args.conda_env, args.repo_dir)
            print(f"  Wrote {job_path}")

        if args.launch:
            if job_path is None:
                raise SystemExit("--launch requires job files -- drop --no-job-file")
            subprocess.run(["sbatch", job_path], check=True)


if __name__ == "__main__":
    main()
