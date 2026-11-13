"""
Write one SLURM job per state that runs 3_analysis/band_importance.py on
the GPU partition -- the band-importance counterpart to generate_train_
config.py/generate_validate_config.py's job files, except there's no sail
YAML config to generate here: band_importance.py is a standalone script
(it builds sail's dataset/model objects directly in-process, see its
docstring), so each job's body is a plain `python 3_analysis/band_
importance.py ...` invocation instead of `simba <config>`. Same SBATCH
header/module-load/conda-activate convention as generate_train_config.py's
JOB_TEMPLATE, reused here rather than re-invented.

Usage:
    python pipeline_configs/generate_band_importance_jobs.py \
        --states oh az ca ga pa --variable wealth_index \
        --years 2016 2017 2018 2019 --offdiag --launch

    # Fewer states / a narrower grid, without launching:
    python pipeline_configs/generate_band_importance_jobs.py \
        --states oh --variable wealth_index --years 2017 2018 2019
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_train_config import PROJECT_ROOT, TARGET_CHOICES, load_registry  # noqa: E402

JOB_TEMPLATE = """#!/bin/bash
#SBATCH -N {nodes}            # number of nodes
#SBATCH -G {gpus}
#SBATCH -c {cpus}            # number of cores
#SBATCH -t {walltime}   # time in d-hh:mm:ss
#SBATCH -p {partition}      # partition
#SBATCH -q {qos}       # QOS
#SBATCH -J {job_name}
#SBATCH -o slurm.{job_name}.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.{job_name}.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

#Activate our environment
source activate {conda_env}

#Change to the directory of our project
cd {repo_dir}

#Band importance for one state -- see 3_analysis/band_importance.py's
#docstring. Unlike the train/validate jobs, this isn't a sail `simba`
#config -- it's a standalone script that builds sail's dataset/model
#objects directly.
{command}
"""


def build_command(state: str, args) -> str:
    parts = [
        "python", "3_analysis/band_importance.py",
        "--state", state, "--variable", args.variable,
        "--years", *[str(y) for y in args.years],
        "--quarters", *[str(q) for q in args.quarters],
        "--methods", *args.methods,
        "--n-repeats", str(args.n_repeats),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--out-dir", args.out_dir,
    ]
    if args.offdiag:
        parts.append("--offdiag")
    if args.use_state_registry:
        parts.append("--use-state-registry")
    if args.version:
        parts += ["--version", args.version]
    if args.cache_dir:
        parts += ["--cache-dir", args.cache_dir]
    if args.overwrite:
        parts.append("--overwrite")
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", default="wealth_index", choices=TARGET_CHOICES)
    p.add_argument("--years", required=True, nargs="+", type=int)
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--methods", nargs="+", default=["permutation", "group_permutation", "mean_ablation"],
                    choices=["permutation", "group_permutation", "mean_ablation"])
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--offdiag", action="store_true")
    p.add_argument("--use-state-registry", action="store_true")
    p.add_argument("--version", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", default="3_analysis/outputs/band_importance")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-jobs-dir", default=None,
                    help="Default: configs/sail/tlags/band_importance/")
    p.add_argument("--nodes", default="1")
    p.add_argument("--gpus", default="1")
    p.add_argument("--cpus", default="8")
    p.add_argument("--walltime", default="24:00:00")
    p.add_argument("--partition", default="general")
    p.add_argument("--qos", default="grp_hbaier")
    p.add_argument("--conda-env", default="geomain")
    p.add_argument("--repo-dir", default="/home/hbaier/packages/uswealth-geoai")
    p.add_argument("--launch", action="store_true",
                    help="sbatch each job file immediately after writing it")
    args = p.parse_args()

    registry = load_registry()
    unknown = [s for s in args.states if s.lower() not in registry]
    if unknown:
        raise SystemExit(f"--states has state(s) not in state_registry.yml: {unknown} "
                          f"(known: {sorted(registry)})")

    out_jobs_dir = args.out_jobs_dir or os.path.join(PROJECT_ROOT, "configs", "sail", "tlags",
                                                        "band_importance")
    os.makedirs(out_jobs_dir, exist_ok=True)

    for state in args.states:
        state = state.lower()
        job_name = f"band_importance_{state}_{args.variable}"
        command = build_command(state, args)
        content = JOB_TEMPLATE.format(
            nodes=args.nodes, gpus=args.gpus, cpus=args.cpus, walltime=args.walltime,
            partition=args.partition, qos=args.qos, job_name=job_name,
            conda_env=args.conda_env, repo_dir=args.repo_dir, command=command,
        )
        job_path = os.path.join(out_jobs_dir, f"{job_name}.sh")
        with open(job_path, "w") as f:
            f.write(content)
        print(f"Wrote {job_path}")
        print(f"  {command}")

        if args.launch:
            subprocess.run(["sbatch", job_path], check=True)


if __name__ == "__main__":
    main()
