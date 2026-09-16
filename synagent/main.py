import logging
from pathlib import Path

import yaml
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

INPUT_FILE_NAME = "result.txt"
OUTPUT_FILE_NAME = "output.txt"


def load_inputs_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        inputs = yaml.safe_load(f)
    return inputs


def load_experimental_result(result_file):
    """Read the target value (a single float) from result.txt."""
    with open(result_file, encoding="utf-8") as f:
        return float(f.read().strip())


def main(args, source_maiml=None, sem_image=None):
    from synagent.core.inference import run_all, load_history, save_history

    if args.run_dir == "default":
        run_dir = Path(__file__).resolve().parents[1]
    else:
        run_dir = Path(args.run_dir).absolute()
    in_file = run_dir / "data" / INPUT_FILE_NAME
    out_file = run_dir / OUTPUT_FILE_NAME

    yaml_path = run_dir / args.input_yaml
    inputs = load_inputs_yaml(yaml_path)

    dim = inputs.get("dim", 1)
    n_expl = len(inputs["expl_variable"])
    if dim != n_expl:
        raise ValueError(
            f"Mismatch: dim={dim} but {n_expl} expl_variable(s) defined in {yaml_path.name}."
        )

    # Record the measured target for the pending proposal.
    history = load_history(run_dir)
    if history:
        # Guard against a stale measurement: the MaiML used for this result
        # must differ from the one ingested for the previous proposal.
        if source_maiml is not None:
            prev_src = (
                history[-2].get("source_maiml") if len(history) >= 2 else None
            )
            if prev_src is not None and prev_src == source_maiml:
                raise RuntimeError(
                    f"Measurement MaiML '{source_maiml}' is identical to the "
                    f"previous proposal's; result.txt may be stale (measurement "
                    f"not updated). Aborting before recording a duplicated target."
                )
            history[-1]["source_maiml"] = source_maiml
        if sem_image is not None:
            # Store relative to run_dir so history.json stays portable.
            history[-1]["sem_image"] = str(
                Path(sem_image).resolve().relative_to(run_dir.resolve())
            )
        history[-1]["target"] = load_experimental_result(in_file)
        save_history(history, run_dir)

    proposal_number = len(history) + 1 if history else 1
    logging.info(f"===== Proposal #{proposal_number} =====")

    expls = run_all(
        inputs=inputs,
        llm_model_proposal=args.llm_model_proposal,
        run_dir=run_dir,
        mode_strategy=getattr(args, "mode_strategy", "alternate"),
        session_memory=getattr(args, "session_memory", False),
        session_max_images=getattr(args, "session_max_images", 3),
    )

    output_txt(expls, inputs, dim, out_file)


def output_txt(expls, inputs, dim, out_file):
    """Write the grid-snapped proposal to output.txt (one index per line)."""
    grid_values = [
        snap_to_grid(
            expls[i],
            inputs["expl_variable"][f"expl{i+1}"]["range"],
            inputs["expl_variable"][f"expl{i+1}"]["grids"],
        )
        for i in range(dim)
    ]
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(str(v) for v in grid_values) + "\n")


def _build_grid(range_, grid_spec):
    n_grids, scale_flag = grid_spec
    if scale_flag == 0:
        return np.linspace(range_[0], range_[1], n_grids)
    return np.logspace(np.log10(range_[0]), np.log10(range_[1]), n_grids)


def snap_to_grid(value, range_, grid_spec):
    """Snap value to the nearest grid point, return the grid INDEX.

    grid_spec: [n_grids, scale_flag]  (scale_flag == 0: linear, else: log)
    """
    grid = _build_grid(range_, grid_spec)
    return int(np.argmin(np.abs(grid - value)))
