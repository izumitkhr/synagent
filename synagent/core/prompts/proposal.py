import math

import numpy as np


def _expl_keys(record):
    return sorted(k for k in record if k.startswith("expl_"))


def _expl_tuple(record):
    return tuple(record[k] for k in _expl_keys(record))


def _format_expls(record):
    return ", ".join(f"{k} = {record[k]:.3f}" for k in _expl_keys(record))


def _format_target(value, opt_scale):
    if opt_scale == "log":
        value = math.log10(value) if value > 0 else float("nan")
    return f"{value:.3f}"


def get_history_string(history, inputs):
    """Render history as (condition -> target) pairs sorted by condition.

    Hypotheses and reflections are omitted (they are synthesised in the
    understanding block); pending proposals are kept so the LLM avoids
    re-proposing them.
    """
    opt_scale = inputs.get("opt-scale", "lin")
    if len(history) == 0:
        return "This is your first proposal.\n"
    sorted_history = sorted(history, key=_expl_tuple)
    lines = ["Observed (condition -> target) pairs, sorted by condition:"]
    for record in sorted_history:
        expls_str = _format_expls(record)
        target = record.get("target")
        if target is None:
            lines.append(f"- {expls_str} -> (pending - proposal awaiting measurement)")
        else:
            lines.append(f"- {expls_str} -> target = {_format_target(target, opt_scale)}")
    return "\n".join(lines) + "\n"


def get_experimental_setup(inputs):
    exp_overview = inputs["overview"]
    exp_target = inputs["target_variable"]
    exp_expl = inputs["expl_variable"]
    exp_conditions = inputs["conditions"]
    opt_scale = inputs.get("opt-scale", "lin")
    if opt_scale == "log":
        target_unit = f"log10({exp_target['unit']})"
    else:
        target_unit = exp_target["unit"]
    prompt = f"""# Overview
{exp_overview}

# Target variable (objective to maximize/minimize or reach a target)
- {exp_target['name']}
  Description: {exp_target['description']} (Unit: {target_unit})

# Explanatory variables (controllable parameters)"""
    for i, (key, var) in enumerate(exp_expl.items()):
        n_grid, scale_flag = var["grids"][0], var["grids"][1]
        assert n_grid > 1
        if scale_flag == 0:
            grid_points = np.linspace(var["range"][0], var["range"][1], n_grid)
        else:
            grid_points = np.logspace(
                np.log10(var["range"][0]), np.log10(var["range"][1]), n_grid
            )
        grid_values_str = ", ".join(f"{v:.4g}" for v in grid_points)
        prompt += f"""
- expl_{i+1}: {var['name']}
  Description: {var['description']}
  Allowed values ({var['unit']}): [{grid_values_str}]
  You MUST choose exactly one value from this list."""

    prompt += "\n\n# Fixed experimental conditions\n"
    for cond in exp_conditions:
        prompt += f"- {cond}\n"
    return prompt


def get_mode_block(mode):
    if mode == "verify":
        return """You are in VERIFY mode.
- Goal: propose an experiment whose outcome, under your current understanding, SHOULD achieve (or come close to) the target objective.
- The proposal is an assertion: "if my current understanding is correct, this condition will succeed".
- The point is to confirm a specific claim in your current understanding by predicting a successful outcome at this condition.
"""
    if mode == "falsify":
        return """You are in FALSIFY mode.
- Goal: propose an experiment in a region your current understanding predicts will NOT achieve the target, so the result either confirms you can deprioritize that region or refutes the prediction and forces you to widen your map.
- The proposal is an assertion: "if my current understanding is correct, this condition will fail / underperform".
- Purpose: counter-balance the verify mode's pull toward the currently-promising region. Deliberately step away from the area you would otherwise keep exploiting.
- Anti-clustering preference: examine the observed-data section. Identify the largest contiguous range of the grid where NO prior sample exists, and strongly prefer conditions in such under-sampled regions over conditions one or two grid steps away from a previously-sampled point. A condition adjacent to a verified point is verify in disguise.
- For genuinely unexplored regions, a coarse "this should not work because <weak mechanism>" prediction is acceptable - you do NOT need a high-confidence mechanistic claim. A surprise success in such a region is the highest-information outcome of any experiment in this campaign.
- Avoid only trivially-bad conditions whose outcome no one would learn from (e.g. a literal extreme corner of the grid that is obviously non-functional). Otherwise, broad coverage of the parameter space beats incremental boundary testing.
"""
    raise ValueError(f"Unknown mode: {mode}")


def get_experimental_task(max_iter, mode):
    mode_block = get_mode_block(mode)
    return f"""{mode_block}
Your overall task is to evolve your understanding of the system by extracting information through up to {max_iter} experiments.
Follow this decision procedure:
1. Read your current understanding (or, if there is none yet, the overview). Identify the single claim (hypothesis) you will put to the test in this iteration. Whenever your current understanding articulates a physical/chemical mechanism, the hypothesis should reference that mechanism explicitly rather than only the phenomenological observation it implies.
2. Choose an experimental condition that fits the mode block above and that genuinely tests this hypothesis.
3. Predict the expected target value if your current understanding is correct. This prediction will later be compared against the actual measurement to judge whether your understanding is supported or refuted, so it must be a concrete number — not a range or a hedge.
4. Check the candidate against the allowed grid values and any hard constraints; fix if it violates them.
5. Briefly explain: which hypothesis is being tested, why the chosen condition fits the mode, and how you derived the expected target.
"""


def get_output_format(mode, inputs):
    dim = inputs["dim"]
    exp_target = inputs["target_variable"]
    exp_expl = inputs["expl_variable"]
    opt_scale = inputs.get("opt-scale", "lin")
    target_name = exp_target["name"]
    target_desc = f"log10 of {target_name}" if opt_scale == "log" else target_name

    expl_lines = "\n".join(
        f"    'expl_{i+1}': <number; one of the allowed values for {exp_expl[f'expl{i+1}']['name']} listed in the setup above>,"
        for i in range(dim)
    )

    return f"""- Output format: A single JSON object with the following keys in this exact order:
  {{
    'mode': '{mode}',
    'hypothesis': '<the specific claim from the current understanding that this experiment will test; reference the underlying physical/chemical mechanism whenever your understanding articulates one>',
{expl_lines}
    'expected_target': <number; your point prediction for {target_desc}, in the same units as shown in the setup above>,
    'reason': '<rationale: which hypothesis is being tested, why the chosen expl_* fits the mode, and how you derived the expected_target>'
  }}
- Only output the JSON object. Do not include any other text.
- 'mode' MUST be exactly '{mode}'.
- 'expected_target' MUST be a single number. Do NOT output a range or a string.
- Each 'expl_*' value MUST be exactly one of the allowed values listed above. Do NOT interpolate or round to other values.
- The candidate (combination of expl_* values) must be different from any condition already listed in the observed data above.
"""


def get_prompt(history, inputs, understanding, mode):
    hist_str = get_history_string(history, inputs)
    experimental_setup = get_experimental_setup(inputs)
    experimental_task = get_experimental_task(max_iter=inputs["max_iter"], mode=mode)
    output_format = get_output_format(mode=mode, inputs=inputs)

    prompt = f"""=== Experimental setup ===
{experimental_setup}
"""
    prompt += f"""=== Observed data ===
{hist_str}
"""
    if len(history) > 0 and understanding:
        prompt += f"""=== Your current understanding ===
{understanding[-1]['understanding']}

"""
    prompt += f"""=== Your task ===
{experimental_task}
"""
    prompt += f"""=== Output requirements ===
{output_format}
"""
    return prompt
