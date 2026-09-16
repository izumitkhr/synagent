import logging
import os

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref')
logger = logging.getLogger(__name__)


def _extract_target_value(result, field, skill_name):
    """Extract the scalar optimization target from a skill's result.

    A skill returns JSON-serializable data. The campaign config
    (target_variable.skill / .field) decides which skill and which field of
    its result define the target. ``field=None`` means the result itself is
    the scalar.
    """
    if field:
        if not isinstance(result, dict) or field not in result:
            raise ValueError(
                f"target_variable.field '{field}' not found in the result of "
                f"skill '{skill_name}'. Result keys: "
                f"{list(result) if isinstance(result, dict) else type(result).__name__}"
            )
        value = result[field]
    else:
        value = result
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Target value from skill '{skill_name}' (field={field!r}) is not "
            f"numeric: {value!r}. Fix target_variable.skill/.field in the "
            f"inputs yaml."
        )


def _write_result_txt(value, output_dir):
    """Write the target scalar to output_dir/result.txt (closed-loop handoff)."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'result.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{value:.8f}\n")


def preprocess(output_dir, maiml_path, run_dir, inputs_yaml_rel,
               llm_model_skill='gpt-5.5', skill_agentic=True):
    """Analyze a measurement file with a skill and return the target value.

    The skill named by ``target_variable.skill`` in the inputs yaml is run
    (and generated first if it does not exist yet). Without a pinned skill,
    the LLM selects a fitting skill from the library or generates a new one.
    Skills write their JSON result to output_dir/result.json; the scalar
    target is extracted here and written to output_dir/result.txt.

    ``skill_agentic`` (default True) lets new-skill generation use a
    tool-using loop in which the LLM runs code to verify its analysis.
    """
    from synagent.preprocess.skill_generation import (
        select_or_generate_skill, run_skill, SKILLS_DIR,
    )
    from synagent.main import load_inputs_yaml

    inputs = load_inputs_yaml(os.path.join(run_dir, inputs_yaml_rel))
    target_cfg = inputs.get('target_variable') or {}
    pinned_skill = target_cfg.get('skill')

    if pinned_skill:
        # A missing pinned skill is generated once, under exactly this name,
        # so the target definition cannot drift after iteration 1. Check the
        # artifact, not just the directory (a stale dir must regenerate).
        skill_dir = SKILLS_DIR / pinned_skill
        if not (skill_dir / "analyze.py").is_file():
            logger.info(
                "Pinned target skill '%s' not found - generating it from "
                "the current measurement...", pinned_skill,
            )
            from synagent.preprocess.skill_generation import generate_skill
            target_field = target_cfg.get('field')
            if target_field:
                shape_note = (
                    f"Return a dict exposing the target scalar under the "
                    f"key '{target_field}'."
                )
            else:
                shape_note = (
                    "Return the target scalar as a bare float (the "
                    "campaign config has no result field configured)."
                )
            extra_request = (
                "This skill defines the campaign's OPTIMIZATION TARGET: "
                "it must quantify the experiment's target_variable "
                f"({target_cfg.get('name')}) from the provided "
                f"measurement. {shape_note}"
            )
            skill_dir, test_value = generate_skill(
                run_dir=run_dir,
                input_yaml_rel=inputs_yaml_rel,
                llm_model=llm_model_skill,
                agentic=skill_agentic,
                maiml_path=maiml_path,
                inputs_yaml=inputs,
                extra_request=extra_request,
                force_name=pinned_skill,
            )
            if test_value is None:
                raise RuntimeError(
                    f"Generation of pinned target skill '{pinned_skill}' "
                    f"did not converge (self-test failed). See the last "
                    f"attempt under {skill_dir}."
                )
        logger.info('Using pinned target skill: %s', pinned_skill)
        result = run_skill(skill_dir, maiml_path, REF_DIR, output_dir)
        value = _extract_target_value(
            result, target_cfg.get('field'), pinned_skill
        )
        _write_result_txt(value, output_dir)
        return value

    # No pinned skill: the LLM picks a fitting skill (or generates one). A
    # non-scalar result is a config error (pin target_variable.skill/.field
    # instead of relying on selection).
    skill_dir = select_or_generate_skill(
        maiml_path=maiml_path,
        inputs_yaml_rel=inputs_yaml_rel,
        run_dir=run_dir,
        llm_model=llm_model_skill,
        agentic=skill_agentic,
    )
    logger.info('Using skill: %s', skill_dir.name)
    result = run_skill(skill_dir, maiml_path, REF_DIR, output_dir)
    value = _extract_target_value(result, None, skill_dir.name)
    _write_result_txt(value, output_dir)
    return value
