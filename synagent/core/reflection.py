import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from synagent.core.llm import call_llm
from synagent.core.prompts.reflection import (
    get_prompt as get_prompt_reflection,
    get_chained_prompt as get_chained_prompt_reflection,
)

logger = logging.getLogger(__name__)

class ReflectionOut(BaseModel):
    """Structured output of the reflection turn (session mode).

    Field order is generation order: verdict and reasoning first, then the
    SEM observation, and the abduction last so it can draw on the
    morphological evidence. ``sem_observation`` is null when no SEM image was
    attached; ``abduction`` is null when supported.
    """
    judged_as: Literal["supported", "refuted"]
    reasoning: str
    sem_observation: str | None
    abduction: str | None


# Cap on the JSON text a single use_skill call feeds back into the turn.
_SKILL_RESULT_MAX_CHARS = 4000
# Cap on ad-hoc exploration calls per reflection turn, and on how much of
# each exploration is persisted into history.json.
_EXPLORE_BUDGET = 5
_EXPLORE_HISTORY_MAX_CHARS = 2000
_EXPLORE_MAX_IMAGES_PER_CALL = 2


def _load_skill_library():
    """Return {skill_name: (skill_dir, metadata)} from the skills/ library."""
    from synagent.preprocess.skill_generation import _list_skills_with_metadata
    return {d.name: (d, meta) for d, meta in _list_skills_with_metadata()}


def _skills_catalog_block(skills):
    """Prompt block advertising the skill library to the reflection turn."""
    lines = [
        "",
        "=== Skill library ===",
        "You have a use_skill(skill_name) tool that runs one of the following "
        "analysis skills on THIS iteration's measurement files and returns its "
        "structured result. Before finalizing your judgement, call it whenever "
        "a quantitative analysis would inform the reflection (e.g. morphology "
        "statistics for the attached SEM image). You may call it several times.",
    ]
    for name, (_d, meta) in skills.items():
        lines.append(
            f"- {name} [{meta.get('technique')}]: {meta.get('description')} "
            f"(applicability: {meta.get('applicability')})"
        )
    lines.append(
        "\nIf NO listed skill provides a quantitative analysis you need, you "
        "may call create_skill(technique, request) ONCE this iteration: a "
        "skill-generation agent will build, self-test, run and register a new "
        "skill in the library, and you get its result. Prefer existing skills; "
        "create only what is genuinely missing, and make the request specific "
        "(what to quantify, from which measurement)."
    )
    return "\n".join(lines) + "\n"


def _route_measurement(technique, run_dir, sem_stem=None, xrd_name=None):
    """Map a technique name to this iteration's measurement file.

    Files are pinned via what history records (``xrd_name`` = source_maiml,
    ``sem_stem`` = stem of sem_image) so that files accumulating in data/
    across iterations are never guessed. Returns ``(maiml_path, None)`` or
    ``(None, "[error] ...")``.
    """
    t = str(technique or "").lower()
    run_dir = Path(run_dir)
    if "xrd" in t:
        if xrd_name:
            candidates = [p for p in run_dir.glob("*.maiml")
                          if p.name == xrd_name]
            if not candidates:
                return None, (f"[error] The XRD MaiML recorded for this "
                              f"iteration ('{xrd_name}') is no longer in "
                              f"{run_dir}.")
        else:
            candidates = sorted(run_dir.glob("*.maiml"))
            if len(candidates) > 1:
                return None, ("[error] Multiple XRD MaiML files present and "
                              "none recorded for this iteration - refusing "
                              "to guess.")
    elif "sem" in t:
        if not sem_stem:
            return None, ("[error] No SEM measurement is recorded for this "
                          "iteration.")
        candidates = [p for p in (run_dir / "data").glob("sem_*.maiml")
                      if p.stem == sem_stem]
    else:
        return None, (f"[error] No measurement-file routing for technique "
                      f"'{technique}'. Supported: XRD, SEM.")
    if not candidates:
        return None, (f"[error] No measurement file for technique "
                      f"'{technique}' is available this iteration.")
    return candidates[0], None


def _execute_skill(skill_dir, maiml, skill_name, record, created=False):
    """Run a skill on a measurement file; format, log and record the result."""
    from synagent.preprocess.preprocess import REF_DIR
    from synagent.preprocess.skill_generation import run_skill

    with tempfile.TemporaryDirectory(prefix="use_skill_") as td:
        try:
            result = run_skill(skill_dir, maiml, REF_DIR, td)
        except Exception as e:
            logger.warning("skill %s failed: %s", skill_name, e)
            return f"[error] Skill '{skill_name}' failed on {maiml.name}: {e}"
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) > _SKILL_RESULT_MAX_CHARS:
        text = text[:_SKILL_RESULT_MAX_CHARS] + "...[truncated]"
    logger.info("skill %s on %s -> %d chars", skill_name, maiml.name, len(text))
    entry = {"skill": skill_name, "maiml": maiml.name,
             "result": json.loads(text) if not text.endswith("[truncated]") else text}
    if created:
        entry["created"] = True
    record.append(entry)
    return text


def _explore_block():
    """Prompt block describing the explore_data tool."""
    return (
        "\n=== Data exploration ===\n"
        "You also have an explore_data(code) tool: it executes Python with "
        "these predefined variables: xrd_maiml_path, sem_maiml_path, "
        "sem_image_path (str or None when absent), ref_dir, output_dir. "
        "Use print() to inspect values (e.g. parse the XRD MaiML's <content> "
        "arrays to check peak positions, look for secondary-phase peaks, or "
        "probe the SEM image). Any PNG your code saves under output_dir is "
        "returned to you as a VIEWABLE image (max "
        f"{_EXPLORE_MAX_IMAGES_PER_CALL} per call) - plot and look. "
        f"Budget: {_EXPLORE_BUDGET} calls per iteration. Ad-hoc findings are "
        "provisional evidence: prefer use_skill results for quantitative "
        "claims, and if you repeat the same exploration across iterations, "
        "crystallize it with create_skill.\n"
    )


def _explore_log(log_tag, code, output, n_images):
    """Append one exploration round to prompts/reflection_explore.log."""
    import datetime
    import os
    log_dir = os.environ.get("SYNAGENT_PROMPT_LOG_DIR")
    if not log_dir:
        return
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(Path(log_dir) / "reflection_explore.log", "a",
              encoding="utf-8") as f:
        f.write(
            f"\n{'=' * 78}\n[{stamp}] tag={log_tag}\n--- CODE ---\n{code}\n"
            f"--- OUTPUT ({len(output)} chars, {n_images} image(s)) ---\n"
            f"{output}\n"
        )


def _explore_impl(code, namespace, explore_dir, record, budget, log_tag):
    """Core of explore_data: run code, collect stdout + newly saved PNGs."""
    from synagent.preprocess.skill_generation import run_python_snippet

    if budget["remaining"] <= 0:
        return (f"[error] Exploration budget exhausted "
                f"({_EXPLORE_BUDGET} calls per iteration)."), []
    budget["remaining"] -= 1

    before = set(Path(explore_dir).glob("*.png"))
    output = run_python_snippet(code, namespace)
    new_pngs = sorted(set(Path(explore_dir).glob("*.png")) - before,
                      key=lambda p: p.stat().st_mtime)
    new_pngs = new_pngs[-_EXPLORE_MAX_IMAGES_PER_CALL:]

    logger.info("explore_data: %d chars out, %d new image(s)",
                len(output), len(new_pngs))
    _explore_log(log_tag, code, output, len(new_pngs))
    record.append({
        "code": code,
        "output": output[:_EXPLORE_HISTORY_MAX_CHARS],
        "images": [p.name for p in new_pngs],
    })
    return output, new_pngs


def _build_explore_tool(namespace, explore_dir, record, budget, log_tag):
    """Create the explore_data function tool (ad-hoc code, text+image out)."""
    from agents import function_tool, ToolOutputImage, ToolOutputText
    from synagent.core.llm import _image_data_url

    @function_tool
    def explore_data(code: str):
        """Execute Python to inspect this iteration's raw measurement data.

        Predefined variables: xrd_maiml_path, sem_maiml_path, sem_image_path
        (str or None), ref_dir, output_dir. print() to see values; any PNG
        saved under output_dir is returned as a viewable image. Time-limited;
        avoid long loops.

        Args:
            code: Python source to execute.
        """
        output, pngs = _explore_impl(code, namespace, explore_dir, record,
                                     budget, log_tag)
        out = [ToolOutputText(type="text", text=output)]
        for p in pngs:
            out.append(ToolOutputImage(type="image",
                                       image_url=_image_data_url(p)))
        return out

    return explore_data


def _build_use_skill_tool(skills, run_dir, record, sem_stem=None,
                          xrd_name=None):
    """Create the use_skill function tool bound to this iteration's files.

    Failures are returned to the agent as ``[error] ...`` strings; successful
    calls are appended to ``record`` for history.json.
    """
    from agents import function_tool

    @function_tool
    def use_skill(skill_name: str) -> str:
        """Run one analysis skill from the skill library on the current
        iteration's measurement files and return its structured result as JSON.

        Args:
            skill_name: Name of the skill, exactly as listed in the
                "Skill library" section of the prompt.
        """
        entry = skills.get(skill_name)
        if entry is None:
            return (f"[error] Unknown skill '{skill_name}'. "
                    f"Available: {', '.join(skills)}")
        skill_dir, meta = entry
        maiml, err = _route_measurement(meta.get("technique"), run_dir,
                                        sem_stem=sem_stem, xrd_name=xrd_name)
        if err:
            return err
        return _execute_skill(skill_dir, maiml, skill_name, record)

    return use_skill


def _create_skill_impl(technique, request, skills, run_dir, inputs, record,
                       llm_model, budget, sem_stem=None, xrd_name=None):
    """Core of the create_skill tool: nested skill generation, then run.

    generate_skill runs in a dedicated thread because it drives its own
    Agents SDK loop. On success the new skill is registered in ``skills``.
    """
    if budget["remaining"] <= 0:
        return ("[error] Skill-creation budget exhausted for this iteration "
                "(1 creation per iteration). Use existing skills instead.")
    maiml, err = _route_measurement(technique, run_dir, sem_stem=sem_stem,
                                    xrd_name=xrd_name)
    if err:
        return err
    budget["remaining"] -= 1

    from synagent.preprocess.skill_generation import (
        generate_skill, _read_skill_metadata,
    )
    logger.info("create_skill: generating for technique=%s request=%r",
                technique, request[:120])
    box = {}

    def worker():
        try:
            box["out"] = generate_skill(
                run_dir=run_dir, input_yaml_rel=None, llm_model=llm_model,
                maiml_path=maiml, inputs_yaml=inputs, extra_request=request,
            )
        except Exception as e:  # reported to the agent, not fatal to the turn
            box["err"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    if "err" in box:
        return f"[error] Skill generation failed: {box['err']}"
    skill_dir, test_result = box["out"]
    if skill_dir is None or test_result is None:
        return ("[error] Skill generation did not converge "
                "(self-test failed after retries).")
    meta = _read_skill_metadata(skill_dir) or {}
    skills[skill_dir.name] = (skill_dir, meta)
    logger.info("create_skill: new skill '%s' registered", skill_dir.name)
    return _execute_skill(skill_dir, maiml, skill_dir.name, record,
                          created=True)


def _build_create_skill_tool(skills, run_dir, inputs, record, llm_model,
                             budget, sem_stem=None, xrd_name=None):
    """Create the create_skill function tool (nested skill generation)."""
    from agents import function_tool

    @function_tool
    def create_skill(technique: str, request: str) -> str:
        """Generate a NEW analysis skill (when no listed skill fits), register
        it in the skill library, run it on this iteration's measurement, and
        return its structured result as JSON. Budget: once per iteration.

        Args:
            technique: Which of this iteration's measurements to analyze:
                "XRD" or "SEM".
            request: Specific description of what to quantify (metric,
                method hints, expected output fields).
        """
        return _create_skill_impl(technique, request, skills, run_dir,
                                  inputs, record, llm_model, budget,
                                  sem_stem=sem_stem, xrd_name=xrd_name)

    return create_skill


def reflect_on_latest(history, inputs, understanding, llm_model="gpt-5.5", log_tag=None, run_dir=None, session=None):
    """LLM judge for the latest history entry: supported vs refuted, plus abduction.

    Writes the result into ``history[-1]["reflection"]``. No-op when there is
    no entry, no target yet, no expected_target, or a reflection already
    recorded. Sends a minimal prompt when the proposal's context is available
    in the session (or via ``response_id``), the stand-alone prompt otherwise.
    """
    if not history:
        return history
    last = history[-1]
    if last.get("target") is None:
        return history
    if last.get("reflection") is not None:
        return history
    if last.get("expected_target") is None:
        logger.info("Skipping reflection: latest history entry has no expected_target.")
        return history

    # SEM image recorded for this measurement (path relative to run_dir).
    # A missing file is an error, not a silent degradation.
    image_paths = None
    sem_image = last.get("sem_image")
    if sem_image:
        sem_path = Path(sem_image)
        if not sem_path.is_absolute() and run_dir is not None:
            sem_path = Path(run_dir) / sem_path
        if not sem_path.is_file():
            raise FileNotFoundError(
                f"history.json references SEM image '{sem_image}' but it does "
                f"not exist at {sem_path}."
            )
        image_paths = [sem_path]

    if session is not None:
        # The proposal turn already lives in the session, so the minimal
        # prompt applies (unless the session is fresh).
        from synagent.core.session_llm import call_llm_session, session_has_items
        if session_has_items(session):
            prompt = get_chained_prompt_reflection(last, with_sem_image=bool(image_paths))
        else:
            prompt = get_prompt_reflection(
                inputs=inputs, history=history, understanding=understanding,
                with_sem_image=bool(image_paths),
            )

        # Tools: use_skill, create_skill (once per iteration), explore_data
        # (budgeted ad-hoc code on the raw measurement files).
        tools = None
        skills_used = []
        explorations = []
        explore_dir = None
        if run_dir is not None:
            skills = _load_skill_library()
            prompt += _skills_catalog_block(skills)
            prompt += _explore_block()
            creation_budget = {"remaining": 1}
            explore_budget = {"remaining": _EXPLORE_BUDGET}
            explore_dir = tempfile.mkdtemp(prefix="reflection_explore_")
            # Pin file routing to the measurements recorded in history, never
            # to glob order (data/ accumulates files across iterations).
            sem_stem = Path(sem_image).stem if sem_image else None
            xrd_name = last.get("source_maiml")
            sem_maiml, _err = _route_measurement("sem", run_dir,
                                                 sem_stem=sem_stem)
            xrd_maiml, _err = _route_measurement("xrd", run_dir,
                                                 xrd_name=xrd_name)
            from synagent.preprocess.preprocess import REF_DIR
            namespace = {
                "xrd_maiml_path": str(xrd_maiml) if xrd_maiml else None,
                "sem_maiml_path": str(sem_maiml) if sem_maiml else None,
                "sem_image_path": str(image_paths[0]) if image_paths else None,
                "ref_dir": str(REF_DIR),
                "output_dir": explore_dir,
            }
            tools = [
                _build_use_skill_tool(skills, run_dir, skills_used,
                                      sem_stem=sem_stem, xrd_name=xrd_name),
                _build_create_skill_tool(skills, run_dir, inputs,
                                         skills_used, llm_model,
                                         creation_budget, sem_stem=sem_stem,
                                         xrd_name=xrd_name),
                _build_explore_tool(namespace, explore_dir, explorations,
                                    explore_budget, log_tag),
            ]

        logger.debug("Reflection prompt:\n%s", prompt)
        try:
            output, response_id = call_llm_session(
                prompt,
                session,
                llm_model=llm_model,
                log_tag=log_tag,
                image_paths=image_paths,
                tools=tools,
                output_type=ReflectionOut,
            )
        finally:
            if explore_dir is not None:
                import shutil
                shutil.rmtree(explore_dir, ignore_errors=True)
        if output.get("sem_observation") is None:
            output.pop("sem_observation", None)
        if skills_used:
            last["skills_used"] = skills_used
        if explorations:
            last["explorations"] = explorations
    else:
        previous_response_id = last.get("response_id")
        if previous_response_id:
            prompt = get_chained_prompt_reflection(last, with_sem_image=bool(image_paths))
        else:
            prompt = get_prompt_reflection(
                inputs=inputs, history=history, understanding=understanding,
                with_sem_image=bool(image_paths),
            )
        logger.debug("Reflection prompt:\n%s", prompt)
        output, response_id = call_llm(
            prompt,
            llm_model=llm_model,
            format="json_object",
            previous_response_id=previous_response_id,
            log_tag=log_tag,
            image_paths=image_paths,
        )
    last["reflection"] = output
    last["response_id"] = response_id
    logger.info("Reflection: %s", output.get("judged_as"))
    return history
