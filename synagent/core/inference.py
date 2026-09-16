import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, create_model

from synagent.core.prompts.proposal import get_prompt as get_prompt_proposal
from synagent.core.prompts.understanding import (
    get_prompt as get_prompt_understanding,
    get_chained_prompt as get_chained_prompt_understanding,
)
from synagent.core.reflection import reflect_on_latest
from synagent.core.mode import pick_mode
from synagent.core.llm import call_llm

logger = logging.getLogger(__name__)


class UnderstandingOut(BaseModel):
    """Structured output of the understanding-update turn (session mode)."""
    understanding: str


def _proposal_output_type(dim):
    """Build the proposal turn's output model for the given dimensionality.

    The expl_1..expl_{dim} keys depend on the campaign config, so the model
    is created at runtime; field order matches the prompt's reasoning order.
    """
    fields = {f"expl_{i + 1}": (float, ...) for i in range(dim)}
    return create_model(
        "ProposalOut",
        mode=(Literal["verify", "falsify"], ...),
        hypothesis=(str, ...),
        **fields,
        expected_target=(float, ...),
        reason=(str, ...),
    )


def update_history(history, inputs, understanding, mode, llm_model="gpt-5.5", log_tag=None, session=None):
    """Run the proposal turn.

    Without ``session``: fresh Responses API call, no chaining.
    With ``session``: the turn is appended to the persistent run_dir session,
    so the LLM also sees all prior turns (incl. recent SEM images).
    """
    prompt = get_prompt_proposal(
        history=history, inputs=inputs, understanding=understanding, mode=mode
    )
    logger.debug("Proposal prompt:\n%s", prompt)
    if session is not None:
        from synagent.core.session_llm import call_llm_session
        output, response_id = call_llm_session(
            prompt, session, llm_model=llm_model, log_tag=log_tag,
            output_type=_proposal_output_type(inputs.get("dim", 1)),
        )
    else:
        output, response_id = call_llm(
            prompt, llm_model=llm_model, format="json_object", log_tag=log_tag
        )
    output.setdefault("mode", mode)
    output.setdefault("target", None)
    output["response_id"] = response_id
    output["proposed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    history.append(output)
    dim = inputs.get("dim", 1)
    expls = [output[f"expl_{i+1}"] for i in range(dim)]
    return history, expls


def load_history(run_dir: Path):
    """Load history from a JSON file."""
    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    return history


def save_history(history, run_dir: Path):
    """Save history to a JSON file."""
    history_path = run_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def update_understanding(history, inputs, understanding, llm_model="gpt-5.5", previous_response_id=None, log_tag=None, session=None):
    """Run the understanding-update turn.

    With ``session``: the turn runs inside the persistent run_dir session
    (the proposal + reflection context is already there) and a minimal
    prompt is sent. Otherwise, when ``previous_response_id`` is provided,
    the call is chained off the prior reflection turn server-side; else the
    stand-alone prompt is used. Returns ``(understanding, response_id)``.
    """
    if session is not None or previous_response_id:
        prompt = get_chained_prompt_understanding(understanding)
    else:
        prompt = get_prompt_understanding(history=history, inputs=inputs, understanding=understanding)
    logger.debug("Understanding prompt:\n%s", prompt)
    if session is not None:
        from synagent.core.session_llm import call_llm_session
        output, response_id = call_llm_session(
            prompt, session, llm_model=llm_model, log_tag=log_tag,
            output_type=UnderstandingOut,
        )
    else:
        output, response_id = call_llm(
            prompt,
            llm_model=llm_model,
            format="json_object",
            previous_response_id=previous_response_id,
            log_tag=log_tag,
        )
    output["updated_by"] = "understanding_llm"
    understanding.append(output)
    return understanding, response_id


def load_understanding(run_dir: Path):
    understanding_path = run_dir / "understanding.json"
    if understanding_path.exists():
        with open(understanding_path, "r", encoding="utf-8") as f:
            understanding = json.load(f)
    else:
        understanding = []
    return understanding


def save_understanding(understanding, run_dir: Path):
    understanding_path = run_dir / "understanding.json"
    with open(understanding_path, "w", encoding="utf-8") as f:
        json.dump(understanding, f, indent=2)


def run_all(
    inputs,
    run_dir: Path,
    llm_model_proposal: str = "gpt-5.5",
    mode_strategy: str = "alternate",
    session_memory: bool = False,
    session_max_images: int = 3,
):
    history = load_history(run_dir)
    understanding = load_understanding(run_dir)

    # Session-memory mode: all turns share one conversation persisted in
    # run_dir/session.db, so recent SEM images stay visible across iterations.
    session = None
    if session_memory:
        from synagent.core.session_llm import get_session
        session = get_session(run_dir, max_images=session_max_images)
        logger.info(
            "Session memory: ON (db=%s, max replayed images=%d)",
            run_dir / "session.db", session_max_images,
        )

    # history[-1] is the pending proposal from the previous run (1-based index
    # = len(history)); this run closes it out and proposes the next one.
    closing_iter = len(history)
    next_iter = closing_iter + 1

    # === Reflection on the latest measurement ===
    # Runs before the understanding update so that step can consume it.
    if history and history[-1].get("target") is not None and history[-1].get("reflection") is None:
        history = reflect_on_latest(
            history=history,
            inputs=inputs,
            understanding=understanding,
            llm_model=llm_model_proposal,
            log_tag=f"iter{closing_iter:02d}_reflection",
            run_dir=run_dir,
            session=session,
        )
        save_history(history, run_dir)

    # === Update understanding ===
    if history:
        previous_response_id = None if session is not None else history[-1].get("response_id")
        understanding, response_id = update_understanding(
            history=history,
            inputs=inputs,
            understanding=understanding,
            llm_model=llm_model_proposal,
            previous_response_id=previous_response_id,
            log_tag=f"iter{closing_iter:02d}_understanding",
            session=session,
        )
        if response_id:
            history[-1]["response_id"] = response_id
            save_history(history, run_dir)
        save_understanding(understanding, run_dir)

    # === Pick mode and propose the next experiment ===
    mode = pick_mode(history, strategy=mode_strategy)
    logger.info("Proposal mode: %s", mode)
    updated_history, expls = update_history(
        history, inputs, understanding, mode=mode, llm_model=llm_model_proposal,
        log_tag=f"iter{next_iter:02d}_proposal",
        session=session,
    )
    save_history(updated_history, run_dir)
    return expls
