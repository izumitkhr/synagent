"""Persistent LLM session for the proposal/reflection/understanding turns.

Uses the OpenAI Agents SDK's SQLiteSession, persisted to ``{run_dir}/session.db``
so the conversation — including attached SEM images — survives across the
per-iteration process restarts of run.py. This gives the reflection turn
direct visual access to earlier iterations' SEM images.

Cost control: only the most recent ``max_images`` SEM images are replayed;
earlier ones are replaced by a text placeholder (their content survives as
``sem_observation`` text in the conversation).
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from agents import Agent, Runner, SQLiteSession, set_tracing_disabled

from synagent.core.llm import (
    _image_data_url, _log_prompt_to_file, configure_agents_default_client,
)

logger = logging.getLogger(__name__)

set_tracing_disabled(True)

SESSION_ID = "synagent"
SESSION_DB_NAME = "session.db"

OMITTED_IMAGE_PLACEHOLDER = (
    "[An SEM image was attached here in an earlier iteration; it is omitted "
    "from the replayed context to save tokens. Its content is described by "
    "the sem_observation recorded in that iteration's reflection.]"
)


class TrimmedImageSession(SQLiteSession):
    """SQLiteSession that replays only the most recent ``max_images`` images.

    Trimming happens on read, so all images stay stored in the database.
    """

    def __init__(self, *args, max_images=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_images = max_images

    # Keys that may hold a content-part list carrying images: "content" for
    # user/assistant messages, "output" for function_call_output items (e.g.
    # explore_data returning a plot as ToolOutputImage).
    _PART_LIST_KEYS = ("content", "output")

    async def get_items(self, limit=None):
        items = await super().get_items(limit)
        image_positions = []  # (part_list, part_idx)
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in self._PART_LIST_KEYS:
                parts = item.get(key)
                if not isinstance(parts, list):
                    continue
                for j, part in enumerate(parts):
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        image_positions.append((parts, j))
        n_trim = len(image_positions) - self.max_images
        if n_trim <= 0:
            return items
        for parts, j in image_positions[:n_trim]:
            parts[j] = {
                "type": "input_text",
                "text": OMITTED_IMAGE_PLACEHOLDER,
            }
        logger.debug("Trimmed %d old image(s) from session replay.", n_trim)
        return items


def get_session(run_dir, max_images=3):
    """Open (or create) the persistent session for this run_dir."""
    db_path = Path(run_dir) / SESSION_DB_NAME
    return TrimmedImageSession(SESSION_ID, db_path, max_images=max_images)


def session_has_items(session):
    """True if the session already contains prior conversation turns."""
    return bool(asyncio.run(session.get_items(1)))


def call_llm_session(prompt, session, llm_model="gpt-5.5",
                     image_paths=None, log_tag=None, tools=None,
                     output_type=None):
    """Run one turn in the persistent session; returns ``(output, None)``.

    The second element is always ``None`` (no response_id chaining in session
    mode; kept for call_llm signature symmetry). ``tools`` are Agents SDK
    function tools the turn may call. ``output_type`` (pydantic model) makes
    the SDK enforce the output schema and returns the validated dict;
    without it, the final text is returned as-is.
    """
    configure_agents_default_client()
    image_note = f" (+{len(image_paths)} image(s))" if image_paths else ""
    logger.info(
        "Calling LLM in session (model=%s, output=%s)%s...",
        llm_model, output_type.__name__ if output_type else "text", image_note,
    )
    if image_paths:
        content = [{"type": "input_text", "text": prompt}]
        for p in image_paths:
            content.append(
                {"type": "input_image", "image_url": _image_data_url(p)}
            )
        llm_input = [{"role": "user", "content": content}]
    else:
        llm_input = prompt

    agent = Agent(name="synagent", model=llm_model, tools=tools or [],
                  output_type=output_type)
    # Tool-bearing turns (reflection) need headroom above the per-tool budgets
    # (explore_data 5 + create_skill 1 + several use_skill calls).
    max_turns = 16 if tools else 10
    result = Runner.run_sync(agent, llm_input, session=session,
                             max_turns=max_turns)
    if output_type is not None:
        output = result.final_output.model_dump()
        raw = json.dumps(output, ensure_ascii=False)
    else:
        output = raw = str(result.final_output)

    log_dir = os.environ.get("SYNAGENT_PROMPT_LOG_DIR")
    if log_dir:
        _log_prompt_to_file(
            log_dir, log_tag, llm_model, "session", None, prompt, raw,
            image_paths=image_paths,
        )

    logger.info("LLM session call completed.")
    return output, None
