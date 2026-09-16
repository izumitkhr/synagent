import base64
import datetime
import json
import logging
import mimetypes
import os

import openai

logger = logging.getLogger(__name__)

# Per-attempt timeout and retry budget for all OpenAI calls. A shorter
# timeout with more retries recovers from a silently-dead connection in
# minutes rather than the 10-minute stall of the SDK defaults (600 s, 2).
LLM_TIMEOUT_SECONDS = float(os.environ.get("SYNAGENT_LLM_TIMEOUT", "180"))
LLM_MAX_RETRIES = int(os.environ.get("SYNAGENT_LLM_MAX_RETRIES", "5"))

_agents_client_configured = False


def configure_agents_default_client():
    """Point the Agents SDK at an AsyncOpenAI client with our timeout/retry (idempotent)."""
    global _agents_client_configured
    if _agents_client_configured:
        return
    from openai import AsyncOpenAI
    from agents import set_default_openai_client
    set_default_openai_client(
        AsyncOpenAI(timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)
    )
    _agents_client_configured = True
    logger.info("LLM client: timeout=%.0fs, max_retries=%d",
                LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES)


def _image_data_url(image_path):
    """Encode a local image file as a base64 data URL for the Responses API."""
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"Not a recognized image file: {image_path}")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _log_prompt_to_file(log_dir, log_tag, llm_model, format, previous_response_id, prompt, response_text, image_paths=None):
    """Append a prompt/response pair to ``{log_dir}/{log_tag}.log``."""
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{log_tag or 'untagged'}.log"
    log_path = os.path.join(log_dir, filename)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(
            f"[{datetime.datetime.now().isoformat(timespec='seconds')}] "
            f"model={llm_model} format={format} chained={bool(previous_response_id)}"
        )
        if previous_response_id:
            f.write(f" previous_response_id={previous_response_id}")
        f.write("\n" + "=" * 78 + "\n\n--- PROMPT ---\n")
        if image_paths:
            f.write(f"[attached images: {', '.join(str(p) for p in image_paths)}]\n")
        f.write(prompt)
        if not prompt.endswith("\n"):
            f.write("\n")
        f.write("\n--- RESPONSE ---\n")
        f.write(response_text)
        if not response_text.endswith("\n"):
            f.write("\n")
        f.write("\n")


def call_llm(prompt, llm_model="gpt-5.5", format=None, previous_response_id=None, log_tag=None, image_paths=None):
    """Call the OpenAI Responses API and return ``(parsed_output, response_id)``.

    ``previous_response_id`` chains the call off a prior response server-side.
    ``image_paths`` are attached to the user message as vision inputs. When
    ``SYNAGENT_PROMPT_LOG_DIR`` is set, the prompt and response are appended
    to ``{log_dir}/{log_tag}.log``.
    """
    chained_note = " (chained)" if previous_response_id else ""
    image_note = f" (+{len(image_paths)} image(s))" if image_paths else ""
    logger.info(
        "Calling LLM (model=%s, format=%s)%s%s...",
        llm_model, format, chained_note, image_note,
    )
    client = openai.OpenAI(timeout=LLM_TIMEOUT_SECONDS,
                           max_retries=LLM_MAX_RETRIES)
    if image_paths:
        content = [{"type": "input_text", "text": prompt}]
        for p in image_paths:
            content.append(
                {"type": "input_image", "image_url": _image_data_url(p)}
            )
        llm_input = [{"role": "user", "content": content}]
    else:
        llm_input = prompt
    kwargs = {"model": llm_model, "input": llm_input}
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    if format == "json_object":
        kwargs["text"] = {"format": {"type": "json_object"}}
    elif format is not None:
        raise ValueError(f"format={format} is not supported!")

    response = client.responses.create(**kwargs)
    raw = response.output_text
    response_id = response.id

    log_dir = os.environ.get("SYNAGENT_PROMPT_LOG_DIR")
    if log_dir:
        _log_prompt_to_file(
            log_dir, log_tag, llm_model, format, previous_response_id, prompt, raw,
            image_paths=image_paths,
        )

    if format == "json_object":
        try:
            output = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON output: %s\nRaw output:\n%s", e, raw)
            raise

    else:
        output = raw

    logger.info("LLM call completed.")
    return output, response_id
