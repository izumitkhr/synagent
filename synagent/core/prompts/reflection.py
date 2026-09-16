from synagent.core.prompts.proposal import get_experimental_setup


SEM_IMAGE_BLOCK = """
A scanning electron microscopy (SEM) image of the film surface from this experiment is attached.
Use it as complementary morphological evidence when judging the outcome:
- Note grain size/shape, surface coverage, roughness, cracks, droplets, secondary-phase particles, or any inhomogeneity.
- Relate the observed morphology to the measured target value and to physical/chemical mechanisms (e.g. adatom mobility, Li loss, secondary phase formation) where possible.
- A featureless/smooth image is itself information (e.g. amorphous or very fine-grained film) - state it explicitly.
"""


def get_sem_output_line(with_sem_image):
    if with_sem_image:
        return (
            ', "sem_observation": "<concise description of the film morphology '
            'seen in the attached SEM image and how it informs the judgement>"'
        )
    return ""


def get_task(last_entry, understanding):
    mode = last_entry.get("mode", "verify")
    hypothesis = last_entry.get("hypothesis", "(none recorded)")
    expected = last_entry.get("expected_target")
    actual = last_entry.get("target")
    if understanding:
        current_understanding = understanding[-1]["understanding"]
    else:
        current_understanding = (
            "(no prior understanding update yet; rely on the overview in the "
            "experimental setup section as the implicit prior)"
        )

    return f"""You are reflecting on whether the latest experimental result supports or refutes the current understanding of the system.

The previous proposal was made in mode: "{mode}".
- "verify": the proposal predicted that this condition WOULD achieve the target goal under the current understanding.
- "falsify": the proposal predicted that this condition would NOT achieve the target goal under the current understanding.

Hypothesis being tested:
{hypothesis}

Expected target (predicted before the experiment): {expected}
Actual target (measured): {actual}

Current understanding of the system:
{current_understanding}

Decision procedure:
1. Compare the expected and actual targets, taking the mode and the hypothesis into account.
2. Judge whether the outcome is consistent with the current understanding.
   - In "verify" mode the prediction is "success". Supported = the actual target is close to the expected target and both indicate success. Refuted = the actual target is far from the expected one, or the outcome is qualitatively opposite (predicted success but failed).
   - In "falsify" mode the prediction is "failure". Supported = the actual target is close to the expected one and both indicate failure. Refuted = the condition unexpectedly performed well, i.e. the predicted-failure condition actually achieved the target.
3. If refuted, perform abduction: propose the single most plausible explanation for why the actual outcome diverged from the prediction. Refer to physical / chemical mechanisms that are relevant to this system whenever possible. Be specific: name the claim in the current understanding that is now in doubt, and the alternative claim that the new evidence points to.
4. If supported, briefly state which claim in the current understanding is now more strongly backed by this experiment, and avoid over-extrapolating. When the supported claim references a specific physical / chemical mechanism, say explicitly that this mechanism is now more strongly supported in the tested regime (without over-generalising to untested regimes).
"""


def get_output_format(with_sem_image=False):
    sem_line = get_sem_output_line(with_sem_image)
    return f"""- Output format: A single JSON object with the following structure:
  {{"judged_as": "supported" | "refuted", "reasoning": "<why>"{sem_line}, "abduction": "<plausible explanation when refuted, or null when supported>"}}
- Only output the JSON object. Do not include any other text.
- "reasoning" and "abduction" should be Markdown strings. Set "abduction" to null when "judged_as" is "supported".
"""


def get_prompt(inputs, history, understanding, with_sem_image=False):
    setup = get_experimental_setup(inputs)
    task = get_task(history[-1], understanding)
    if with_sem_image:
        task += SEM_IMAGE_BLOCK
    output_format = get_output_format(with_sem_image)
    return f"""=== Experimental setup ===
{setup}

=== Your task ===
{task}

=== Output requirements ===
{output_format}
"""


def get_chained_prompt(last_entry, with_sem_image=False):
    """Minimal reflection turn when chained off the proposal LLM's prior call.

    Assumes the LLM already has the experimental setup, the hypothesis, the
    mode, the expected_target, and the current understanding in its session
    memory from the proposal turn. Only the actual measurement and the
    judging instructions need to be sent now.
    """
    actual = last_entry.get("target")
    sem_block = SEM_IMAGE_BLOCK if with_sem_image else ""
    sem_line = get_sem_output_line(with_sem_image)
    return f"""The experiment you just proposed has now been carried out.

Actual measured target: {actual}
{sem_block}
Reflect on whether this outcome supports or refutes the prediction you made above.

Decision procedure:
1. Compare the actual target with the expected target you committed to, taking the mode and the hypothesis into account.
2. Judge whether the outcome is consistent with your current understanding.
   - In "verify" mode the prediction was "success". Supported = actual is close to expected and both indicate success. Refuted = actual is far from expected, or qualitatively opposite (predicted success but failed).
   - In "falsify" mode the prediction was "failure". Supported = actual is close to expected and both indicate failure. Refuted = the predicted-failure condition unexpectedly performed well.
3. If refuted, perform abduction: propose the single most plausible explanation for why the actual outcome diverged. Be specific: name the claim in your understanding that is now in doubt, and the alternative claim that the new evidence points to. Use physical / chemical mechanisms relevant to this system whenever possible.
4. If supported, briefly state which claim is now more strongly backed by this experiment, and avoid over-extrapolating. When the supported claim references a specific physical / chemical mechanism, say explicitly that this mechanism is now more strongly supported in the tested regime (without over-generalising to untested regimes).

Output requirements:
- Output a single JSON object: {{"judged_as": "supported" | "refuted", "reasoning": "<why>"{sem_line}, "abduction": "<plausible explanation when refuted, or null when supported>"}}
- Only output the JSON object. Do not include any other text.
- "reasoning" and "abduction" should be Markdown strings. Set "abduction" to null when "judged_as" is "supported".
"""
