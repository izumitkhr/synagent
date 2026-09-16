from synagent.core.prompts.proposal import get_experimental_setup, get_history_string


def get_reflection_block(history):
    if not history:
        return ""
    last = history[-1]
    reflection = last.get("reflection")
    if not reflection:
        return ""

    judged = reflection.get("judged_as", "?")
    reasoning = reflection.get("reasoning", "")
    abduction = reflection.get("abduction")
    mode = last.get("mode", "?")
    hypothesis = last.get("hypothesis", "(none recorded)")
    expected = last.get("expected_target")
    actual = last.get("target")

    block = f"""Reflection on the latest experiment is available:
- Mode of the proposal: {mode}
- Hypothesis tested: {hypothesis}
- Expected target: {expected}
- Actual target: {actual}
- Judged as: {judged}
- Reasoning: {reasoning}
"""
    if abduction:
        block += f"""- Abductive explanation for the divergence: {abduction}

Use the abduction as a candidate revision: if it holds up, name explicitly which claim in your previous understanding is now in doubt, which alternative claim takes its place, and which open questions follow. Do not treat the abduction as proven — flag it as the working hypothesis for the next iteration.
"""
    else:
        block += """
The outcome supported the prediction. State explicitly which claim is now more strongly backed by the evidence and resist over-extrapolating: the same condition will not be re-tested, so prefer narrow, well-supported claims over sweeping generalisations.
"""
    return block


def get_task(history, understanding):
    prompt = """Your task is to update your understanding of the system based on the latest experimental result and the reflection on it. Make the update explicit: which prior claim changes, which is reinforced, which open questions emerge.

Whenever possible, ground each claim in a specific physical/chemical mechanism relevant to this system (e.g. surface adatom mobility, cation/anion ordering thermodynamics, defect formation, volatility of a component, phase stability, kinetics vs equilibrium). When evidence lets you, commit to which mechanism you believe is dominant in the current regime - do not merely enumerate possibilities. If the data do not yet distinguish between mechanisms, say so explicitly and flag what observation would discriminate them.
"""
    reflection_block = get_reflection_block(history)
    if reflection_block:
        prompt += "\n" + reflection_block

    if len(understanding) > 0:
        prev = understanding[-1]
        prompt += f"""
Your previous understanding is as follows:
{prev['understanding']}
"""
    else:
        prompt += """
This is your first time to describe your understanding. Please describe your understanding of the system based on the experimental result, and (if a reflection block is available above) incorporate its judgement.
"""
    return prompt


def get_output_format():
    return """- Output format: A JSON object with the following structure:
  {"understanding": "<understanding>"}
- Only output the JSON object. Do not include any other text.
- "understanding" should be a string formatted in Markdown.
"""


def get_prompt(inputs, history, understanding):
    experimental_setup = get_experimental_setup(inputs)
    history_str = get_history_string(history, inputs)
    task = get_task(history, understanding)
    output_format = get_output_format()

    prompt = f"""=== Experimental setup ===
{experimental_setup}
"""
    prompt += f"""=== Observed data ===
{history_str}
"""
    prompt += f"""=== Your task ===
{task}
"""
    prompt += f"""=== Output requirements ===
{output_format}
"""
    return prompt


def get_chained_prompt(understanding):
    """Minimal understanding-update turn when chained off the reflection turn.

    Assumes the LLM already has the experimental setup, the prior understanding,
    the hypothesis, the expected/actual targets, and the reflection
    (judgement + abduction) in its session memory from the proposal and
    reflection turns. We only need to ask for the update and re-state the
    output schema.
    """
    has_prior = bool(understanding)
    if has_prior:
        prior_note = (
            "Update the understanding you saw earlier (the most recent understanding in your "
            "context) to incorporate the reflection you just produced. Make the update explicit: "
            "which prior claim changes, which is reinforced, which open questions emerge. If your "
            "reflection produced an abduction, treat it as the candidate revision (working "
            "hypothesis), not as proven.\n\n"
            "Whenever possible, ground each claim in a specific physical/chemical mechanism "
            "relevant to this system. When evidence lets you, commit to which mechanism you "
            "believe is dominant in the current regime - do not merely enumerate possibilities. "
            "If the data do not yet distinguish between mechanisms, say so explicitly and flag "
            "what observation would discriminate them."
        )
    else:
        prior_note = (
            "This is the first understanding you produce. Describe your understanding of the "
            "system based on the experimental setup, the result you just saw, and the reflection "
            "you just produced. Whenever possible, ground each claim in a specific "
            "physical/chemical mechanism relevant to this system rather than only a "
            "phenomenological observation."
        )
    return f"""{prior_note}

Output requirements:
- Output a single JSON object: {{"understanding": "<understanding>"}}
- Only output the JSON object. Do not include any other text.
- "understanding" should be a string formatted in Markdown.
"""
