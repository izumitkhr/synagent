"""LLM-driven generation of measurement-analysis skills.

Given a MaiML measurement file from any characterization technique (XRD,
SEM/TEM, XPS, Raman, impedance spectroscopy, ...), this module generates a
self-contained skill that reads such a file and produces an analysis result.
The technique is inferred at runtime from the MaiML file and the experiment's
`target_variable`, never hard-coded.

Skill contract: analyze() returns a JSON-serializable object (a bare float is
the trivially-scalar case) and writes it to result.json. Which skill (and
which field of its result) defines the optimization target is decided by the
campaign config (target_variable.skill/.field), not by the skill itself.

    skills/<skill_name>/
    ├── SKILL.md           ← YAML frontmatter + description (markdown)
    └── analyze.py         ← analyze(maiml_path, ref_dir, output_dir)

Generated skills are written under skills/ once they pass the self-test.
"""

import contextlib
import datetime
import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import traceback
import xml.etree.ElementTree as ET
from importlib import util as importlib_util
from pathlib import Path

import yaml
from pydantic import BaseModel

from synagent.core.llm import call_llm, _image_data_url
from synagent.main import load_inputs_yaml
from synagent.preprocess.preprocess import REF_DIR

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
SKILLS_DIR = _HERE / "skills"


# --------------------------------------------------------------------------- #
# MaiML parsing (technique-agnostic)
# --------------------------------------------------------------------------- #

def _local(tag):
    """Strip the XML namespace from a tag, e.g. '{ns}content' -> 'content'."""
    return tag.rsplit("}", 1)[-1]


def _child_text(elem, child_localname):
    """Text of the first direct child whose local tag matches, else None."""
    for c in elem:
        if _local(c.tag) == child_localname:
            text = (c.text or "").strip()
            return text or None
    return None


# Property keys that carry no analytical value (identifiers / hashes / paths).
_PARAM_SKIP = re.compile(r"uuid|guid|hash|uri|^id$", re.IGNORECASE)


def _summarize_maiml(maiml_path, max_arrays=8, sample_n=8, max_params=60):
    """Technique-agnostic summary of a MaiML measurement file.

    MaiML records measured data as self-describing
    ``<content key=.. axis=.. units=.. size=..>`` arrays and measurement
    conditions as ``<property key=..><value>..</value></property>`` pairs, with
    the instrument and the measurement protocol named near the top of the
    document. This walks those generically — without assuming a fixed tree
    layout — so the same summary works for XRD / TEM / SEM / XPS / Raman /
    impedance / ... files.
    """
    tree = ET.parse(maiml_path)
    root = tree.getroot()

    instrument = vendor = None
    technique_hints = []
    for elem in root.iter():
        ln = _local(elem.tag)
        if ln == "instrument" and instrument is None:
            instrument = _child_text(elem, "name")
        elif ln == "vendor" and vendor is None:
            vendor = _child_text(elem, "name")
        elif ln in ("protocol", "method"):
            name = _child_text(elem, "name")
            if name and name not in technique_hints:
                technique_hints.append(name)

    # Self-describing measured-data arrays.
    arrays = []
    for elem in root.iter():
        if _local(elem.tag) != "content":
            continue
        nums = []
        for child in elem:
            if _local(child.tag) == "value" and child.text:
                for tok in child.text.split():
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        pass
        info = {
            "key": elem.get("key") or elem.get("axis"),
            "units": elem.get("units"),
            "size": elem.get("size") or (len(nums) if nums else None),
        }
        if nums:
            info["min"] = round(min(nums), 4)
            info["max"] = round(max(nums), 4)
            info["sample_head"] = [round(v, 4) for v in nums[:sample_n]]
        arrays.append(info)
        if len(arrays) >= max_arrays:
            break

    # Scalar measurement conditions (key -> value, with units when present).
    params = {}
    for elem in root.iter():
        if _local(elem.tag) != "property":
            continue
        key = elem.get("key")
        if not key or _PARAM_SKIP.search(key):
            continue
        val = _child_text(elem, "value")
        if not val or len(val) > 80:
            continue
        units = elem.get("units")
        params[key] = f"{val} {units}".strip() if units else val
        if len(params) >= max_params:
            break

    return {
        "instrument": instrument,
        "vendor": vendor,
        "technique_hints": technique_hints,
        "data_arrays": arrays,
        "conditions": params,
    }


_IMAGE_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
# Formats the vision API accepts as-is; others are converted to PNG.
_VISION_NATIVE_EXTS = (".png", ".jpg", ".jpeg")


def _referenced_image_paths(maiml_path, max_images=2):
    """Image files referenced by <uri> entries, resolved next to the MaiML."""
    root = ET.parse(maiml_path).getroot()
    out = []
    for elem in root.iter():
        if _local(elem.tag) != "uri" or not elem.text:
            continue
        text = elem.text.strip()
        if text.lower().endswith(_IMAGE_EXTS):
            p = Path(maiml_path).parent / text
            if p.is_file():
                out.append(p)
        if len(out) >= max_images:
            break
    return out


def _as_vision_files(image_paths, tmp_dir):
    """Return vision-API-ready copies (convert BMP/TIFF to PNG in tmp_dir)."""
    from PIL import Image
    vision = []
    for p in image_paths:
        if p.suffix.lower() in _VISION_NATIVE_EXTS:
            vision.append(p)
        else:
            dst = Path(tmp_dir) / (p.stem + ".png")
            Image.open(p).save(dst)
            vision.append(dst)
    return vision


# --------------------------------------------------------------------------- #
# Reference / skill catalog helpers
# --------------------------------------------------------------------------- #

def _read_ref_files(ref_dir):
    """Return {filename: full text content} for each .txt in ref_dir.

    Reference files are optional and technique-dependent (e.g. XRD nominal-peak
    lists, instrument calibration tables). The directory may be empty.
    """
    contents = {}
    if not ref_dir or not Path(ref_dir).exists():
        return contents
    for p in sorted(Path(ref_dir).glob("*.txt")):
        contents[p.name] = p.read_text(encoding="utf-8-sig", errors="replace")
    return contents


def _read_skill_metadata(skill_dir):
    """Read SKILL.md frontmatter, returning a flat dict.

    Agent Skills convention: `name` and `description` at top level, domain
    fields (technique, material, ..., outputs, applicability) nested under
    `metadata:`. The nested mapping is flattened into the returned dict.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    m = re.match(r"---\n(.*?)\n---", skill_md.read_text(encoding="utf-8"), re.DOTALL)
    if not m:
        return None
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None
    nested = meta.pop("metadata", None)
    if isinstance(nested, dict):
        meta.update(nested)
    return meta


def _list_skills_with_metadata():
    """Return [(skill_dir, metadata_dict), ...] for each skill under SKILLS_DIR."""
    if not SKILLS_DIR.exists():
        return []
    out = []
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        meta = _read_skill_metadata(d)
        if meta is None:
            continue
        out.append((d, meta))
    return out


def _list_existing_skills():
    """Flat-dict view of the library, for the generation prompt's dedup list."""
    return [{"name": d.name, **meta} for d, meta in _list_skills_with_metadata()]


# --------------------------------------------------------------------------- #
# Skill generation prompt
# --------------------------------------------------------------------------- #

SYSTEM_INSTRUCTION = """You are an analysis SKILL generator for an autonomous (closed-loop) materials science laboratory.
Every experiment produces ONE MaiML measurement file. Depending on the experiment, that file may come from ANY characterization technique — X-ray diffraction (XRD), electron microscopy (TEM/SEM/STEM), X-ray photoelectron spectroscopy (XPS), Raman/IR spectroscopy, AC impedance spectroscopy, profilometry, etc. Given the experiment context and one representative MaiML file, produce a small self-contained Python SKILL that reads such a MaiML file and produces an analysis result.

UNIFIED CONTRACT: every skill extracts structured information from the measurement. `analyze()` returns a JSON-serializable object (typically a dict) and writes it to result.json. When the measurement quantifies a clear scalar metric (e.g. the experiment's `target_variable`), expose that metric under a clear, documented key of the returned dict — the closed-loop campaign config (inputs.yaml `target_variable.skill` / `.field`) selects which skill and which field feed the optimizer. The skill itself does NOT decide that and does NOT write result.txt.

HARD RULE - the skill must analyze the PROVIDED measurement: infer the technique from the provided MaiML file and build the skill for THAT technique. If the provided measurement does not quantify `target_variable` (e.g. the target is an XRD peak ratio but the provided file is an SEM image), extract complementary information from the provided measurement instead (e.g. film morphology statistics). NEVER produce a skill for a different technique than the provided file, and NEVER finalize a skill that merely returns its sentinel/failure value on the provided file.

## HOW TO APPROACH IT

1. Infer the measurement technique from the MaiML `instrument`, `protocol`/`method` names, and the data arrays it contains (each `<content>` array is self-labeled with key/axis/units). Do NOT assume XRD — read what the file actually is.
2. Read the experiment's `target_variable` in inputs.yaml — when the provided measurement quantifies it, that defines exactly what scalar to expose (units/scale included); otherwise it tells you what complementary information is useful to extract.
3. Choose the standard analysis method for that technique and goal, and implement it deterministically.

## OUTPUT CONTRACT

Return JSON with these fields:
- "skill_name": snake_case identifier under 40 chars, following `<material>_<metric-or-feature>_<technique>`, specific enough to disambiguate similar skills (append the instrument when a material+metric+technique combo is otherwise ambiguous). Examples across techniques: "lco_peak_ratio_xrd", "lagp_ionic_conductivity_eis", "ito_grain_size_sem", "tio2_anatase_fraction_raman", "lco_li_co_ratio_xps"
- "skill_md": full SKILL.md as a string with YAML frontmatter, then a markdown body. Frontmatter follows the Agent Skills convention: `name` and `description` at top level, and ALL domain fields nested under a `metadata:` mapping with keys technique, material, substrate, instrument, metric, outputs, applicability. `technique` is the characterization method (e.g. "XRD", "AC impedance spectroscopy", "SEM"). `outputs` is a short description of the fields of the returned JSON object and their meaning (e.g. "value: I(003)/I(006) peak-height ratio; peaks: matched peak list"). `applicability` is a short free-form note on when this skill applies (e.g. valid data range, required signal, instrument family). Use null/"unknown" for fields that do not apply to the technique. Example frontmatter:
  ---
  name: lagp_ionic_conductivity_eis
  description: <one sentence>
  metadata:
    technique: AC impedance spectroscopy
    material: LAGP
    substrate: null
    instrument: <instrument name>
    metric: ionic conductivity
    outputs: "value: log10 conductivity in S/cm; resistance_ohm: fitted resistance"
    applicability: <one sentence>
  ---
- "analyze_py": complete Python source for analyze.py

## analyze.py requirements

- Define `def analyze(maiml_path: str, ref_dir: str, output_dir: str)`
- Return a JSON-serializable object (typically a dict; a bare float is acceptable for a trivially-scalar analysis) and write it to `{output_dir}/result.json` (json.dump with indent, ensure_ascii=False). Do NOT write result.txt - the closed-loop orchestrator derives it from your result via the campaign config.
- Parse the MaiML GENERICALLY: locate `<content>` data arrays by their `key`/`axis` attribute and read `<property>` conditions by their `key`, rather than by hardcoded element indices — element positions differ between instruments and techniques.
- Allowed imports: numpy, scipy (signal, interpolate, optimize, stats, ndimage), matplotlib, PIL (Pillow), xml.etree.ElementTree, pandas, math, os, re, pathlib, logging, json, sys
- Handle edge cases (no signal / required feature absent / weak data) WITHOUT raising: return a clearly-marked failure object (e.g. {"status": "no_signal", "value": 0.001}). Keep any scalar metric field present with a small sentinel value (e.g. 0.001) so downstream extraction does not break.
- Be deterministic for the same input
- Do NOT print to stdout; use logging.getLogger(__name__) if needed
- Avoid matplotlib.pyplot.show(); save figures with savefig instead
- Close matplotlib figures (plt.close(fig)) so the function can be re-called safely

## Domain guidance

- XRD (thin-film 2θ scans): reference file(s) in ref_dir list nominal 2θ positions for expected reflections (often with h, k, l). Depending on the sample and instrument, you may need to account for some of the following — judge which are relevant:
  - Peak shift: peaks can shift from nominal (strain, thermal expansion, sample-height offsets that move all peaks together), so a tolerance window (~1° in 2θ) with nearest-peak matching is usually needed, even for weak or asymmetric peaks.
  - Kβ ghosts: the source's weaker Kβ line can leave a "ghost" of each strong reflection at a predictable lower 2θ (from the Kα/Kβ wavelengths via Bragg's law); consider excluding those positions.
  - Substrate peaks: sharp, intense substrate reflections (and their Kβ ghosts) may need excluding so they are not assigned to the film.
  - Weak reflections: a required reflection may be intrinsically weak or suppressed by texture (thin films do not follow powder intensities), so avoid discarding it just for being small.
  - Background: a local background may need subtracting before measuring peak heights; thin-film background can be high or sloping (e.g. sample fluorescence).
  - Sentinel: consider the sentinel when no peak exists anywhere inside a required reflection's window (amorphous / wrong phase).
  - Other effects may also matter (e.g. Kα1/Kα2 splitting at high angle, impurity phases, detector saturation on very strong peaks).
- AC impedance spectroscopy: identify the frequency / Z' / Z'' (real / imaginary impedance) arrays; extract resistance from the Nyquist semicircle (e.g. low-frequency real-axis intercept or equivalent-circuit fit) and convert to conductivity using the provided cell geometry / electrode conditions; report in the units/scale that `target_variable` requests (e.g. log10(σ) if asked).
- Imaging (SEM/TEM/STEM): when the metric is a morphology statistic (grain/particle size, coverage, roughness), derive it from the relevant data arrays or embedded measurements; use any scale/calibration in the MaiML or ref_dir. The actual image file (e.g. a .bmp) is usually NOT embedded in the MaiML but referenced by a relative `<uri>` (e.g. "./sem_xxx.bmp"); resolve it against the MaiML file's own directory and read it with PIL / matplotlib. Sidecar files referenced by other `<uri>` entries (e.g. an instrument-metadata .txt) may hold pixel-size / magnification calibration.
- For any technique, use reference or calibration files in ref_dir when relevant; ref_dir may be empty.

Return ONLY valid JSON (no markdown fences around the JSON itself). Strings inside JSON should use escaped newlines.
"""


AGENTIC_NOTE = """## INTERACTIVE VERIFICATION (run_python tool available)

You have a `run_python(code)` tool that executes Python on THIS machine. Predefined variables in its namespace: `maiml_path` (the real MaiML file), `ref_dir`, and `output_dir` (a fresh temp dir). It returns captured stdout/stderr (and a traceback on error); use print() to inspect values.

Do NOT write analyze.py blind. First USE the tool to validate your approach against the real data, and iterate until it is correct:
- Parse the MaiML and confirm you locate the right `<content>` arrays (by key/axis) and conditions (by key).
- Run your candidate analysis and PRINT intermediate results to check them — e.g. that identified features/peaks land where expected, values are physically plausible (not the sentinel/failure value), and units/scale match the experiment's target_variable.
- Verify the final analyze() actually writes result.json and that re-running gives the same result (determinism).

Only once you have confirmed correctness by running code, return the final JSON (skill_name, skill_md, analyze_py). Never ask the user anything — resolve every uncertainty by running code."""


def build_prompt(inputs_yaml, maiml_summary, maiml_xml, ref_files,
                 existing_skills, previous_attempt=None, previous_error=None,
                 agentic=False, extra_request=None, attached_images=None):
    parts = [SYSTEM_INSTRUCTION]
    if agentic:
        parts.append(AGENTIC_NOTE)
    if attached_images:
        names = ", ".join(attached_images)
        parts.append(
            "## ATTACHED IMAGE(S)\n"
            f"The image file(s) referenced by the MaiML ({names}) are attached "
            "to this message for VISUAL inspection. Look at them before "
            "designing the analysis: identify the feature types actually "
            "present (grains, particles, cracks, droplets, flat texture, "
            "gradients, charging artifacts) and choose methods/thresholds "
            "accordingly. Your analyze.py must still read the image from disk "
            "itself - the attachment is only for your design-time judgement."
        )
    if extra_request:
        parts.append(
            "## REQUEST FROM THE EXPERIMENT AGENT\n"
            "The closed-loop experiment agent asked for this skill while "
            "reflecting on a measurement:\n"
            f"{extra_request}\n"
            "Fulfil this request, subject to every rule above (especially the "
            "HARD RULE and the unified contract)."
        )
    parts.append("## EXPERIMENT CONTEXT")

    parts.append("### inputs.yaml (experiment context; see the rules above for "
                 "how `target_variable` relates to your output)\n```yaml\n" +
                 yaml.safe_dump(inputs_yaml, allow_unicode=True, sort_keys=False) +
                 "```")
    parts.append("### MaiML summary (technique inferred from instrument/protocol/"
                 "data arrays)\n```json\n" +
                 json.dumps(maiml_summary, indent=2, ensure_ascii=False) +
                 "\n```")
    if agentic:
        # The model can read the files itself via run_python, so we do not paste
        # the (often huge) MaiML XML or reference contents — only point to them.
        parts.append(
            "### MaiML file — NOT pasted here (to save context). The full file is "
            "on disk at the `maiml_path` predefined for run_python — the SAME file "
            "passed to analyze() at runtime. It is namespaced XML; read and parse "
            "it yourself with run_python. The summary above orients you to its "
            "instrument, technique, and data arrays."
        )
        names = ", ".join(ref_files) if ref_files else "(none)"
        parts.append(
            "## REFERENCE FILES (optional, in ref_dir/) — NOT pasted here. Read "
            "them from `ref_dir` with run_python as needed; formats differ per "
            f"file, so infer each file's structure from its content.\n"
            f"Available files: {names}"
        )
    else:
        parts.append("### MaiML file (full XML — the SAME file that will be passed "
                     "to analyze() at runtime as maiml_path)\n```xml\n" +
                     maiml_xml + "\n```")

        parts.append("## REFERENCE FILES (optional, located in ref_dir/, full content; "
                     "may be empty)\n"
                     "These are the exact files that will be present in ref_dir at "
                     "runtime — analyze() may read them by name. Formats differ per "
                     "file, so infer each file's structure from its content below.")
        if ref_files:
            for name, text in ref_files.items():
                parts.append(f"### {name}\n```\n" + text.rstrip() + "\n```")
        else:
            parts.append("(none provided)")

    if existing_skills:
        parts.append("## EXISTING SKILLS (for de-duplication context)")
        for s in existing_skills:
            tech = s.get("technique", "?")
            parts.append(f"- {s['name']} [{tech}]: "
                         f"{s.get('description', '(no description)')}")

    if previous_attempt and previous_error:
        parts.append("## PREVIOUS ATTEMPT FAILED — fix it")
        parts.append("### Previous SKILL.md\n```markdown\n" +
                     previous_attempt.get("skill_md", "") + "\n```")
        parts.append("### Previous analyze.py\n```python\n" +
                     previous_attempt.get("analyze_py", "") + "\n```")
        parts.append("### Error / Traceback\n```\n" + previous_error + "\n```")
        parts.append("Return a corrected JSON. Keep what was correct; fix only what failed.")

    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Skill selection (routing)
# --------------------------------------------------------------------------- #

def _select_skill_via_llm(skills, maiml_path, inputs_yaml_rel, run_dir, llm_model):
    """Ask the LLM to choose a skill based on SKILL.md metadata + measurement context.

    Returns the chosen skill directory Path, or None if the LLM says nothing matches.
    """
    inputs_yaml = load_inputs_yaml(Path(run_dir) / inputs_yaml_rel)
    maiml_summary = _summarize_maiml(maiml_path)

    parts = [
        "You are routing an experimental measurement to the most appropriate "
        "analysis SKILL. Pick exactly one skill from the list below, or return "
        "null if none of the listed skills is a good fit for the current "
        "measurement (in which case a new skill will be generated). A skill is a "
        "good fit only when its technique AND target metric match the current "
        "measurement and experiment goal.",
        "## Available skills",
    ]
    for d, meta in skills:
        parts.append(
            f"### {d.name}\n"
            f"- description: {meta.get('description', '(no description)')}\n"
            f"- technique: {meta.get('technique')}\n"
            f"- material: {meta.get('material')}\n"
            f"- substrate: {meta.get('substrate')}\n"
            f"- instrument: {meta.get('instrument')}\n"
            f"- metric: {meta.get('metric')}\n"
            f"- outputs: {meta.get('outputs')}\n"
            f"- applicability: {meta.get('applicability')}"
        )

    parts.append("## Current measurement context")
    parts.append(
        "### inputs.yaml\n```yaml\n"
        + yaml.safe_dump(inputs_yaml, allow_unicode=True, sort_keys=False)
        + "```"
    )
    parts.append(
        "### MaiML summary\n```json\n"
        + json.dumps(maiml_summary, indent=2, ensure_ascii=False)
        + "\n```"
    )

    parts.append(
        "Respond with JSON: "
        "{\"skill_name\": <name from the list or null>, "
        "\"rationale\": \"<one sentence>\"}. "
        "Use null for skill_name only when no listed skill applies."
    )
    prompt = "\n\n".join(parts)

    response, _ = call_llm(prompt, llm_model=llm_model, format="json_object")
    if not isinstance(response, dict):
        logger.warning("Skill selection: non-dict response %r", response)
        return None
    name = response.get("skill_name")
    rationale = response.get("rationale", "")
    logger.info("Skill selection: %r (%s)", name, rationale)
    if not name:
        return None
    for d, _meta in skills:
        if d.name == name:
            return d
    logger.warning("Skill selection: LLM returned unknown skill name %r", name)
    return None


def select_or_generate_skill(maiml_path, inputs_yaml_rel, run_dir,
                             llm_model="gpt-5.5", agentic=False, max_tool_calls=20):
    """Find a skill matching the current measurement; generate a new one if none fits.

    With an existing library, the LLM picks a skill from the SKILL.md metadata
    or signals that a new one is needed. `agentic=True` lets generation use a
    tool-using loop bounded by `max_tool_calls`.
    """
    skills = _list_skills_with_metadata()

    if skills:
        selected = _select_skill_via_llm(skills, maiml_path, inputs_yaml_rel,
                                         run_dir, llm_model)
        if selected is not None:
            return selected
        logger.info("LLM said no existing skill fits — generating a new skill...")
    else:
        logger.info("No skills available — generating a new skill (LLM call)...")

    skill_dir, _ = generate_skill(run_dir, inputs_yaml_rel,
                                  llm_model=llm_model, agentic=agentic,
                                  max_tool_calls=max_tool_calls)
    return skill_dir


# --------------------------------------------------------------------------- #
# Skill execution
# --------------------------------------------------------------------------- #

def _load_analyze(skill_dir, mod_prefix="_skill"):
    """Import the skill's analyze.py and return its analyze() function."""
    spec = importlib_util.spec_from_file_location(
        f"{mod_prefix}_{skill_dir.name}", skill_dir / "analyze.py"
    )
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze


def run_skill(skill_dir, maiml_path, ref_dir, output_dir):
    """Run the skill's analyze() on a measurement file.

    Unified contract: returns the JSON-serializable result as-is (a bare float
    is the trivially-scalar case); analyze() writes result.json itself.
    """
    analyze = _load_analyze(skill_dir)
    return analyze(str(maiml_path), str(ref_dir), str(output_dir))


# --------------------------------------------------------------------------- #
# Skill generation
# --------------------------------------------------------------------------- #

def _sanitize_skill_name(name):
    return re.sub(r"[^a-z0-9_]", "_", name.strip().lower())[:40] or "skill"


def parse_response(raw):
    obj = raw if isinstance(raw, dict) else json.loads(raw)
    return _sanitize_skill_name(obj["skill_name"]), obj["skill_md"], obj["analyze_py"]


def write_skill(name, skill_md, analyze_py):
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        shutil.rmtree(skill_dir)
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    (skill_dir / "analyze.py").write_text(analyze_py, encoding="utf-8")
    return skill_dir


def test_skill(skill_dir, maiml_path, ref_dir):
    """Import the generated analyze.py and run analyze() on the sample MaiML."""
    try:
        analyze_fn = _load_analyze(skill_dir, mod_prefix="_generated")
    except Exception:
        return None, "While importing analyze.py:\n" + traceback.format_exc()

    with tempfile.TemporaryDirectory(prefix="skill_test_") as test_out:
        try:
            result = analyze_fn(str(maiml_path), str(ref_dir), test_out)
        except Exception:
            return None, "While running analyze():\n" + traceback.format_exc()

    if result is None:
        return None, "analyze() returned None"
    # Unified contract: a bare scalar or any JSON-serializable object.
    try:
        return float(result), None
    except (TypeError, ValueError):
        pass
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        return None, (
            "analyze() returned a value that is neither a float nor "
            f"JSON-serializable: {result!r}"
        )
    return result, None


# --------------------------------------------------------------------------- #
# Agentic generation: let the LLM run code to verify/refine its analysis
# --------------------------------------------------------------------------- #

class SkillSpec(BaseModel):
    """Structured output contract of the skill-generation agent."""
    skill_name: str
    skill_md: str
    analyze_py: str


def run_python_snippet(code, namespace, timeout=90, max_chars=8000):
    """Execute LLM-authored `code` with the given variables injected.

    Runs in a daemon thread, captures stdout+stderr (and any traceback), and
    truncates the returned text. The timeout is soft: a CPU-bound thread
    keeps running in the background until the process exits. Carries the
    same trust level as running a generated analyze.py.
    """
    box = {}

    def worker():
        buf = io.StringIO()
        ns = dict(namespace)
        ns["__name__"] = "__run_python__"
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(code, "<run_python>", "exec"), ns)
        except Exception:
            buf.write("\n[EXCEPTION]\n" + traceback.format_exc())
        box["text"] = buf.getvalue()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return f"[TIMEOUT after {timeout}s — code did not finish; avoid long loops]"
    text = box.get("text", "") or "[no output produced]"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[...truncated {len(text) - max_chars} chars]"
    return text


def _run_python_tool(code, maiml_path, ref_dir, timeout=90, max_chars=8000):
    """Skill-generation flavor of run_python_snippet (fresh temp output_dir)."""
    with tempfile.TemporaryDirectory(prefix="skill_explore_") as td:
        ns = {
            "maiml_path": str(maiml_path),
            "ref_dir": str(ref_dir),
            "output_dir": td,
        }
        return run_python_snippet(code, ns, timeout=timeout, max_chars=max_chars)


# In-memory transcript of the generation in progress, persisted afterwards as
# {skill_dir}/generation.log so the skill's provenance stays with the skill.
_GEN_SINK = None


def _agentic_log(text):
    """Append `text` to the in-memory transcript and, when
    SYNAGENT_PROMPT_LOG_DIR is set, to skill_gen_agentic.log in that dir."""
    if _GEN_SINK is not None:
        _GEN_SINK.append(text if text.endswith("\n") else text + "\n")
    log_dir = os.environ.get("SYNAGENT_PROMPT_LOG_DIR")
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "skill_gen_agentic.log"), "a",
              encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _write_generation_log(skill_dir, lines, status):
    """Persist the generation transcript as {skill_dir}/generation.log.

    A write failure must never break the generation itself.
    """
    try:
        path = Path(skill_dir) / "generation.log"
        path.write_text("".join(lines) + f"\n--- {status} ---\n",
                        encoding="utf-8")
        logger.info("Generation transcript saved: %s", path)
    except OSError as e:
        logger.warning("Could not write generation.log to %s: %s", skill_dir, e)


# When this many tool calls (or fewer) remain, start nudging the model to wrap
# up so it lands on a finalized skill instead of being cut off mid-exploration.
_SOFT_LANDING_CALLS = 3


def _agentic_generate(prompt, maiml_path, ref_dir, llm_model, max_tool_calls=20,
                      log_tag=None, image_paths=None):
    """Generate a skill via an Agents SDK tool-using loop (code self-verification).

    An Agent with the run_python tool explores the real MaiML and iterates on
    its analysis; ``output_type=SkillSpec`` makes the SDK enforce the final
    structured output. As the tool budget nears exhaustion, a soft-landing
    note is appended to the tool output so the model converges; once
    exhausted, further tool calls are refused. Raises MaxTurnsExceeded if the
    model still does not finalize.
    """
    from agents import Agent, Runner, function_tool, set_tracing_disabled
    from synagent.core.llm import configure_agents_default_client

    set_tracing_disabled(True)
    configure_agents_default_client()
    state = {"step": 0}

    @function_tool
    def run_python(code: str) -> str:
        """Execute Python on the host to explore the MaiML and validate/iterate on
        your analysis before finalizing analyze.py. Predefined variables:
        maiml_path (str), ref_dir (str), output_dir (str, a fresh temp dir).
        Returns captured stdout/stderr (plus a traceback on error). Use print()
        to inspect values. Avoid long-running loops; execution is time-limited.

        Args:
            code: Python source to execute.
        """
        state["step"] += 1
        step = state["step"]
        if step > max_tool_calls:
            return ("[Tool budget exhausted - this call was NOT executed. "
                    "Return the final skill output now.]")
        out = _run_python_tool(code, maiml_path, ref_dir)
        logger.info("agentic run_python (step %d): %d chars out", step, len(out))
        _agentic_log(
            f"\n--- run_python (step {step}) CODE ---\n{code}\n"
            f"--- run_python (step {step}) OUTPUT ({len(out)} chars) ---\n{out}\n"
        )
        remaining = max_tool_calls - step
        if 0 <= remaining <= _SOFT_LANDING_CALLS:
            note = (
                f"[SYSTEM NOTE] You have {remaining} run_python call(s) left "
                "before tool access is cut off. Wrap up verification now and "
                "return the final skill output (skill_name, skill_md, "
                "analyze_py) within the remaining calls; do not open new "
                "lines of investigation."
            )
            _agentic_log(f"\n--- SOFT-LANDING NOTE (step {step}) ---\n{note}\n")
            out += f"\n\n{note}"
        return out

    agent = Agent(
        name="skill_generator",
        model=llm_model,
        tools=[run_python],
        output_type=SkillSpec,
    )
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    image_note = (f"\n[attached images: "
                  f"{', '.join(str(p) for p in image_paths)}]" if image_paths else "")
    _agentic_log(
        f"\n{'=' * 78}\n[{stamp}] AGENTIC SKILL GENERATION  tag={log_tag} "
        f"model={llm_model} maiml={Path(maiml_path).name}\n{'=' * 78}\n"
        f"\n--- INITIAL PROMPT ---{image_note}\n{prompt}\n"
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
    # max_turns counts model invocations: 1 initial + one per tool round + final.
    result = Runner.run_sync(agent, llm_input, max_turns=max_tool_calls + 2)
    spec = result.final_output
    raw = json.dumps(spec.model_dump(), ensure_ascii=False)
    _agentic_log(f"\n--- FINAL OUTPUT ---\n{raw}\n")
    return spec.model_dump()


def generate_skill(run_dir, input_yaml_rel, max_retries=2, llm_model="gpt-5.5",
                   agentic=True, max_tool_calls=20, maiml_path=None,
                   inputs_yaml=None, extra_request=None, force_name=None):
    """Generate a skill from a representative MaiML file.

    ``maiml_path`` defaults to the single *.maiml in run_dir. ``inputs_yaml``
    (dict) defaults to loading ``run_dir/input_yaml_rel``. ``extra_request``
    is a free-form request from the experiment agent embedded in the prompt.
    ``force_name`` overrides the LLM-chosen skill_name (pinned target skill).
    """
    run_dir = Path(run_dir)
    if maiml_path is None:
        maimls = sorted(run_dir.glob("*.maiml"))
        if not maimls:
            raise FileNotFoundError(f"No .maiml file in {run_dir}")
        if len(maimls) > 1:
            raise RuntimeError(f"Multiple .maiml files in {run_dir}: {[m.name for m in maimls]}")
        maiml_path = maimls[0]
    else:
        maiml_path = Path(maiml_path)
        if not maiml_path.is_file():
            raise FileNotFoundError(f"maiml_path does not exist: {maiml_path}")
    logger.info("Skill generation using MaiML: %s", maiml_path.name)

    if inputs_yaml is None:
        inputs_yaml = load_inputs_yaml(run_dir / input_yaml_rel)
    maiml_summary = _summarize_maiml(maiml_path)
    logger.info("MaiML technique hints: %s (instrument=%s)",
                maiml_summary.get("technique_hints"), maiml_summary.get("instrument"))
    maiml_xml = None if agentic else Path(maiml_path).read_text(encoding="utf-8")
    ref_files = _read_ref_files(REF_DIR)
    existing_skills = _list_existing_skills()

    logger.info("Skill generation mode: %s", "agentic" if agentic else "single-shot")
    existing_names = {s["name"] for s in existing_skills}
    created_this_run = set()
    skill_dir = None
    previous_attempt = None
    previous_error = None
    attempts = max_retries + 1

    global _GEN_SINK
    _GEN_SINK = sink = []

    # Attach images referenced by the MaiML (e.g. the SEM bitmap) as vision
    # input; BMP/TIFF are converted to PNG in a temp dir.
    vision_tmp = tempfile.mkdtemp(prefix="skill_gen_vision_")
    try:
        raw_images = _referenced_image_paths(maiml_path)
        vision_paths = _as_vision_files(raw_images, vision_tmp) if raw_images else []
        if vision_paths:
            logger.info("Attaching %d referenced image(s) for visual inspection.",
                        len(vision_paths))

        for i in range(attempts):
            logger.info("Skill generation attempt %d/%d", i + 1, attempts)
            prompt = build_prompt(inputs_yaml, maiml_summary, maiml_xml,
                                  ref_files, existing_skills,
                                  previous_attempt, previous_error,
                                  agentic=agentic, extra_request=extra_request,
                                  attached_images=[p.name for p in raw_images])
            if agentic:
                response = _agentic_generate(prompt, maiml_path, REF_DIR, llm_model,
                                             max_tool_calls=max_tool_calls,
                                             log_tag=f"attempt{i + 1}",
                                             image_paths=vision_paths or None)
            else:
                stamp = datetime.datetime.now().isoformat(timespec="seconds")
                _agentic_log(
                    f"\n{'=' * 78}\n[{stamp}] SINGLE-SHOT SKILL GENERATION  "
                    f"attempt={i + 1} model={llm_model} "
                    f"maiml={Path(maiml_path).name}\n{'=' * 78}\n"
                    f"\n--- PROMPT ---\n{prompt}\n"
                )
                response, _ = call_llm(prompt, llm_model=llm_model,
                                       format="json_object",
                                       image_paths=vision_paths or None)
                raw = (response if isinstance(response, str)
                       else json.dumps(response, ensure_ascii=False))
                _agentic_log(f"\n--- RESPONSE ---\n{raw}\n")
            name, skill_md, analyze_py = parse_response(response)
            if force_name:
                forced = _sanitize_skill_name(force_name)
                if forced != name:
                    logger.info("Overriding generated skill name %r -> %r",
                                name, forced)
                    name = forced

            # Never overwrite a pre-existing skill; feed a name collision back
            # as a failed attempt.
            if name in existing_names and name not in created_this_run:
                previous_attempt = {"skill_md": skill_md, "analyze_py": analyze_py}
                previous_error = (
                    f"skill_name '{name}' collides with an EXISTING skill; "
                    "overwriting is not allowed. Re-check the HARD RULE: the skill "
                    "must analyze the technique of the PROVIDED MaiML file. If it "
                    "genuinely does, give it a distinct, more specific name."
                )
                logger.warning("Attempt %d rejected: %s", i + 1, previous_error)
                _agentic_log(f"\n--- ATTEMPT {i + 1} REJECTED (name collision) "
                             f"---\n{previous_error}\n")
                continue
            created_this_run.add(name)
            skill_dir = write_skill(name, skill_md, analyze_py)

            value, error = test_skill(skill_dir, maiml_path, REF_DIR)
            if error is None:
                logger.info("Skill generation OK on attempt %d. Result: %s",
                            i + 1, value)
                _write_generation_log(
                    skill_dir, sink,
                    f"REGISTERED '{skill_dir.name}' — self-test OK on attempt "
                    f"{i + 1}/{attempts}, result: {str(value)[:500]}",
                )
                return skill_dir, value
            logger.warning("Attempt %d failed:\n%s", i + 1, error)
            _agentic_log(f"\n--- ATTEMPT {i + 1} SELF-TEST FAILED ---\n{error}\n")
            previous_attempt = {"skill_md": skill_md, "analyze_py": analyze_py}
            previous_error = error
    finally:
        shutil.rmtree(vision_tmp, ignore_errors=True)
        _GEN_SINK = None

    logger.error("Skill generation failed after %d attempts.", attempts)
    if skill_dir is not None:
        _write_generation_log(
            skill_dir, sink,
            f"FAILED after {attempts} attempt(s) — this dir holds the last "
            f"(broken) attempt",
        )
    return skill_dir, None
