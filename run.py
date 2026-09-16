import argparse
import glob
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

from synagent.preprocess.preprocess import preprocess
from synagent.main import main as run_inference

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # LLM / main inference
    parser.add_argument('--run_dir', type=str, default='default',
                        help='Working directory for all input/output files (output.txt, *.maiml, data/, history.json). "default" resolves to the project root.')
    parser.add_argument('--input_yaml', type=str, default='settings/1d.yaml',
                        help='YAML file (relative to run_dir) defining target/explanatory variables and grid spec; dim is read from the yaml.')
    parser.add_argument('--llm_model_proposal', type=str, default='gpt-5.5',
                        help='LLM model used to generate the next experiment proposal.')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--verify_only', dest='mode_strategy',
                            action='store_const', const='verify_only',
                            help='Always propose in verify mode (skip falsify branch). Mutually exclusive with --falsify_only.')
    mode_group.add_argument('--falsify_only', dest='mode_strategy',
                            action='store_const', const='falsify_only',
                            help='Always propose in falsify mode (skip verify branch). Mutually exclusive with --verify_only.')
    parser.set_defaults(mode_strategy='alternate')
    parser.add_argument('--debug', action='store_true',
                        help='Enable DEBUG-level logging.')
    parser.add_argument('--no_prompt_log', action='store_true',
                        help='Disable per-call prompt logging. By default, every LLM prompt and response is appended to {run_dir}/prompts/iterNN_{role}.log. Set the SYNAGENT_PROMPT_LOG_DIR env var to override the location; this flag wins over the env var.')
    # Session memory (persistent LLM conversation per run_dir)
    parser.add_argument('--session_memory',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Run proposal/reflection/understanding turns in a persistent LLM session '
                             'stored in {run_dir}/session.db, so recent SEM images from earlier iterations '
                             'stay visually available for comparison. Start a campaign with this setting '
                             'and keep it unchanged. Use --no_session_memory for the response_id-chained behavior.')
    parser.add_argument('--session_max_images', type=int, default=3,
                        help='Max number of most-recent SEM images replayed from the session on each call; '
                             'older images are replaced by a text placeholder (their sem_observation text remains). '
                             'All images stay stored in session.db, so raising this later restores them.')
    # Multimodal (SEM) linkage
    parser.add_argument('--raw_data_dir', type=str, default=None,
                        help='Instrument-side folder receiving the latest MaiML/bitmap files '
                             '(incl. SampleDelivery). Searched for the SEM MaiML/bitmap linked to '
                             'the XRD MaiML in run_dir. Required unless --no_sem is given.')
    parser.add_argument('--no_sem', action='store_true',
                        help='Skip the XRD->SEM linkage step (XRD-only).')
    # Skill generation
    parser.add_argument('--generate_skill', action='store_true',
                        help='Generate a new measurement-analysis SKILL via LLM from the measurement file in run_dir. '
                             'Skips preprocess and inference. Writes to synagent/preprocess/skills/<skill_name>/.')
    parser.add_argument('--max_skill_retries', type=int, default=2,
                        help='Max auto-retries on skill generation error (default 2 → 3 total attempts).')
    parser.add_argument('--skill_maiml', type=str, default=None,
                        help='With --generate_skill: explicit path to the representative MaiML file '
                             '(e.g. a SEM MaiML outside run_dir; its <uri>-referenced files must sit next to it). '
                             'Default: the single *.maiml in run_dir.')
    parser.add_argument('--llm_model_skill', type=str, default='gpt-5.5',
                        help='LLM model used for skill selection / generation.')
    return parser.parse_args()


def _resolve_run_dir(args):
    """Resolve --run_dir to an absolute path. Mirrors synagent.main's resolution."""
    if args.run_dir == 'default':
        return ROOT
    return os.path.abspath(args.run_dir)


def _has_measurement_input(run_dir):
    # Measurement input format is XRD MaiML (*.maiml).
    return len(glob.glob(os.path.join(run_dir, '*.maiml'))) > 0


def _history_needs_target(run_dir):
    history_path = os.path.join(run_dir, 'history.json')
    if not os.path.exists(history_path):
        return False
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return False
    return history[-1].get('target') is None


def run_measurement(args, run_dir):
    """Run the measurement/preprocess step.

    Returns ``(source_maiml, sem_png)``: the basename of the MaiML file used
    so the inference step can detect a stale / duplicated measurement, and
    the path of the fetched SEM image (or None when --no_sem).
    """
    output_dir = os.path.join(run_dir, 'data')
    sem_png = None
    maiml_files = glob.glob(os.path.join(run_dir, '*.maiml'))
    if len(maiml_files) > 1:
        names = ', '.join(os.path.basename(p) for p in sorted(maiml_files))
        raise RuntimeError(
            f'Multiple .maiml files found in {run_dir}: {names}. '
            f'Keep only one to disambiguate.'
        )
    source_maiml = os.path.basename(maiml_files[0])
    if not args.no_sem:
        # Fetch the SEM MaiML/bitmap linked to this XRD MaiML from the
        # instrument-side raw folder. Any linkage failure raises
        # LinkageError and aborts the iteration (no fallback by design).
        from synagent.preprocess.maiml_linkage import fetch_sem_image
        sem_files = fetch_sem_image(
            xrd_maiml_path=maiml_files[0],
            raw_dir=args.raw_data_dir,
            dest_dir=output_dir,
        )
        sem_png = sem_files['sem_png']
        logger.info('SEM image for vision input: %s', sem_png)
    target = preprocess(
        output_dir=output_dir,
        maiml_path=maiml_files[0],
        run_dir=run_dir,
        inputs_yaml_rel=args.input_yaml,
        llm_model_skill=args.llm_model_skill,
    )

    logger.info('Target value: %.8f', target)
    logger.info('Output: %s', output_dir)
    return source_maiml, sem_png


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run_dir = _resolve_run_dir(args)
    logger.info('run_dir: %s', run_dir)

    # Prompt logging precedence: --no_prompt_log > existing env var > default
    # ({run_dir}/prompts/). Set BEFORE the --generate-skill branch so the
    # agentic generation transcript (skill_gen_agentic.log) is captured there
    # too, not only for in-loop generation.
    if args.no_prompt_log:
        os.environ.pop('SYNAGENT_PROMPT_LOG_DIR', None)
    elif 'SYNAGENT_PROMPT_LOG_DIR' not in os.environ:
        os.environ['SYNAGENT_PROMPT_LOG_DIR'] = os.path.join(run_dir, 'prompts')

    if args.generate_skill:
        from synagent.preprocess.skill_generation import generate_skill
        skill_dir, ir = generate_skill(
            run_dir=run_dir,
            input_yaml_rel=args.input_yaml,
            max_retries=args.max_skill_retries,
            llm_model=args.llm_model_skill,
            maiml_path=args.skill_maiml,
        )
        if ir is None:
            logger.error('Skill generation failed. See %s for last attempt.', skill_dir)
        else:
            logger.info('Skill saved at: %s', skill_dir)
            if isinstance(ir, float):
                logger.info('Test result on sample MaiML: target=%.8f', ir)
            else:
                # Analysis skills return structured (JSON-serializable) data.
                logger.info('Test result on sample MaiML (analysis): %s', ir)
        return

    log_dir = os.environ.get('SYNAGENT_PROMPT_LOG_DIR')
    if log_dir:
        logger.info('Prompt log dir: %s', log_dir)

    has_input = _has_measurement_input(run_dir)
    needs_target = _history_needs_target(run_dir)

    if needs_target and not has_input:
        logger.warning(
            'Waiting for measurement: history.json expects a target but no *.maiml found. '
            'Place the *.maiml file in %s and re-run.', run_dir,
        )
        return

    if has_input and not args.no_sem and args.raw_data_dir is None:
        raise SystemExit(
            '--raw_data_dir is required to fetch the SEM image linked to the XRD '
            'measurement. Pass it, or use --no_sem to run XRD-only.'
        )

    source_maiml = None
    sem_png = None
    if has_input:
        source_maiml, sem_png = run_measurement(args, run_dir)
    else:
        logger.info('No measurement input found in %s — bootstrap mode (skipping preprocess).', run_dir)

    run_inference(args, source_maiml=source_maiml, sem_image=sem_png)


def do_nothing():
    """No-op entry point required by the instrument-side orchestration."""
    pass


if __name__ == '__main__':
    main()
