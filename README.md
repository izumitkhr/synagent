# SynAgent

SynAgent is a closed-loop experimental planning agent for autonomous thin-film synthesis, driven by multimodal LLMs. It builds an understanding of the synthesis process through a verify-falsify scheme: each proposal comes with a hypothesis and an expected outcome, alternating between conditions predicted to succeed (*verify*) and conditions predicted to fail (*falsify*). After each measurement (XRD patterns and SEM images), the agent checks the prediction and revises its understanding. Analysis routines for new data are generated on the fly as reusable skills.

## Installation

```bash
pip install -e .
cp .env.example .env   # set OPENAI_API_KEY
```

## Usage

```bash
python run.py --run_dir <RUN_DIR> --input_yaml <INSTRUCTION_FILE> --raw_data_dir <INSTRUMENT_DATA_DIR>
```

`run.py` is called once per closed-loop iteration. It reads the XRD `*.maiml` file in `run_dir`, computes the target value, and writes the next proposal to `run_dir/output.txt`. An example instruction file is given in `settings/1d.yaml`. Reference files used during skill generation (e.g. nominal XRD peak positions, as `.txt`) should be placed in `synagent/preprocess/ref/`. See `python run.py --help` for options.

## Platform

SynAgent is compatible with dLab, a digital laboratory that interconnects modular synthesis and measurement instruments via robotic sample transport and outputs all measurement data in the MaiML format. SynAgent reads XRD and SEM data from these MaiML files. The platform is described in:

```bibtex
@article{Nishio2025DigitalLaboratory,
  author    = {Nishio, Kazunori and Aiba, Akira and Takihara, Kei and Suzuki, Yota
               and Nakayama, Ryo and Kobayashi, Shigeru and Abe, Akira and Baba, Haruki
               and Katagiri, Shinichi and Omoto, Kazuki and Ito, Kazuki and Shimizu, Ryota
               and Hitosugi, Taro},
  title     = {A digital laboratory with a modular measurement system and standardized data format},
  journal   = {Digital Discovery},
  year      = {2025},
  volume    = {4},
  number    = {7},
  pages     = {1734--1742},
  publisher = {Royal Society of Chemistry},
  doi       = {10.1039/d4dd00326h},
  url       = {https://doi.org/10.1039/d4dd00326h}
}
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{takahara2026synagent,
  title         = {Hypothesis-Driven Autonomous Materials Synthesis with Multimodal LLM Agents},
  author        = {Takahara, Izumi and Nishio, Kazunori and Aiba, Akira and Kobayashi, Shigeru
                   and Nakajima, Takao and Hitosugi, Taro and Mizoguchi, Teruyasu},
  year          = {2026},
  eprint        = {2609.18598},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  url           = {https://arxiv.org/abs/2609.18598}
}
```

## License

This project is released under the [MIT License](LICENSE).
