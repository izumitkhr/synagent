# SynAgent

SynAgent is a closed-loop experimental planning agent for autonomous thin-film synthesis, driven by multimodal LLMs. It builds an understanding of the synthesis process through a verify-falsify scheme: each proposal comes with a hypothesis and an expected outcome, alternating between conditions predicted to succeed (*verify*) and conditions predicted to fail (*falsify*). After each measurement (XRD patterns and SEM images), the agent checks the prediction and revises its understanding. Analysis routines for new data are generated on the fly as reusable skills.

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
