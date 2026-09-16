# SynAgent

SynAgent is a closed-loop experimental planning agent for autonomous thin-film synthesis, driven by multimodal LLMs. It builds an understanding of the synthesis process through a verify-falsify scheme: each proposal comes with a hypothesis and an expected outcome, alternating between conditions predicted to succeed (*verify*) and conditions predicted to fail (*falsify*). After each measurement (XRD patterns and SEM images), the agent checks the prediction and revises its understanding. Analysis routines for new data are generated on the fly as reusable skills.

This repository contains the code used in the paper *Hypothesis-Driven Autonomous Materials Synthesis with Multimodal LLM Agents*.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{takahara2026synagent,
  title   = {Hypothesis-Driven Autonomous Materials Synthesis with Multimodal LLM Agents},
  author  = {Takahara, Izumi and Nishio, Kazunori and Aiba, Akira and Kobayashi, Shigeru
             and Nakajima, Takao and Hitosugi, Taro and Mizoguchi, Teruyasu},
  year    = {2026},
  note    = {Submitted}
}
```

## License

This project is released under the [MIT License](LICENSE).
