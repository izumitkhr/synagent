def pick_mode(history, strategy="alternate"):
    """Pick the proposal mode for the next iteration.

    Strategies:
    - ``alternate`` (default): ``verify`` -> ``falsify`` -> ``verify`` -> ...,
      starting from ``verify`` when there is no prior history.
    - ``verify_only``: always return ``verify``.
    - ``falsify_only``: always return ``falsify``.
    """
    if strategy == "verify_only":
        return "verify"
    if strategy == "falsify_only":
        return "falsify"
    if strategy != "alternate":
        raise ValueError(f"Unknown mode strategy: {strategy!r}")
    if not history:
        return "verify"
    last_mode = history[-1].get("mode")
    if last_mode == "verify":
        return "falsify"
    return "verify"
