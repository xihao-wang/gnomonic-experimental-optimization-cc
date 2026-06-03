"""Local StrongSORT option defaults for the gnomonic tracker package.

The original project kept these values in a repository-level ``opts.py`` that
parses command-line positional arguments at import time. That pattern is unsafe
inside this project because importing the tracker should not parse CLI args.

Only the fields used by the copied StrongSORT modules are defined here. Runtime
entry points can still mutate ``opt`` before constructing the tracker.
"""

from types import SimpleNamespace


opt = SimpleNamespace(
    # Base association options
    BoT=True,
    ECC=False,
    NSA=False,
    EMA=False,
    MC=False,
    woC=False,
    max_cosine_distance=0.4,
    nn_budget=100,
    min_confidence=0.6,
    nms_max_overlap=1.0,
    min_detection_height=0,

    # STM/LTM memory options
    short_memory_size=5,
    long_memory_size=30,
    memory_init_hits=10,
    long_memory_stride=1,
    memory_sim_threshold=0.8,
    memory_min_confidence=0.7,
    short_memory_gate=0.7,
    short_distance_weight=0.5,
    beta=0.8,
    k=3,
    ambiguity_distance_threshold=0.02,
    ambiguity_margin=0.003,
    match_conf_margin_scale=0.02,

    # Phase/truncation options kept for copied code compatibility
    phase_old_sim_threshold=0.45,
    phase_short_sim_threshold=0.75,
    phase_consistency_threshold=0.75,
    phase_patience=3,
    phase_min_long_memory=6,
    phase_min_short_memory=3,
    phase_reset_cooldown=5,

    # Feature switches
    enable_stm_ltm=True,
    enable_memory_init_control=True,
    enable_memory_matching=True,
    enable_topk_matching=True,
    enable_phase_truncation=False,
)
