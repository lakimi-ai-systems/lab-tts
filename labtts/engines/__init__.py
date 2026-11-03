"""TTS engines available to the experiments, by the name used in config.yaml.

To add one: implement labtts.engines.base.Engine in a new module and register it below. Modules are imported only
when their engine is used, so an engine's extra dependencies are only needed to run that engine.
"""
import importlib

from labtts.engines.base import Engine, Synthesis
from labtts.voices import Voice

# engine name -> (module, class, extra constructor arguments)
REGISTRY = {
    "qwen3-tts-0.6b": ("labtts.engines.qwen3_tts", "Qwen3TTSEngine", {"model_size": "0.6b"}),
    "qwen3-tts-1.7b": ("labtts.engines.qwen3_tts", "Qwen3TTSEngine", {"model_size": "1.7b"}),
    "audio8-air": ("labtts.engines.audio8", "Audio8Engine", {}),
    "ming-tts-0.5b": ("labtts.engines.ming_tts", "MingTTSEngine", {}),  # own environment
    "qwen3-tts-1.7b-voicedesign": ("labtts.engines.qwen3_tts", "Qwen3VoiceDesignEngine", {}),
}

# Engines that only run in their own environment (see "Adding an engine" in the README): a run without --engines
# leaves them out, and they are generated with --generate-only from that environment.
OWN_ENVIRONMENT = {"ming-tts-0.5b"}


def load_engine(name: str, voice: Voice | None = None) -> Engine:
    if name not in REGISTRY:
        raise KeyError(f"Unknown engine '{name}'. Registered engines: {', '.join(REGISTRY)}")
    module_name, class_name, arguments = REGISTRY[name]
    engine_class = getattr(importlib.import_module(module_name), class_name)
    return engine_class(name=name, voice=voice, **arguments)


__all__ = ["Engine", "Synthesis", "REGISTRY", "OWN_ENVIRONMENT", "load_engine"]
