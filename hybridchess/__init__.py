from __future__ import annotations

import importlib
import importlib.util
import os
import sys


def _is_colab() -> bool:
    try:
        import google.colab
        return os.path.isdir("/content")
    except Exception:
        return False


if _is_colab():
    _cfg_mod_name = ".config_colab"
    _profile = "colab"
else:
    _cfg_mod_name = ".config_local"
    _profile = "local"

_cfg_mod = importlib.import_module(_cfg_mod_name, __name__)
setattr(_cfg_mod, "CONFIG_PROFILE", _profile)
sys.modules[__name__ + ".config"] = _cfg_mod
