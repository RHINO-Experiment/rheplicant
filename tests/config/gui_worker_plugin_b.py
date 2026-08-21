import os
from pathlib import Path

from jax import config

print("plugin-b stdout noise")
with Path(os.environ["RHEPLICANT_TEST_PLUGIN_TRACE"]).open(
    "a", encoding="utf-8"
) as stream:
    stream.write(f"b:x64={config.read('jax_enable_x64')}\n")
