"""Packaging hook: ship the configuration files with the package.

``configs/`` stays at the top of the source tree (that is where the evidence rules and the
documentation point); a built wheel carries a copy as ``basp/_configs``, which
``basp.evidence.CONFIG_DIR`` uses outside a checkout.  Everything else is in pyproject.toml.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithConfigs(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent / "configs"
        target = Path(self.build_lib) / "basp" / "_configs"
        target.mkdir(parents=True, exist_ok=True)
        files = sorted(source.glob("*.json"))
        if not files:
            raise RuntimeError(f"no configuration files in {source}: refusing to build an unusable package")
        for f in files:
            self.copy_file(str(f), str(target / f.name))


setup(cmdclass={"build_py": BuildWithConfigs})
