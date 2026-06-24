"""
setup.py — Legacy editable-install shim for Raspberry Pi OS / old pip.

Modern pip (>=21.3) supports PEP 660 and reads pyproject.toml directly for
editable installs.  Older pip on Raspberry Pi OS requires a setup.py.
This file simply delegates to setuptools using the same metadata as pyproject.toml.

PEP 517 build backend lives only in pyproject.toml: setuptools.build_meta
(not setuptools.backends.legacy — removed in newer setuptools).
"""
from setuptools import setup, find_packages

setup(
    name="smartbinocular",
    version="0.1.0",
    description="Real-time NIR + thermal fusion pipeline for night-vision binoculars (RPi4)",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Keep in sync with pyproject.toml [project] dependencies.
        "numpy>=1.26,<2",
        "opencv-python>=4.8,<4.13",
    ],
    entry_points={
        "console_scripts": [
            "smartbinocular=smartbinocular.main:main",
        ],
    },
    package_data={
        "smartbinocular": ["assets/*.json"],
    },
)
