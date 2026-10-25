"""Describes the machine and software an evaluation run executed on."""

import os
import platform
from importlib import metadata

from pii_toolbox.core import RunEnvironment


def collect_environment() -> RunEnvironment:
    """Machine, operating system, Python runtime and the version of every installed library."""
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    }
    return RunEnvironment(
        machine=f"{platform.machine()}, {os.cpu_count()} cores, {platform.processor() or 'unknown processor'}",
        os=platform.platform(),
        runtime=f"{platform.python_implementation()} {platform.python_version()}",
        library_versions=dict(sorted(versions.items(), key=lambda item: item[0].lower())),
    )
