"""Child process: render the large scene and report measured peak RSS."""

import os
import resource
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flyerdash import render  # noqa: E402
from tests.large_scene import DPI, build_scene  # noqa: E402


def main():
    output_path = sys.argv[1]
    project = build_scene()
    render(project, DPI, output_path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024  # Linux reports KiB, macOS reports bytes
    print(f"RSS_BYTES={peak}")


if __name__ == "__main__":
    main()
