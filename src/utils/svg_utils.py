from __future__ import annotations

"""
Template for converting bitmap glyphs to SVG using potrace/autotrace.
"""

import os
import subprocess
from pathlib import Path
from typing import Iterable


def convert_folder_to_svg(
    input_dir: Path, output_dir: Path, tool: str = "potrace", extra_args: Iterable[str] = ()
) -> None:
    """
    Convert all bitmap files in input_dir to SVGs in output_dir using the given tool.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for png_path in sorted(input_dir.glob("*.png")):
        svg_path = output_dir / (png_path.stem + ".svg")
        cmd = []
        if tool == "potrace":
            # potrace -s input.png -o output.svg
            cmd = ["potrace", "-s", str(png_path), "-o", str(svg_path), *extra_args]
        elif tool == "autotrace":
            # autotrace input.png -output-file output.svg -output-format svg
            cmd = [
                "autotrace",
                str(png_path),
                "--output-file",
                str(svg_path),
                "--output-format",
                "svg",
                *extra_args,
            ]
        else:
            print(f"Unsupported tool: {tool}. Skipping {png_path.name}")
            continue

        try:
            subprocess.run(cmd, check=True)
            print(f"Converted {png_path.name} -> {svg_path.name}")
        except FileNotFoundError:
            print(f"Tool '{tool}' not found. Install it or adjust the command. Skipping.")
            break
        except subprocess.CalledProcessError as exc:
            print(f"Conversion failed for {png_path.name}: {exc}")


if __name__ == "__main__":
    # Example usage (will no-op if tools are missing):
    convert_folder_to_svg(Path("glyphs"), Path("svgs"), tool="potrace")
