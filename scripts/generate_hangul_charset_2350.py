from __future__ import annotations

"""
Generate the standard 2,350 modern Hangul syllables file (KS X 1001 order).
"""

import sys
from pathlib import Path
from typing import List


def _build_ksx1001_charset() -> List[str]:
    """Return the 2,350 Hangul syllables from KS X 1001 in canonical order."""
    return [
        bytes([lead, trail]).decode("euc_kr")
        for lead in range(0xB0, 0xC9)
        for trail in range(0xA1, 0xFF)
    ]


CHARSET_2350: List[str] = _build_ksx1001_charset()


def main() -> None:
    if len(CHARSET_2350) != 2350:
        print(
            f"Error: charset length is {len(CHARSET_2350)}, expected 2350.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(set(CHARSET_2350)) != len(CHARSET_2350):
        print("Error: charset contains duplicate characters.", file=sys.stderr)
        sys.exit(1)

    output_path = Path("charset_2350.txt")
    try:
        output_path.write_text("\n".join(CHARSET_2350) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"Error writing {output_path}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
