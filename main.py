from __future__ import annotations

import polars as pl

"""
LINK = "https://www.pnas.org/doi/10.1073/pnas.211566398"
"""


def main() -> int:
    ds = pl.read_csv("data/GCM_Normal.res", separator="\t")

    print(ds.describe())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
