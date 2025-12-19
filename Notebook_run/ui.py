from __future__ import annotations

from typing import Optional

import pandas as pd


def display_scrollable_df(df: pd.DataFrame, *, max_rows: int = 50, max_height_px: int = 320) -> None:
    try:
        from IPython.display import HTML, display  # type: ignore
    except Exception:
        print(df.head(int(max_rows)))
        return

    view = df.head(int(max_rows))
    html = view.to_html(index=False)
    display(
        HTML(
            f"<div style='max-height:{int(max_height_px)}px; overflow:auto; border:1px solid #ddd;'>"
            f"{html}</div>"
        )
    )

