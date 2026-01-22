from __future__ import annotations

from typing import Any

import altair
import narwhals as nw
from narwhals.typing import IntoFrame

from ggsql._ggsql import split_query, render as _render

__all__ = ["split_query", "render_altair"]
__version__ = "0.1.0"


def render_altair(
    df: IntoFrame,
    viz: str,
    **kwargs: Any,
) -> altair.Chart:
    """Render a DataFrame with a VISUALISE spec to an Altair chart.

    Parameters
    ----------
    df
        Data to visualize. Accepts polars, pandas, or any narwhals-compatible
        DataFrame. LazyFrames are collected automatically.
    viz
        VISUALISE spec string (e.g., "VISUALISE x, y DRAW point")
    **kwargs
        Additional keyword arguments passed to `altair.Chart.from_json()`.
        Common options include `validate=False` to skip schema validation.

    Returns
    -------
    altair.Chart
        An Altair chart object.
    """
    df = nw.from_native(df, pass_through=True)

    if isinstance(df, nw.LazyFrame):
        df = df.collect()

    if not isinstance(df, nw.DataFrame):
        raise TypeError("df must be a narwhals DataFrame or compatible type")

    pl_df = df.to_polars()

    vegalite_json = _render(pl_df, viz, writer="vegalite")

    return altair.Chart.from_json(vegalite_json, **kwargs)
