from pathlib import Path

import altair as alt
import polars as pl

method_keys = ["DMP", "SSSN", "TYL", "TYL0", "MCL"]
method_names = {
    "MCL": "Monte-Carlo",
    "DMP": "Dynamic message passing",
    "SSSN": "SSS-Noself",
    "TYL": "2nd Taylor",
    "TYL0": "2nd Maclaurin",
}

column = alt.Column(
    "p",
    header=alt.Header(
        labels=False,
        title="Difference of propagation probability distributions",
    ),
)
row = alt.Row(
    "n",
    header=alt.Header(
        labels=False,
        title="Difference of initial active nodes",
        titleOrient="right",
    ),
)


def plot_result_error(location: str, cat: str) -> alt.FacetChart:
    dfs = []
    p = Path(location)

    for ns in range(1, 1 + 3):
        for ps in range(1, 1 + 3):
            ddf = pl.read_csv(p.joinpath(f"{cat}-{ns}-{ps}_error.csv"))
            dfs.append(ddf.with_columns([pl.lit(ns).alias("n"), pl.lit(ps).alias("p")]))

    df: pl.DataFrame = (
        pl.concat(dfs)
        .filter(pl.col("method1") == "MCL")
        .select(["method2", "mean", "n", "p"])
    )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("method2", title=None)
            .sort(method_keys)
            .axis(labelExpr=f"{method_names}[datum.label]", labelAngle=45),
            y=alt.Y("mean", title="Mean absolute error of probs."),
        )
        .properties(height=180, width=160)
        .facet(column=column, row=row)
    )
    return chart


def plot_computation_time(location: str, cat: str) -> alt.FacetChart:
    dfs = []
    p = Path(location)

    for ns in range(1, 1 + 3):
        for ps in range(1, 1 + 3):
            ddf = pl.read_csv(p.joinpath(f"{cat}-{ns}-{ps}_time.csv"))
            dfs.append(ddf.with_columns([pl.lit(ns).alias("n"), pl.lit(ps).alias("p")]))

    df: pl.DataFrame = pl.concat(dfs).rename({"time(us)": "time"})

    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X("method", title=None)
            .sort(method_keys)
            .axis(labelExpr=f"{method_names}[datum.label]", labelAngle=45),
            y=alt.Y("time")
            .scale(type="log")
            .axis(format=".0e", title="Computation time (\u00b5s)"),
        )
        .properties(height=180, width=200)
        .facet(column=column, row=row)
    )
    return chart


def main():
    format = "png"
    # format = "pdf"
    for cat in ["twitter", "facebook"]:
        plot_computation_time("../test/sample-result", cat).save(f"{cat}_time.{format}")
        plot_result_error("../test/sample-result", cat).save(f"{cat}_error.{format}")


if __name__ == "__main__":
    main()
