from pathlib import Path

import altair as alt
import polars as pl


def plot_comutation_time(location: str, cat: str) -> alt.FacetChart:
    dfs = []
    p = Path(location)

    for ns in range(1, 1 + 3):
        for ps in range(1, 1 + 3):
            ddf = pl.read_csv(p.joinpath(f"{cat}-{ns}-{ps}_time.csv"))
            dfs.append(ddf.with_columns([pl.lit(ns).alias("n"), pl.lit(ps).alias("p")]))

    df: pl.DataFrame = pl.concat(dfs).rename({"time(us)": "time"})
    print(df.filter(pl.col("time") == 0))

    method_keys = ["MCL", "DMP", "SSSN", "TYL", "TYL0"]
    method_names = {
        "MCL": "Monte-Carlo",
        "DMP": "Dynamic message passing",
        "SSSN": "SSS-Noself",
        "TYL": "2nd Taylor",
        "TYL0": "2nd Maclaurin",
    }
    chart = (
        alt.Chart(df)  # df.filter(pl.col("method") != "MCL"))
        .mark_point()
        .encode(
            x=alt.X("method", title=None)
            .sort(method_keys)
            .axis(labelExpr=f"{method_names}[datum.label]", labelAngle=45),
            y=alt.Y("time")
            .scale(type="log")
            .axis(format=".0e", title="Computation time (\u00b5s)"),
        )
        .properties(height=180, width=150)
        .facet(
            column=alt.Column(
                "p",
                header=alt.Header(
                    labels=False,
                    title="Difference of propagation probability distributions",
                ),
            ),
            row=alt.Row(
                "n",
                header=alt.Header(
                    labels=False,
                    title="Difference of initial active nodes",
                    titleOrient="right",
                ),
            ),
        )
    )
    return chart


def main():
    chart = plot_comutation_time("../test/sample-result", "twitter")
    chart.save("plot.pdf")


if __name__ == "__main__":
    main()
