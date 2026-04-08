import altair as alt
import polars as pl


def main():
    dfs = []
    for ns in range(1, 1 + 3):
        for ps in range(1, 1 + 3):
            ddf = pl.read_csv(f"../test/sample-result/twitter-{ns}-{ps}_time.csv")
            dfs.append(ddf.with_columns([pl.lit(ns).alias("n"), pl.lit(ps).alias("p")]))

    df: pl.DataFrame = pl.concat(dfs).rename({"time(us)": "time"})
    print(df.filter(pl.col("time") == 0))

    chart = (
        alt.Chart(df.filter(pl.col("method") != "MCL"))
        .mark_point()
        .encode(x="n", y=alt.Y("time").scale(type="log"), color="method")
        # .properties(width=160, height=160)
        .facet(column="p")
    )
    chart.save("plot.png")


if __name__ == "__main__":
    main()
