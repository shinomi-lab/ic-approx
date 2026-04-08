import polars as pl


def main():
    dfs = []
    for ns in range(1, 1 + 3):
        for ps in range(1, 1 + 3):
            ddf = pl.read_csv(f"../test/sample-result/twitter-{ns}-{ps}_time.csv")
            dfs.append(ddf.with_columns([pl.lit(ns).alias("n"), pl.lit(ps).alias("p")]))

    df = pl.concat(dfs)
    print(df)


if __name__ == "__main__":
    main()
