def transform(df):

    if "end" in df.columns:

        df.drop(
            columns=["end"],
            inplace=True,
        )

    return df