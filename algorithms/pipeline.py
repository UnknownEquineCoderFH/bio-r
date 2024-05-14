import polars as pl

dirty = pl.read_csv("data/GCM_Total.res", separator="\t", truncate_ragged_lines=True)

print(dirty.head())

dirty.describe()


def clean_column_names(data: pl.DataFrame) -> pl.DataFrame:
    new_columns = []
    for i, col in enumerate(data.columns):
        if "duplicated" in col:
            new_col = f"{data.columns[i-1]}_class"
        elif col == "":
            new_col = f"{data.columns[i-1]}_class"
        else:
            new_col = col
        new_columns.append(new_col)

    data.columns = new_columns

    return data


ds = clean_column_names(dirty)

ds.describe()

dirty = pl.read_csv("data/GCM_Normal.res", separator="\t", truncate_ragged_lines=True)

normal_ds = clean_column_names(dirty)

normal_ds.describe()

classes = [col for col in ds.columns if "class" in col]
mutations_df = ds.select(["Accession"] + classes)

mutations_df = mutations_df.with_columns(
    pl.Series(
        "PRESENT", mutations_df.map_rows(lambda s: sum([1 for i in s if i == "P"]))
    )
)

summed = mutations_df.sort("PRESENT", descending=True).select(["Accession", "PRESENT"])

summed.head(20)


def remove_everything_but_class_labels(data: pl.DataFrame) -> pl.DataFrame:
    for col in data.columns:
        if "_class" not in col and col != "Accession" and col != "Description":
            data = data.drop(col)

    return data


def remove_class_labels(data: pl.DataFrame) -> pl.DataFrame:
    for col in data.columns:
        if "_class" in col:
            data = data.drop(col)

    return data


normal_ds = remove_class_labels(normal_ds)


print(ds.head())
