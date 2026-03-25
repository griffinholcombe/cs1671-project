from datasets import load_dataset, concatenate_datasets

DATASET_NAME = "Jinyan1/COLING_2025_MGT_en"

ds = load_dataset(DATASET_NAME)  # has 'train' and 'dev'

df = concatenate_datasets([ds["train"], ds["dev"]])
df = df.to_pandas()
total_rows = len(df)
print(total_rows)


# ==================================================================
# remove rows with word_count < 3
# ==================================================================
og_length = len(df)
df["word_count"] = df["text"].astype(str).str.split().str.len()
df = df[df["word_count"] >= 3]

new_length = len(df)
print(f"Removed {og_length - new_length} rows from word_count < 3")

# remove duplicates in 'text' column
og_length = len(df)
df = df.drop_duplicates(subset=['text'])
new_length = len(df)
print(f"Removed {og_length - new_length} rows from duplicates")

# ==================================================================
# will stratify by word count, label, sub_source
# create a new column called 'word_count'
# label and sub-source are already in the dataset
# ==================================================================
from sklearn.model_selection import train_test_split

# word_count + key (pandas)
df["wc_bucket"] = (df["text"].astype(str).str.split().str.len() > 100).map(
    {False: "le_100", True: "gt_100"}
)

# and build key off that (instead of word_count)
df["key"] = (
    df["wc_bucket"].astype(str)
    + "_" + df["label"].astype(str)
)

# stratify by (id, key) pairs so all rows for an id stay in the same split
u_df = df[["id", "key"]].drop_duplicates()

train_ids, tmp_ids = train_test_split(
    u_df, test_size=0.2, random_state=42, stratify=u_df["key"]
)
val_ids, test_ids = train_test_split(
    tmp_ids, test_size=0.5, random_state=42, stratify=tmp_ids["key"]
)

train = df[df["id"].isin(train_ids["id"])].copy()
val   = df[df["id"].isin(val_ids["id"])].copy()
test  = df[df["id"].isin(test_ids["id"])].copy()

print(train.shape, val.shape, test.shape)
print('done')


# print value counts of label and word count / length for train/val/test
# these should be % of total rows in the split
print("Train:")
print(train["label"].value_counts() / len(train))
print(train["wc_bucket"].value_counts() / len(train))
print("Val:")
print(val["label"].value_counts() / len(val))
print(val["wc_bucket"].value_counts() / len(val))
print("Test:")
print(test["label"].value_counts() / len(test))
print(test["wc_bucket"].value_counts() / len(test))

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

print('done')