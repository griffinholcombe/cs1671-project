from datasets import load_dataset

# Download dataset from huggginface
ds = load_dataset("Jinyan1/COLING_2025_MGT_en")

# Split dataset into train and dev
train_dataset = ds['train']
dev_dataset = ds['dev']

print(f"Train size: {len(train_dataset)}")
print(f"Dev size: {len(dev_dataset)}")

