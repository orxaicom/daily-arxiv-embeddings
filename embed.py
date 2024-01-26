from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_aug2023refresh_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_aug2023refresh_base")
model.load_adapter(
    "allenai/specter2_aug2023refresh",
    source="hf",
    load_as="specter2_proximity",
    set_active=True,
)

model.to(device)

class ArxivDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.df.iloc[idx]["title"]
        abstract = self.df.iloc[idx]["abstract"]
        text = title + tokenizer.sep_token + abstract
        return text

column_types = {
    "arxiv": str,
    "field": str,
    "subject": str,
    "categories": str,
    "authors": str,
    "title": str,
    "abstract": str,
}

df = pd.read_csv("daily-arxiv-embeddings.csv", dtype=column_types)

df["embedding"] = None

batch_size = 10  # You can adjust the batch size depending on your GPU memory

dataset = ArxivDataset(df)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Process each batch from the DataLoader with tqdm for progress tracking
for i, batch in enumerate(tqdm(dataloader, desc="Processing items")):
    # Preprocess the input and move to the device
    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    ).to(device)

    # Get the embeddings
    output = model(**inputs)
    embeddings = output.last_hidden_state[:, 0, :]

    # Move the embeddings back to CPU for numpy conversion and detach from the computation graph
    embeddings = embeddings.detach().cpu().numpy()

    # Save the embeddings in the DataFrame
    for j in range(embeddings.shape[0]):
        df.at[i*batch_size + j, "embedding"] = embeddings[j].tobytes()

# Save the DataFrame with embeddings to the same CSV file
df.to_csv("daily-arxiv-embeddings.csv", index=False)
