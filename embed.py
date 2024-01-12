from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import pandas as pd
from tqdm import tqdm
import os


os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_aug2023refresh_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_aug2023refresh_base")
model.load_adapter(
    "allenai/specter2_aug2023refresh",
    source="hf",
    load_as="specter2_proximity",
    set_active=True,
)

# Read data from CSV
df = pd.read_csv("daily-arxiv-embeddings.csv")

# Set 'embedding' column to None initially
df["embedding"] = None

# Process each row from the CSV with tqdm for progress tracking
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing items"):
    # Retrieve title and abstract from the CSV
    title = row["title"]
    abstract = row["abstract"]

    # Concatenate title and abstract
    text = title + tokenizer.sep_token + abstract

    # Preprocess the input
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    )

    # Get the embeddings
    output = model(**inputs)
    embedding = output.last_hidden_state[:, 0, :]

    # Save the embedding in the DataFrame
    df.at[index, "embedding"] = embedding.detach().numpy().tobytes()

# Save the DataFrame with embeddings to the same CSV file
df.to_csv("daily-arxiv-embeddings.csv", index=False)
