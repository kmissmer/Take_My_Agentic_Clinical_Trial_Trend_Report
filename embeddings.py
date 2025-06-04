# condition_embeddings_init.py

import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from utils.sql_util import get_table

# Load env
load_dotenv()
base_dir = os.getenv('base_dir', '.')  # fallback if not set
os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)

# Use GPU if available
device = 'cpu'

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Fetch all unique conditions from your DB
query = """
SELECT DISTINCT lower(name) AS condition
FROM conditions
WHERE name IS NOT NULL
"""

df = get_table(query)
df = df.drop_duplicates(subset='condition').reset_index(drop=True)
df['condition_id'] = df.index

# Generate embeddings
print("Embedding conditions...")
condition_embeddings = model.encode(df['condition'].tolist(), convert_to_tensor=True, device=device)

# Save to pickle
out = {
    'condition_embeddings': condition_embeddings.cpu(), 
    'conditions_df': df
}

out_path = os.path.join(base_dir, 'data', 'condition_embeddings.pkl')
with open(out_path, 'wb') as f:
    pd.to_pickle(out, f)

print(f"✅ Embeddings saved to {out_path}")
