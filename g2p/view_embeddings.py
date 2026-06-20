import numpy as np

emb = np.load("phoneme_embeddings_learned.npy")

print("Shape:", emb.shape)
print("\nFirst 5 rows:")
print(emb[:5])