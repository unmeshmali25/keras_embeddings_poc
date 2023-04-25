import tensorflow as tf
import umap.umap_ as umap
import matplotlib.pyplot as plt

def create_embeddings(data): 
  # Create a vocabulary for the column
  vocab = sorted(set(data))
  print(f" After sorting : {vocab} \n")

  # Create a mapping from strings to integer indices
  word2idx = {u:i for i, u in enumerate(vocab)}
  print(f"After mapping to integer indices : {word2idx} \n")

  # Convert the column to a list of integer indices
  column_data_idx = [word2idx[word] for word in data]
  print(f"List of interger indices : {column_data_idx} \n")

  # Create an embedding layer
  embedding_dim = 4  # The size of the embedding vector
  embedding_layer = tf.keras.layers.Embedding(len(vocab), embedding_dim)

  # Create a tensor with the integer indices
  column_data_tensor = tf.constant(column_data_idx, dtype=tf.int32)
  print(f"Tensor with column data : {column_data_tensor} \n")

  # Pass the tensor through the embedding layer to get the embeddings
  embeddings = embedding_layer(column_data_tensor)

  print(f"Here's the embeddings  : {embeddings}")  
  return embeddings 

def umap_embeddings(mat): 
  # Convert the embeddings to a numpy array
  embeddings = mat.numpy()

  # Use UMAP to reduce the dimensionality of the embeddings to 2
  umap_embeddings = umap.UMAP(n_neighbors=5,
                              min_dist=0.3,
                              metric='correlation').fit_transform(embeddings)

  # Plot the UMAP embeddings
  plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])
  plt.show()

  
if __name__=='__main__': 
  # Create a list of strings to represent the column
  column_data = ['apple', 'banana', 'orange', 'banana', 'orange', 'apple']

  embedding_matrix = create_embeddings(column_data)

  umap_embeddings(embedding_matrix)
