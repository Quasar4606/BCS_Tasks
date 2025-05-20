import numpy as np

# Load the compressed .npz file
data = np.load('q_table_compressed.npz')

# Extract the array
q_table = data['q_table']

# Save it as a normal .npy file (uncompressed)
np.save('q_table.npy', q_table)