import numpy as np

#load q table
q_table = np.load('q_table.npy')

#compress it
np.savez_compressed('q_table_compressed.npz', q_table=q_table)
