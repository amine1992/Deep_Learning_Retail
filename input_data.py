import random
import csv
import numpy as np
import pb2scipy as pb


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def next_batch(filename, len_file, size_batch=500 ):

  #generator
  gen = pb.Pb2Scipy(filename,[0,1,2],size_batch)
  #starting point is the line number to start getting batches from
  starting_point = random.randrange(1, len_file - size_batch)

  #define counter for the number of chunks of size_batch size treated so far
  
  for mat in gen:
    num_chunks_treated+=1

    # when we reach the starting_line line, we transform to sparse matrice
    if num_chunks_treated*size_batch >=starting_point:
      X_batch = mat[0].todense().astype(np.float32)
      Y_batch = np.array(mat[2])
      break
  return (X_batch,Y_batch)
