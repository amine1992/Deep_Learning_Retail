def Pb2Scipy(filename, meta=[], chunk_size=-1):
  """  Reads a paperboat file and returns a scipy array
       input: 
       filename the paperboat filename
       meta a list of metadata to be read from the file. It can have values 0,1,2
       output:
         a scipy array and a list with the meta data per row  
       Example:
       import pb2scipy as pb
       (array, meta0, meta1, meta2)=pb.Pb2Sipy(filename, [0,1,2])
  """

  int8=[]
  float32=[]
  int64=[]
  precision=scipy.float64
  chunk_counter=0
  fin=open(filename, "r+b")
  mm=mmap.mmap(fin.fileno(), 0);
  header=mm.readline().strip("\n")
  tokens=header.split(",")
  has_meta=False
  is_sparse=False
  dimensionality=0
  if tokens[1].startswith("meta"):
    has_meta=True
  for t in tokens[2:len(tokens)]:
    t1=t.split(":")
    if t1[0]=="sparse":
      is_sparse=True
      dimensionality+=int(t1[2])
    else:
      dimensionality+=int(t1[1])

    if t1[1]=="double" or t1[0]=="double":
      precision=scipy.float64
      break
    else:
      if t1[1]=="uint8" or t1[0]=="uint8":
        precision=scipy.uint8
        break
      else:
        if t1[1]=="bool" or t1[0]=="bool":
          precision=scipy.bool8
          break
        else:
          if t1[1]=="int32" or t1[0]=="int32":
            precision=scipy.int32
            break
          else:
            if t1[1]=="uint32" or t1[0]=="uint32":
              precision=scipy.uint32
              break
            else:
              if t1[1]=="int64" or t1[0]=="int64":
                precision=scipy.int64
                break
              else:
                if t1[1]=="uint64" or t1[0]=="uint64":
                  precision=scipy.uint64
                  break
  num_of_points=0
  total_points_so_far=0
  for line in iter(mm.readline, ""):
    num_of_points+=1
  fin=open(filename, "r+b")
  mm=mmap.mmap(fin.fileno(), 0)
  mm.readline()

  if is_sparse:
    ii=[]
    jj=[]
    vv=[]
  else:
    if chunk_size==-1:
        mat=scipy.zeros((num_of_points, dimensionality), dtype=precision)
    else:
        mat=scipy.zeros((chunk_size, dimensionality), dtype=precision)

  point_counter=0    
  for line in iter(mm.readline, ""):
    if point_counter==0:
        int8=[]
        float32=[]
        int64=[]
    col_counter=0
    tokens=line.strip(",\n").split(",")
    start_column=0
    if has_meta:
      start_column=3
      for m in meta:
        if m==0:
          int8.append(int(tokens[0]))
        else:
          if m==1:
            float32.append(float(tokens[1]))
          else:
            int64.append(int(tokens[2]))
    if is_sparse:
      for t in tokens[start_column:len(tokens)]:
        (ind,val)=t.split(":")
        ii.append(point_counter)
        jj.append(int(ind))
        vv.append(precision(val))
    else:
      for t in tokens[start_column:len(tokens)]:
        mat[point_counter, col_counter]=precision(t)
        col_counter+=1
    point_counter+=1
    total_points_so_far+=1
    if point_counter==chunk_size or total_points_so_far==num_of_points:
      if is_sparse:
        mat=scipy.sparse.coo_matrix((vv, (ii,jj)), shape=(point_counter, dimensionality))
        vv=[]
        ii=[]
        jj=[]
      else:
          mat=mat[0:point_counter,:]

      point_counter=0
      result=[mat]
      if 0 in meta:
        result.append(int8)
      if 1 in meta:
        result.append(float32)
      if 2 in meta:
        result.append(int64)
      yield result
 