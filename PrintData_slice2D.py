import sys
import os
os.environ['MPLCONFIGDIR'] = "./tmp"
from mpi4py import MPI
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal

import pynga
import pynga.io

clims   = {}; clims["default"] = [-1, 1]
clims["Usol"]=[-8, 8]; clims["Vsol"]=[-8, 8]; clims["Wsol"]=[-8, 8] 
clims["U"]=[-6, 6]; clims["V"]=[-6, 6]; clims["W"]=[-6, 6] 

if __name__ == '__main__':
  # Parse command-line arguments
  import argparse
  #parser = argparse.ArgumentParser(usage=__doc__)
  #parser.add_argument("-case_path",  "--case_path",  type=str, required=True)

  # Inputs
  case_folder   = "/scratch/b/bsavard/zisen347/cases/"
  case_name     = "BoxHIT_ForcingUs_1"
  fields        = ["U", "V", "W", "P", "RHO", "T"] + ["U", "V", "W", "P", "RHO", "T"]
  idirs         = [2, 2, 2, 2, 2, 2] + [2, 2, 2, 2, 2, 2]
  isls          = [1, 1, 1, 1, 1, 1] + [2, 2, 2, 2, 2, 2]

  fields = ["P"] * 6 + ["V"] * 6
  idirs = [1, 1, 2, 2, 3, 3] * 2
  isls = [1, 96, 1, 96, 1, 96] * 2

  # Initialize MPI
  comm = MPI.COMM_WORLD
  npes = comm.Get_size()
  myid = comm.Get_rank()
  mypn = MPI.Get_processor_name()

  # Initialize NGA case
  case_path     = os.path.join(case_folder, case_name)
  hit           = pynga.io.case(comm=comm, case_path=case_path, input="input", config="0config", data_init="0data", nover=0)
  slx, sly, slz = hit.get_slice_inner()
  fl            = pynga.io.data_names(hit.case_path, add_data_init="0data")         # data names
  tl            = pynga.io.timelist(hit.case_path, add_data_init="0data")         # list of time
  print("PrintData processing: ", fl)

  # Initialize output folder
  resfigs_folder = "0ResFigs"; resfigs_case_folder = os.path.join(resfigs_folder, case_name)
  resdata_folder = "0ResData"; resdata_case_folder = os.path.join(resdata_folder, case_name)
  print(resdata_case_folder)
  if (myid == 0):
    if not os.path.exists(resfigs_folder):
      os.mkdir(resfigs_folder)
    if not os.path.exists(resfigs_case_folder):
      os.mkdir(resfigs_case_folder)
    if not os.path.exists(resdata_folder):
      os.mkdir(resdata_folder)
    if not os.path.exists(resdata_case_folder):
      os.mkdir(resdata_case_folder)
  comm.Barrier()
 
  # For each time
  rs = range(0, len(tl), 1)
  print(rs)
  for it in rs:
    # For each field
    for fno in fields:
      # Read data
      dname = fl[it]
      vars = hit.mpi_read_data(comm, dat_name=dname)
      for i, v in enumerate(idirs):
        idir    = idirs[i]                  
        isl     = isls[i]                   
        sfolder = fno + "_" + str(idir) + "_" + str(isl).zfill(4)
        sfolder = os.path.join(resdata_case_folder, sfolder)
        if (myid == 0):
          if not os.path.exists(sfolder):
            os.mkdir(sfolder)
        comm.Barrier()
        # Write data
        strt = '%.3E' % Decimal(tl[it])
        sn = os.path.join(sfolder, strt+'.dat')
        # Write isl-th yz slice in parallel
        #print(np.mean(vars["P"]))
        hit.mpi_write_slice(field=vars[fno], fn = sn, idir=idir, index_F=isl)
        #print(hit.iproc, hit.jproc, hit.kproc, np.amax(vars[fno]), np.unravel_index(vars[fno].argmax(), vars[fno].shape))
        comm.Barrier()

    # End iteration
