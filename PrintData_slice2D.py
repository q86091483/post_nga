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
clims["U"]=[45, 55]; clims["V"]=[-8, 8]; clims["W"]=[-8, 8] 

if __name__ == '__main__':
  # Parse command-line arguments
  import argparse
  #parser = argparse.ArgumentParser(usage=__doc__)
  #parser.add_argument("-case_path",  "--case_path",  type=str, required=True)

  # Inputs
  case_folder   = "/home/zisen347/scratch/scoping_runs/NGA/" 
  case_name     = "104_PlanarFlame_4"
  fields        = ["P", "U", "V", "RHO", "T", "W", "OH", "H2O", ]
  idirs         = [2,]
  isls          = [40, ]

  # Initialize MPI
  comm = MPI.COMM_WORLD
  npes = comm.Get_size()
  myid = comm.Get_rank()
  mypn = MPI.Get_processor_name()

  # Initialize NGA case
  case_path     = os.path.join(case_folder, case_name)
  hit           = pynga.io.case(comm=comm, case_path=case_path, input="input", config="ufs:config.hit", data_init="ufs:data.init.hit", nover=3)
  slx, sly, slz = hit.get_slice_inner()
  fl            = pynga.io.data_names(hit.case_path, add_data_init="ufs:data.init.hit")         # data names
  tl            = pynga.io.timelist(hit.case_path, add_data_init="ufs:data.init.hit")         # list of time

  # Initialize output folder
  resfigs_folder = "0ResFigs"; resfigs_case_folder = os.path.join(resfigs_folder, case_name)
  resdata_folder = "0ResData"; resdata_case_folder = os.path.join(resdata_folder, case_name)
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
  rs = range(0, len(tl), 2)
  for it in rs:
    # For each field
    for fno in fields:
      # Read data
      dname = fl[it]
      vars = hit.mpi_read_data(comm, dat_name=dname)
      for i, v in enumerate(idirs):
        idir    = idirs[i]                  # Plane yz normal to x
        isl     = isls[i]                   # DNS grids 1-192, so we output the mid-plane
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
