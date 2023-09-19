import sys
import os
from mpi4py import MPI
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pynga
import pynga.io


if __name__ == '__main__':
  # Parse command-line arguments
  import argparse
  #parser = argparse.ArgumentParser(usage=__doc__)
  #parser.add_argument("-case_path",  "--case_path",  type=str, required=True)

  # case folder 
  # case names
  # frequency
  # time range
  # field names
  # positions

  # Initialize MPI
  comm = MPI.COMM_WORLD
  npes = comm.Get_size()
  myid = comm.Get_rank()
  mypn = MPI.Get_processor_name()

  # Initialize HIT case
  case_folder   = "/scratch/b/bsavard/zisen347/cases"
  case_name     = "BoxHIT_FlameForcing"
  case_path     = os.path.join(case_folder, case_name)
  hit           = pynga.io.case(comm=comm, case_path=case_path, input="input", config="config", data_init="data.init", nover=1)
  slx, sly, slz = hit.get_slice_inner()
  fl            = pynga.io.data_names(hit.case_path)         # data names
  tl            = pynga.io.timelist(hit.case_path)           # list of time

  vmin = -4.0; vmax = 4.0       # Range of colormap
  ts = []
  res = []
  A = 0.7
  tau = 1 / (2*A)
  fl = ["data.init"] 
  for it in range(0, 1):

    # Output V component - we can change to other field, see vars["field_names"]
    fno  = "W"
    idir = 2                   # Plane yz normal to x
    isl  = 48                  # DNS grids 1-192, so we output the mid-plane

    # Read data
    strt = "0.0"
    dname = fl[it]
    print(dname)
    vars = hit.mpi_read_data(comm, dat_name=dname)

    def get_name(fno, idir, isl, strt):
      return fno + "_" + str(idir) + "_" + str(isl).zfill(4) + "_" + strt
    sn = get_name(fno,idir,isl,strt)

    # Write isl-th yz slice in parallel
    hit.mpi_write_slice(field=vars[fno], fn = sn+".dat", idir=idir, index_F=isl)
    if (myid == 0):
      f   = sn + ".dat"
      phi = np.fromfile(f, dtype='double', count=hit.ny * hit.nz)
      phi = np.reshape(phi, (hit.ny,hit.nz), order='C')
      fig, axt = plt.subplots()
      axt.imshow(phi, vmin=vmin, vmax=vmax)
      print(f)
      plt.savefig(sn+".png")
    comm.Barrier()


    # Read the slice with root rank 0
    #if (myid == 0):
    #  fread = sn + ".dat"
    #  phi = np.fromfile(fread, dtype='double', count=hit.ny * hit.nz)
    #  phi = np.reshape(phi, (hit.ny,hit.nz), order='C')
    #  fig, ax1 = plt.subplots()
    #  ax1.imshow(phi, vmin=vmin, vmax=vmax, cmap="seismic")
    #  plt.savefig(sn + ".png", bbox_inches = "tight")

    # Calculate urms
    u2mean  = np.mean(vars["U"][slx,sly,slz]**2)
    v2mean  = np.mean(vars["V"][slx,sly,slz]**2)
    w2mean  = np.mean(vars["W"][slx,sly,slz]**2)
    umean   = np.mean(vars["U"][slx,sly,slz])
    vmean   = np.mean(vars["V"][slx,sly,slz])
    wmean   = np.mean(vars["W"][slx,sly,slz])
    urms    = np.sqrt(u2mean - umean**2)
    vrms    = np.sqrt(v2mean - vmean**2)
    wrms    = np.sqrt(w2mean - wmean**2)
    k       = 0.5*(urms**2 + vrms**2 + wrms**2)
    
    if (myid == 0):
      print(tl[it], k)  
    ts.append(tl[it])
    res.append(k)
    # End iteration

  if (myid == 0):
    plt, axres = plt.subplots()
    axres.plot(np.asarray(ts) / tau, res, )
    axres.set_xlabel(r"$t/\tau$", fontsize = 22)
    axres.set_ylabel(r"$k$", fontsize = 22)
    axres.set_ylim([0, 10])
    axres.tick_params(axis='both', which='major', labelsize=18)
    axres.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig("k.png", bbox_inches="tight")

  # End MPI processes.
  MPI.Finalize()