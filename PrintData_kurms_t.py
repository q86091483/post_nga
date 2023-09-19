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

  # Initialize MPI
  comm = MPI.COMM_WORLD
  npes = comm.Get_size()
  myid = comm.Get_rank()
  mypn = MPI.Get_processor_name()

  # Initialize HIT case
  case_folder   = "/scratch/b/bsavard/zisen347/cases"
  case_name     = "BoxHIT_ForcingUs"
  case_path     = os.path.join(case_folder, case_name)
  hit           = pynga.io.case(comm=comm, case_path=case_path, input="input", config="config", data_init="data.init", nover=1)
  slx, sly, slz = hit.get_slice_inner()
  fl            = pynga.io.data_names(hit.case_path)         # data names
  tl            = pynga.io.timelist(hit.case_path)           # list of time

  # Initialize output folder
  resfigs_folder = "0ResFigs"; resfigs_case_folder = os.path.join(resfigs_folder, case_name)
  resdata_folder = "0ResData"; resdata_case_folder = os.path.join(resdata_folder, case_name)

  print("here")
  if myid == 0:
    if not os.path.exists(resfigs_folder):
      os.mkdir(resfigs_folder)
    if not os.path.exists(resfigs_case_folder):
      os.mkdir(resfigs_case_folder)
    if not os.path.exists(resdata_folder):
      os.mkdir(resdata_folder)
    if not os.path.exists(resdata_case_folder):
      os.mkdir(resdata_case_folder)
  comm.Barrier()
  
  vmin = -4.0; vmax = 4.0       # Range of colormap
  res_t     = []
  res_k     = []
  res_urms  = []
  res_vrms  = []
  res_wrms  = []
  A = 0.7
  tau = 1 / (2*A)
  for it in range(0, len(tl), 1):

    # Output V component - we can change to other field, see vars["field_names"]
    fno  = "W"
    idir = 2                   # Plane yz normal to x
    isl  = 48                  # DNS grids 1-192, so we output the mid-plane

    # Read data
    strt = "0.0"
    dname = fl[it]
    vars = hit.mpi_read_data(comm, dat_name=dname)

    def get_name(fno, idir, isl, strt):
      return fno + "_" + str(idir) + "_" + str(isl).zfill(4) + "_" + strt
    sn = get_name(fno,idir,isl,strt)

    # Write isl-th yz slice in parallel
    hit.mpi_write_slice(field=vars[fno], fn = sn+".dat", idir=idir, index_F=isl)
    comm.Barrier()

    # Calculate urms
    u2mean  = hit.calc_mean_all(comm=comm, field=vars["U"]**2)
    v2mean  = hit.calc_mean_all(comm=comm, field=vars["V"]**2) 
    w2mean  = hit.calc_mean_all(comm=comm, field=vars["W"]**2) 
    umean   = hit.calc_mean_all(comm=comm, field=vars["U"]) 
    vmean   = hit.calc_mean_all(comm=comm, field=vars["V"]) 
    wmean   = hit.calc_mean_all(comm=comm, field=vars["W"]) 
    urms    = np.sqrt(u2mean - umean**2)
    vrms    = np.sqrt(v2mean - vmean**2)
    wrms    = np.sqrt(w2mean - wmean**2)
    k       = 0.5*(urms**2 + vrms**2 + wrms**2)
    
    res_t.append(tl[it])
    res_k.append(k)
    res_urms.append(urms)
    res_vrms.append(vrms)
    res_wrms.append(wrms)

    # End iteration

  if (myid == 0):
    np.savetxt(os.path.join(resdata_case_folder, 'res_t.dat'), res_t)
    np.savetxt(os.path.join(resdata_case_folder, 'res_k.dat'), res_k)
    np.savetxt(os.path.join(resdata_case_folder, 'res_urms.dat'), res_urms)
    np.savetxt(os.path.join(resdata_case_folder, 'res_vrms.dat'), res_vrms)
    np.savetxt(os.path.join(resdata_case_folder, 'res_wrms.dat'), res_wrms)

    plt, axres = plt.subplots()
    k0s = 13.5*A*A*(0.18*6.28)**2
    u0s = np.sqrt((k0s*2)/3)
    k0s = 1.0
    u0s = 1.0
    axres.plot(np.asarray(res_t) / tau, np.asarray(res_k)/k0s, "b-", label="k")
    axres.plot(np.asarray(res_t) / tau, np.asarray(res_urms)/u0s, label='urms')
    axres.plot(np.asarray(res_t) / tau, np.asarray(res_vrms)/u0s, label='vrms')
    axres.plot(np.asarray(res_t) / tau, np.asarray(res_wrms)/u0s, label='wrms') 
    axres.plot([0, res_t[-1]/tau], [8.4, 8.4], 'b--') 
    axres.legend()
    axres.set_xlabel(r"$t/\tau$", fontsize = 22)
    axres.set_ylabel(r"$k$", fontsize = 22)
    axres.set_ylim([0, 12])
    axres.tick_params(axis='both', which='major', labelsize=18)
    axres.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig(os.path.join(resfigs_case_folder, "kurms_t.png"), bbox_inches="tight")

  # End MPI processes.
  MPI.Finalize()