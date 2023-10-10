import sys
import os
os.environ['MPLCONFIGDIR'] = "./tmp"
from mpi4py import MPI
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from decimal import Decimal

import pynga
import pynga.io

clims   = {}; clims["default"] = [-1, 1]
Usolmax = 10; clims["Usol"]=[-Usolmax, Usolmax]; clims["Vsol"]=[-Usolmax, Usolmax]; clims["Wsol"]=[-Usolmax, Usolmax] 
clims["U"]=[0, 25]; clims["V"]=[-Usolmax, Usolmax]; clims["W"]=[-Usolmax, Usolmax]; clims["P"]=[101082+1000, 101082-1000]; clims["T"]=[290, 310]; clims["RHO"]= [1.15, 1.21]
clims["P"]=[102382+50, 102382-50]
if __name__ == '__main__':
  # Parse command-line arguments
  import argparse
  #parser = argparse.ArgumentParser(usage=__doc__)
  #parser.add_argument("-case_path",  "--case_path",  type=str, required=True)

  # Inputs
  case_folder   = "/scratch/zisen347/scoping_runs/NGA/"
  case_name     = "103_RectTurb_flame1D"
  fields        = ["U", "U", "U", "T", "T", "T", "P", "P", "P", "RHO", "RHO", "RHO"]
  idirs         = [3,  2,  1,  3,  2, 1,  3,  2, 1,  3,  2, 1]
  isls          = [72, 64, 2, 72, 64, 2, 72, 64, 2, 72, 64, 2]
  fields        = ["V"]
  idirs         = [1, 1, 1]
  isls          = [358, 359, 360]
  fields        = ["V", "U", "P", "RHO"]
  idirs         = [1, 1, 1, 1]
  isls          = [72, 358, 358, 358]
  fields        = ["T"]
  idirs         = [1, ]
  isls          = [1]




  # Initialize MPI
  comm = MPI.COMM_WORLD
  npes = comm.Get_size()
  myid = comm.Get_rank()
  mypn = MPI.Get_processor_name()

  # Initialize NGA case
  case_path     = os.path.join(case_folder, case_name)
  hit           = pynga.io.case(comm=comm, case_path=case_path, input="input", config="config.flame1D", data_init="data.init.flame1D", nover=1)
  slx, sly, slz = hit.get_slice_inner()
  fl            = pynga.io.data_names(hit.case_path, add_data_init="data.init.flame1D")         # data names
  tl            = pynga.io.timelist(hit.case_path, add_data_init="data.init.flame1D")           # list of time
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

  vmin = -4.0; vmax = 4.0       # Range of colormap
  A = 0.7
  tau = 1 / (2*A)

  lg = [{}, {}, {}] # lg[idir]["size","ylabel","extent"]
  lg[0]["size"] = (hit.ny, hit.nz)
  lg[1]["size"] = (hit.nx, hit.nz)
  lg[2]["size"] = (hit.nx, hit.ny)
  lg[0]["ylabel"] = "z"; lg[0]["xlabel"] = "y"
  lg[1]["ylabel"] = "z"; lg[1]["xlabel"] = "x"
  lg[2]["ylabel"] = "y"; lg[2]["xlabel"] = "x"
  lg[0]["extent"] = np.array([hit.y[0], hit.y[-1], hit.z[0], hit.z[-1]])
  lg[1]["extent"] = np.array([hit.x[0], hit.x[-1], hit.z[0], hit.z[-1]])
  lg[2]["extent"] = np.array([hit.x[0], hit.x[-1], hit.y[0], hit.y[-1]])

  # For each time
  for it in range(0, len(tl), 10):
    # For each field
    for fno in fields:
      # Read data
      dname = fl[it]
      vars = hit.mpi_read_data(comm, dat_name=dname)
      for i, v in enumerate(idirs):
        idir    = idirs[i]                  # Plane yz normal to x
        isl     = isls[i]                   # DNS grids 1-192, so we output the mid-plane
        sname   = fno + "_" + str(idir) + "_" + str(isl).zfill(4)
        sfolder = os.path.join(resdata_case_folder, sname)
        ffolder = os.path.join(resfigs_case_folder, sname)
        if (myid == 0):
          if not os.path.exists(ffolder):
            os.mkdir(ffolder)
        comm.Barrier()
        if (myid == 0):
          strt = '%.3E' % Decimal(tl[it])
          dn = os.path.join(sfolder, strt+'.dat') 
          phi = np.fromfile(dn, dtype='double', count=lg[idir-1]["size"][0]*lg[idir-1]["size"][1])
          phi = np.reshape(phi, lg[idir-1]["size"], order='C').transpose()

          figunit = 5.5; labelsize = 22
          lx = lg[idir-1]["extent"][1]-lg[idir-1]["extent"][0]
          ly = lg[idir-1]["extent"][3]-lg[idir-1]["extent"][2]
          lxy = np.sqrt(lx**2 + ly**2)
          fig, axt = plt.subplots(figsize=(figunit*lx/lxy,figunit*ly/lxy))

          im = axt.imshow(phi, vmin=clims[fno][0], vmax=clims[fno][1], 
                          cmap="viridis", origin='lower',
                          extent=lg[idir-1]["extent"])
          axt.set_xlabel(lg[idir-1]["xlabel"], fontsize=labelsize)
          axt.set_ylabel(lg[idir-1]["ylabel"], fontsize=labelsize)
          axt.set_xticks([lg[idir-1]["extent"][0], lg[idir-1]["extent"][1]])
          axt.set_yticks([lg[idir-1]["extent"][2], lg[idir-1]["extent"][3]])
          axt.tick_params(axis='both', which='major', labelsize=labelsize-4)
          axt.tick_params(axis='both', which='minor', labelsize=labelsize-4)
          axt.set_title(fno, fontsize=labelsize)
          divider = make_axes_locatable(axt)
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im, cax=cax, orientation='vertical')
          fn = os.path.join(ffolder, strt+'.png')
          plt.savefig(fn, bbox_inches="tight")
          plt.close()
