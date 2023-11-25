import sys
import os
os.environ['MPLCONFIGDIR'] = "./tmp"
from mpi4py import MPI
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from decimal import Decimal

import pynga
import pynga.io

clims   = {}; cmaps = {}; 

clims["U"]  = 0 + np.array([-10, 10])
cmaps["U"]  = cm.seismic

clims["V"]  = 0 + np.array([-20, 20]); 
cmaps["V"]  = cm.seismic

clims["W"]  = 0 + np.array([-20, 20]); 
cmaps["W"]  = cm.seismic

clims["T"]  = 355 + np.array([-20, 20]); 
cmaps["T"]  = cm.seismic

clims["RHO"]= 1 + np.array([-0.05, 0.05])
cmaps["RHO"]  = cm.seismic

clims["P"]  = 101325 + np.array([-5000, 5000])
cmaps["P"]  = cm.seismic

clims["O2"] = 0.232 + np.array([-0.2, 0.2])

cmaps["default"] = cm.viridis
clims["default"] = [-1, 1]

if __name__ == '__main__':
  # Parse command-line arguments
  import argparse
  #parser = argparse.ArgumentParser(usage=__doc__)
  #parser.add_argument("-case_path",  "--case_path",  type=str, required=True)

  # Inputs
  case_folder   = "/scratch/b/bsavard/zisen347/cases/"
  case_name     = "BoxHIT_ForcingUs_1"
  fields        = ["U", "V", "W", "P", "RHO", "T"]
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
  hit           = pynga.io.case(comm=comm, case_path=case_path, 
                                input="input", config="0config", data_init="0data", nover=0)
  slx, sly, slz = hit.get_slice_inner()
  fl            = pynga.io.data_names(hit.case_path, 
                                      add_data_init="0data")         # data names
  tl            = pynga.io.timelist(hit.case_path, 
                                    add_data_init="0data")           # list of time
  print("Plot processing: ", fl)
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
  A = 7000
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
  rs = range(0, len(tl), 1)
  for it in rs:
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

          figunit = 5.5; labelsize = 18
          lx = lg[idir-1]["extent"][1]-lg[idir-1]["extent"][0]
          ly = lg[idir-1]["extent"][3]-lg[idir-1]["extent"][2]
          lxy = np.sqrt(lx**2 + ly**2)
          fig, axt = plt.subplots(figsize=(figunit*lx/lxy,figunit*ly/lxy))
          if ("V" in fields):
            m = np.mean(phi)
            v2 = np.sqrt(np.mean((phi-m)**2))
            print(fno, "m", m, "rms ", v2)
          if fno in cmaps.keys():
            cmap = cmaps[fno]
          else:
            cmap = cmaps["default"]
          if fno in clims.keys():
            im = axt.imshow(phi, vmin=clims[fno][0], vmax=clims[fno][1], 
                          cmap=cmap, origin='lower',
                          extent=lg[idir-1]["extent"])
          else:
            im = axt.imshow(phi, #vmin=clims[fno][0], vmax=clims[fno][1], 
                          cmap=cmap, origin='lower',
                          extent=lg[idir-1]["extent"])
          axt.set_xlabel(lg[idir-1]["xlabel"], fontsize=labelsize)
          axt.set_ylabel(lg[idir-1]["ylabel"], fontsize=labelsize)
          axt.set_xticks([lg[idir-1]["extent"][0], lg[idir-1]["extent"][1]])
          axt.set_yticks([lg[idir-1]["extent"][2], lg[idir-1]["extent"][3]])
          axt.set_xticklabels(['%.2E'%Decimal(lg[idir-1]["extent"][0]), '%.2E'%Decimal(lg[idir-1]["extent"][1])])
          axt.set_yticklabels(['%.2E'%Decimal(lg[idir-1]["extent"][2]), '%.2E'%Decimal(lg[idir-1]["extent"][3])])
          axt.tick_params(axis='both', which='major', labelsize=labelsize-4)
          axt.tick_params(axis='both', which='minor', labelsize=labelsize-4)
          axt.set_title(fno, fontsize=labelsize)
          divider = make_axes_locatable(axt)
          cax = divider.append_axes('right', size='5%', pad=0.05)
          fig.colorbar(im, cax=cax, orientation='vertical')
          fn = os.path.join(ffolder, strt+'.png')
          plt.savefig(fn, bbox_inches="tight")
          plt.close()