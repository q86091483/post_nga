#%%
import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = "./tmp"
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d
import pandas as pd
import cantera as ct

# Input parameters ---------------------------------------
# Input flame
case_name = "103_RectTurb_flame1D_990K_eta1000"
case_folder = "/scratch/zisen347/scoping_runs/NGA" 
cn = os.path.join(case_folder, case_name)
fn_config = os.path.join(cn, "ufs:config.flame1D")

# ---------- Reader ----------
def NGA_reader(dataname, configname):
    data={}
    print("NGA_reader - config file")
    # Open config file
    f = open(configname)   
    # Simulation name
    intname = np.fromfile(f, dtype='int8', count=64)
    simname = "".join([chr(intname[i]) for i in range(64)])
    data["simname"] = simname
    print("Reading simname: ", data["simname"])    
    # Coordinate type
    data["coord"] = np.fromfile(f, dtype='int32', count=1)[0]
    print("Reading icyl: ", data["coord"]) 
    # BC periodicity
    data["xper"] = np.fromfile(f, dtype='int32', count=1)[0]
    data["yper"] = np.fromfile(f, dtype='int32', count=1)[0]
    data["zper"] = np.fromfile(f, dtype='int32', count=1)[0]
    print("Reading xper, yper, zper: ", data["xper"], data["yper"], data["zper"]) 
    # Numerical dimension
    dims = np.fromfile(f, dtype='int32', count=3)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    print("Reading nx, ny, nz: ", nx, ny, nz) 
    # Problem dimension
    data["x"] = np.fromfile(f, dtype='float64', count=nx+1)
    data["y"] = np.fromfile(f, dtype='float64', count=ny+1)
    data["z"] = np.fromfile(f, dtype='float64', count=nz+1)
    print("Reading coordinate x: ", data["x"])
    print("Reading coordinate y: ", data["y"])
    print("Reading coordinate z: ", data["z"])
    data["Lx"]=data["x"][-1] - data["x"][0]
    data["Ly"]=data["y"][-1] - data["y"][0]
    data["Lz"]=data["z"][-1] - data["z"][0]
    print("Reading Lx, Ly, Lz: ", data["Lx"], data["Ly"], data["Lz"])
    # Close config file
    f.close()
    
    # Open data file
    f = open(dataname)    
    print("NGA_reader - data file")
    # Dimensions of the data
    dims = np.fromfile(f, dtype='int32', count = 4)
    data["nx"] = nx
    data["ny"] = ny
    data["nz"] = nz
    data["nVar"] = dims[3]
    print("Reading nx, ny, nz: ", data["nx"], data["ny"], data["nz"])
    nsize = data["nx"] * data["ny"] * data["nz"]
    # Time
    data["dt"]   = np.fromfile(f, dtype='float64', count=1)[0]
    data["time"] = np.fromfile(f, dtype='float64', count=1)[0]
    print("Reading dt, time:", data["dt"], data["time"])
    # Variable names
    data["varnames"] = []
    data["varnames8"] = []
    for iVar in range(data["nVar"]):
        intname = np.fromfile(f, dtype='int8', count=8)
        varname = "".join([chr(intname[i]) for i in range(8)])
        print("Reading varnames", iVar, varname)
        data["varnames"].append(varname.strip())
        data["varnames8"].append(varname)
    print("Reading varnames8: ", data["varnames8"])    
    # Field data
    for fn in data["varnames"]:
        data[fn] = np.fromfile(f, dtype='float64', count=nsize).reshape((dims[0], dims[1], dims[2]),order="F")
        print("Reading ", fn, " (max/min):", np.amax(data[fn]), np.amin(data[fn]))
    # Close data file 
    f.close()
    return data

npx = 2; npy = 2; field_names = ["RHO", "T", "U", "P"]
ts = [0.0, 3.000E-4, 6.000E-04, 6.400E-04, 6.500E-04, 6.600E-04, 6.750E-04, 7.010E-04, 7.320E-04]
ts = [0.0, 3.000E-4, 6.000E-04, 6.400E-04, 6.500E-04, 6.600E-04, 6.750E-04, 7.010E-04, 7.320E-04]
ts = [0.0, 3E-4, 6.130E-4, 7.790E-04, 8.692E-4, 2.164E-03] #9.760E-04, 1.014E-03]
ts = [0.0, 3.100E-04, 6.100E-04, 6.300E-04, 6.800E-04, 8E-4, 8.900E-04] 
ts = [0.0, 2.120E-04]
ts_990K = [0.0, 1.610E-04, 2.390E-04, 2.740E-04, 3.150E-04, 3.610E-04, 
      4.390E-04, 4.900E-04, 5.630E-04, 6.120E-04, 6.580E-04, 6.900E-04, 
      8.360E-04, 9.300E-04, 1.080E-03, 1.116E-03, 1.237E-3, 1.305E-03, 1.416E-03, 1.600E-03, 1.758E-03, 2.000E-03, 2.2E-3, 2.597E-03, 2.834E-03,3.005E-03,
      3.188E-03]
ts_1D = [0.0, 1E-5, 2E-5, 3.001E-5, 5.001E-5, 8.000E-5, 1E-4, 2.9E-4, 5.800E-04, 7.5E-4, 1.137E-03, ]
ts = ts_990K


fig, axs = plt.subplots(figsize=(10,8), ncols=npx, nrows=npy)
fns = ["data.init.flame1D"]
for t in ts[1:]:
    fns.append("ufs:data_" + "%.3E"%(t))
colors = cm.turbo(np.linspace(0, 1, len(ts)))

for i, fn in enumerate(fns):
    fn_data = os.path.join(cn, fn)
    res = NGA_reader(dataname=fn_data, configname=fn_config)
    res["P"] = res["P"] - 101325
    nx  = res["nx"]; ny = res["ny"]; nz = res["nz"]

    x = res["x"][0:nx]; y = res["y"][0:ny]; z = res["z"][0:nz]
    for ipx in range(0, npx):
        for ipy in range(0, npy):
            ax = axs[ipy, ipx]; idn = ipx*npx + ipy; fdn = field_names[idn]
            phi = res[fdn][0:nx,5,5] 
            ax.plot(x, phi, label="%.2E"%(ts[i]), color=colors[i], linewidth=2.5)
            labelsize=20
            ax.set_ylabel(fdn, fontsize=labelsize)
            ax.tick_params(axis='both', which='major', labelsize=labelsize-4)
            ax.tick_params(axis='both', which='minor', labelsize=labelsize-4)
            if idn == 3:
                ax.legend(loc="lower right")
ax2 = ax.twinx()
plt.savefig("laminar1D.png", dpi=300, bbox_inches="tight")
#%%