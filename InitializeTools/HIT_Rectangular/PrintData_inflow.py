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
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import cantera as ct

# Input parameters ---------------------------------------
# Input flame
cn = "Solu_H2_990K_1atm_phi0.35_Uin35.0.csv"
df = pd.read_csv(cn)

# Mechanism
mech    = "h2o2.yaml"
# Species for initilization
spn_out = ["N2", "H", "O2", "O", "OH", "H2", "H2O", "HO2", "H2O2"]
insert_flame = False    # True - Y, T = Cantera solution
                        # False - Y, T = inlet
impose_hit   = True     # True - no velocity fluctuations
                        # False - interpolate from HIT box solution
if_correct_N2 = True    # N2 = 1 - rest
if_replace_fuel = True
fd_din = "/home/zisen347/scratch/scoping_runs/NGA/101_BoxHIT_ZeroH2_1"
fn_din = os.path.join(fd_din, "ufs:data_3.700E-03")
fn_cin = os.path.join(fd_din, "ufs:config.hit")

iflame = np.argmax(df.temperature > (df.temperature[0]+400.))
xflame = df.x[iflame]
gradT_r = (df.temperature[iflame+1] - df.temperature[iflame]) / (df.x[iflame+1] - df.x[iflame])
gradT_l = (df.temperature[iflame] - df.temperature[iflame-1]) / (df.x[iflame] - df.x[iflame-1])
gradT = 0.5 * (gradT_r + gradT_l)
lf = (df.temperature.values[-1] - df.temperature[0]) / gradT
print("Laminar flame thickness [mm]: ", lf*1000)

Lflame = 4.06E-4 * 14.3 * 2.45 * 2.718 
nx_out = 96*7; xper_out = 0
ny_out = 96;   yper_out = 1
nz_out = 96;   zper_out = 1
Lx_out = 4.315E-3*7  # Lflame * 1.0
Ly_out = 4.315E-3  #Lx_out * (ny_out / nx_out)
Lz_out = 4.325E-3  #Lx_out * (nz_out / nx_out)
print("Laminar flame thickness resolved by ", lf / (Lx_out/nx_out), " points.")

new_pos = xflame + 0.0 * Lflame
U_in    = df["velocity"][0]
P_atm   = df["pressure"][0] / ct.one_atm
P_in    = df["pressure"][0]
T_in    = df["temperature"][0]
rho_in  = df["density"][0]

x_out = np.linspace(0, Lx_out, nx_out+1)
y_out = np.linspace(0, Ly_out, ny_out+1)
z_out = np.linspace(0, Lz_out, nz_out+1)
xcell_out = (x_out[0:-1] + x_out[1:]) / 2.
ycell_out = (y_out[0:-1] + y_out[1:]) / 2.
zcell_out = (z_out[0:-1] + z_out[1:]) / 2.
xx_out, yy_out, zz_out = np.meshgrid(x_out, y_out, z_out, indexing='ij')
xxcell_out, yycell_out, zzcell_out = np.meshgrid(xcell_out, ycell_out, zcell_out, indexing='ij')

#%%
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

# ---------- Writer ----------
def NGA_writer(data, dataname, configname):
    # Open config
    f = open(configname, "w")
    print("NGA_writer - config file")
    # Simulation name
    simname = data["simname"]
    print("Writing simname: ", data["simname"])    
    bin_simname = [ord(simname[i]) for i in range(len(simname))] + [ord(" ") for i in range(64-len(simname))]
    np.asarray(bin_simname).astype("int8").tofile(f)
    # icyl    
    coord = 0
    np.asarray([coord]).astype("int32").tofile(f)
    print("Writing icyl: ", coord)
    # xper, yper, zper 
    xper = data["xper"]; 
    np.asarray([xper]).astype("int32").tofile(f)
    yper = data["yper"]; 
    np.asarray([yper]).astype("int32").tofile(f)
    zper = data["zper"]; 
    np.asarray([zper]).astype("int32").tofile(f)
    print("Writing xper, yper, zper: ", xper, yper, zper)
    # nx, ny, nz 
    dims = np.asarray([data["nx"], data["ny"], data["nz"]])
    dims.astype("int32").tofile(f)
    print("Writing nx, ny, nz", dims)
    # x, y, z 
    data["x"].astype("float64").tofile(f)
    data["y"].astype("float64").tofile(f)
    data["z"].astype("float64").tofile(f)
    print("Writing x, y, z: ", data["x"], data["y"], data["z"])
    # Close config 
    f.close()
    
    # Open data file
    f = open(dataname, "w")
    print("NGA_writer - data file")
    # nx, ny, nz 
    dims.astype("int32").tofile(f)
    print("Writing nx, ny, nz: ", dims)
    # dt, time 
    np.asarray([data["dt"]]).astype("float64").tofile(f)
    np.asarray([data["time"]]).astype("float64").tofile(f)
    print("Writing dt, time: ", data["dt"], data["time"])
    # varnames 
    for iVar in range(dims[3]):
        varname = data["varnames8"][iVar]
        bin_varname = [ord(varname[i]) for i in range(8)]
        np.asarray(bin_varname).astype("int8").tofile(f)
        print("Writing varnames", iVar, varname)    
    # Field data
    for fn in data["varnames"]:
        res = data[fn]
        res = res.flatten(order='F')
        res.astype("float64").tofile(f)
        print("Writing fields (max/min)", fn, np.amax(res), np.amin(res))
    # Close data file 
    f.close()
    return

# ---------- Inflow turbulence writer ----------
dhit = NGA_reader(fn_din, fn_cin)
def NGA_inflowturb_writer(data, dataname, tcur=0.0):
    f = open(dataname, "w")
    varnames = ["U", "V", "W"]
    # nx, ny, nz, nVar
    dims = np.asarray([data["nx"], data["ny"], data["nz"], int(len(varnames))])
    print(dims)
    dims.astype("int32").tofile(f)
    # dt, time
    time = (data["x"][-1] - data["x"][0]) / np.mean(data["U"])
    dt = time / data["nx"]
    np.asarray([dt]).astype("float64").tofile(f)
    np.asarray([time]).astype("float64").tofile(f)
    # varnames
    for iVar, vn in enumerate(varnames):
        varname8 = vn.ljust(8)
        bin_varname = [ord(varname8[i]) for i in range(8)]
        np.asarray(bin_varname).astype("int8").tofile(f)
    # icyl
    np.asarray([0]).astype("int32").tofile(f)
    # y, z
    y = data["y"]
    z = data["z"]
    y = y.flatten(order='F')
    z = z.flatten(order='F')
    y.astype("float64").tofile(f)
    z.astype("float64").tofile(f)

    # Write
    rt = tcur / time
    rt = 1 - (rt - int(rt))
    nx = data["nx"]
    imid = np.amax([int(rt * nx), 0])
    imid = np.amin([imid, nx-1])
    sorder = list(reversed(range(0,imid+1))) + list(reversed(range(imid+1,nx))) 
    print(sorder)

    # field - flipped around x direction
    vns = ["U", "V", "W"]
    #for ix in reversed(range(0, data["nx"])):
    for ix in sorder:
        buf = np.zeros((data["ny"],data["nz"], len(vns)), order="F")
        for iv, fn in enumerate(vns):
            buf[:,:,iv] = data[fn][ix,:,:]
            res  = data[fn][ix,:,:]
            output = res.flatten(order='F')
            output.astype("float64").tofile(f)

    f.close()
    return 
NGA_inflowturb_writer(dhit, "ufs:inflow.dat", tcur=4.440E-4)
#%%