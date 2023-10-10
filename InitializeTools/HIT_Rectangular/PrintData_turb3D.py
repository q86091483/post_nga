#%%
import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = "./tmp"
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import cantera as ct

# Input parameters ---------------------------------------
# Input flame
cn = "Solu_H2_950K_1atm_phi0.4.csv"
df = pd.read_csv(cn)
# Mechanism
mech    = "h2o2.yaml"
# Species for initilization
spn_out = ["N2", "H", "O2", "O", "OH", "H2", "H2O", "HO2", "H2O2"]
insert_flame  = False    # True - Y, T = Cantera solution
                        # False - Y, T = inlet
if_hit  = False          # True - no velocity fluctuations
                        # False - interpolate from HIT box solution
if_correct_N2 = True    # N2 = 1 - rest
ratio_U = 10.0
fn_din  = "./data.hitbox.6"
fn_cin  = "./config.hitbox.6"

iflame = np.argmax(df.temperature > (df.temperature[0]+400.))
xflame = df.x[iflame]
gradT_r = (df.temperature[iflame+1] - df.temperature[iflame]) / (df.x[iflame+1] - df.x[iflame])
gradT_l = (df.temperature[iflame] - df.temperature[iflame-1]) / (df.x[iflame] - df.x[iflame-1])
gradT = 0.5 * (gradT_r + gradT_l)
lf = (df.temperature.values[-1] - df.temperature[0]) / gradT
print("Laminar flame thickness [mm]: ", lf*1000)

#%%
L_to_lf = 5; nf = 8
nx_out = 12 * nf * L_to_lf
ny_out =  1 * nf * L_to_lf
nz_out =  1 * nf * L_to_lf
Lx_out = 12 * lf * L_to_lf
Ly_out =  1 * lf * L_to_lf #Lx_out * (ny_out / nx_out)
Lz_out =  1 * lf * L_to_lf #Lx_out * (nz_out / nx_out)
print("Lx: ", Lx_out*1000, " [mm],  nx = ", nx_out)
print("Ly: ", Ly_out*1000, " [mm],  ny = ", ny_out)
print("Lz: ", Lz_out*1000, " [mm],  nz = ", nz_out)
print("Laminar flame thickness resolved by ", lf / (Lx_out/nx_out), " points.")
Lt = 0.2 * Ly_out

new_pos = xflame
U_in    = df["velocity"][0]
P_atm   = df["pressure"][0] / ct.one_atm
P_in    = df["pressure"][0]
T_in    = df["temperature"][0]
rho_in  = df["density"][0]

x_out = np.linspace(0, Lx_out, nx_out+1)
y_out = np.linspace(0, Ly_out, ny_out+1)
z_out = np.linspace(0, Lz_out, nz_out+1)
xx_out, yy_out, zz_out = np.meshgrid(x_out, y_out, z_out, indexing='ij')
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
    dims = np.asarray([data["nx"], data["ny"], data["nz"], data["nVar"]])
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

# 1. Write config & data.init ---------------------------------------
dout = {}
dout["RHO"] = np.zeros((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["U"]   = np.zeros((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["V"]   = np.zeros((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["W"]   = np.zeros((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["T"]   = np.zeros((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["P"]   = np.zeros((nx_out,ny_out,nz_out),dtype="float64",order='F')
for spn in spn_out:
    dout[spn] = np.zeros((nx_out,ny_out,nz_out),dtype="float64",order='F')

# RHO, T, P, mean velocity
if insert_flame:
    xinterp = df.x.values
    if xinterp[0] > x_out[0]:
        xinterp[0] = x_out[0] - 1.0
    if xinterp[1] < x_out[-1]:
        xinterp[1] = x_out[-1] + 1.0

    f_T = interp1d(xinterp, df.temperature)
    f_P = interp1d(xinterp, df.pressure)
    f_d = interp1d(xinterp, df.density)
    f_U = interp1d(xinterp, df.velocity)

    T_1D = f_T(x_out)
    P_1D = f_P(x_out)
    d_1D = f_d(x_out)
    U_1D = f_U(x_out)

    Y_1D = {}
    for spn in spn_out:
        f_Y = interp1d(xinterp, df[spn].values)
        Y_1D[spn] = f_Y(x_out)
    for j in range(0, ny_out):
        for k in range(0, nz_out):
            dout["RHO"][:,j,k] = d_1D[0:nx_out]
            dout["P"][:,j,k] = P_1D[0:nx_out]
            dout["T"][:,j,k] = T_1D[0:nx_out]
            dout["U"][:,j,k] = U_1D[0:nx_out]
            for spn in spn_out:
                dout[spn][:,j,k] =  Y_1D[spn][0:nx_out] #* d_1D[0:nx_out]  
else:
    dout["RHO"][:,:,:] = rho_in
    dout["P"][:,:,:] = P_in
    dout["T"][:,:,:] = T_in
    dout["U"][:,:,:] = U_in * ratio_U
    for spn in spn_out:
        dout[spn][:,:,:] = df[spn].values[0]

dout["simname"]     = "planar flame"
dout["coord"]       = 0
dout["xper"]        = 0 
dout["yper"]        = 1
dout["zper"]        = 1
dout["Lx"]          = Lx_out
dout["Ly"]          = Ly_out
dout["Lz"]          = Lz_out
dout["nx"]          = nx_out
dout["ny"]          = ny_out
dout["nz"]          = nz_out
dout["x"]           = x_out
dout["y"]           = y_out
dout["z"]           = z_out
dout["dx"]          = Lx_out / (nx_out - 1)
dout["dy"]          = Ly_out / (ny_out - 1)
dout["dz"]          = Lz_out / (nz_out - 1)
dout["dt"]          = 1E-10
dout["time"]        = 0.0
dout["varnames"]    = ["RHO", "U", "V", "W", "T", "P"]
dout["varnames"]    = dout["varnames"] + spn_out
dout["nVar"]        = len(dout["varnames"])
dout["varnames8"]   = []
for str in dout.keys():
    dout["varnames8"].append(str.ljust(8))

NGA_writer(data=dout, dataname="data.init.turb3D", configname="config.turb3D")

res = NGA_reader(dataname="data.init.turb3D", configname="config.turb3D")

fn = "U"
fig, ax = plt.subplots(figsize=(3,2.4))
ax.plot(res["x"][0:nx_out], res[fn][0:nx_out,        0,        0], 'r')
ax.plot(res["x"][0:nx_out], res[fn][0:nx_out,        0, nz_out-1], 'b-')
ax.plot(res["x"][0:nx_out], res[fn][0:nx_out, ny_out-1,        0], 'm-.')
ax.plot(res["x"][0:nx_out], res[fn][0:nx_out, ny_out-1, nz_out-1], 'g:')
ax.set_title(fn)

#%%