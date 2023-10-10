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
cn = "Solu_H2_300K_1atm_phi0.4.csv"
df = pd.read_csv(cn)
# Mechanism
mech    = "h2o2.yaml"
# Species for initilization
spn_out = ["N2", "H", "O2", "O", "OH", "H2", "H2O", "HO2", "H2O2"]
if_correct_N2 = True    # N2 = 1 - rest
insert_flame  = True    # True - Y, T = Cantera solution
                        # False - Y, T = inlet
if_hit  = True          # True - no velocity fluctuations
                        # False - interpolate from HIT box solution
fn_din  = "./data.hitbox.6"
fn_cin  = "./config.hitbox.6"

fig, ax = plt.subplots()
ax.plot(df.x, df.temperature)
iflame = np.argmax(df.temperature > (df.temperature[0]+400.))
xflame = df.x[iflame]
gradT_r = (df.temperature[iflame+1] - df.temperature[iflame]) / (df.x[iflame+1] - df.x[iflame])
gradT_l = (df.temperature[iflame] - df.temperature[iflame-1]) / (df.x[iflame] - df.x[iflame-1])
gradT = 0.5 * (gradT_r + gradT_l)
lf = (df.temperature.values[-1] - df.temperature[0]) / gradT
print("Laminar flame thickness [mm]: ", lf*1000)

nx_out = 1180
ny_out = 10
nz_out = 10
Lx_out = 0.0222538
Ly_out = Lx_out * (ny_out / nx_out)
Lz_out = Lx_out * (nz_out / nx_out)
print("Laminar flame thickness resolved by ", lf / (Lx_out/nx_out), " points.")

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

# ---------- Reader ----------
def NGA_reader(dataname, configname):
    data={}

    # config 
    f = open(configname)
    
    # Simulation name
    intname = np.fromfile(f, dtype='int8', count=64)
    simname = "".join([chr(intname[i]) for i in range(64)])
    data["simname"] = simname
    
    # Coordinate type
    data["coord"] = np.fromfile(f, dtype='int32', count=1)[0]
    
    # BC periodicity
    data["xper"] = np.fromfile(f, dtype='int32', count=1)[0]
    data["yper"] = np.fromfile(f, dtype='int32', count=1)[0]
    data["zper"] = np.fromfile(f, dtype='int32', count=1)[0]
    
    # Numerical dimension
    dims = np.fromfile(f, dtype='int32', count=3)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    
    # Problem dimension
    data["x"] = np.fromfile(f, dtype='float64', count=nx+1)
    data["y"] = np.fromfile(f, dtype='float64', count=ny+1)
    data["z"] = np.fromfile(f, dtype='float64', count=nz+1)

    data["Lx"]=data["x"][-1]
    data["Ly"]=data["y"][-1]
    data["Lz"]=data["z"][-1]

    f.close()
    
    # data
    f = open(dataname)
    
    # Dimensions of the data
    dims = np.fromfile(f, dtype='int32', count=4)
    data["nx"] = dims[0]
    data["ny"] = dims[1]
    data["nz"] = dims[2]
    data["nVar"] = dims[3]
    nsize=(1+data["nx"])*(1+data["ny"])*(1+data["nz"])
  
    # Time
    data["dt"]   = np.fromfile(f, dtype='float64', count=1)[0]
    data["time"] = np.fromfile(f, dtype='float64', count=1)[0]
    
    # Variable names
    data["varnames"] = []
    data["varnames8"] = []
    for iVar in range(data["nVar"]):
        intname = np.fromfile(f, dtype='int8', count=8)
        varname = "".join([chr(intname[i]) for i in range(8)])
        print("Reading ", iVar, varname)
        data["varnames"].append(varname.strip())
        data["varnames8"].append(varname)
        
    # Real data
    for fn in data["varnames"]:
        data[fn] = np.fromfile(f, dtype='float64', count=nsize).reshape((nx,ny,nz))
    
    f.close()
    
    return data

# ---------- Writer ----------
def NGA_writer(data, dataname, configname):
    # Open config
    f = open(configname, "w")
    # Simulation name
    simname = data["simname"]
    bin_simname = [ord(simname[i]) for i in range(len(simname))] + [ord(" ") for i in range(64-len(simname))]
    np.asarray(bin_simname).astype("int8").tofile(f)
    # icyl    
    coord = 0
    np.asarray([coord]).astype("int32").tofile(f)
    # xper, yper, zper 
    xper = data["xper"]; 
    np.asarray([xper]).astype("int32").tofile(f)
    yper = data["yper"]; 
    np.asarray([yper]).astype("int32").tofile(f)
    zper = data["zper"]; 
    np.asarray([zper]).astype("int32").tofile(f)
    # nx, ny, nz 
    dims = np.asarray([data["nx"], data["ny"], data["nz"]])
    dims.astype("int32").tofile(f)
    # x, y, z 
    X = np.linspace(0, data["Lx"], data["nx"]+1)
    X.astype("float64").tofile(f)
    Y = np.linspace(0, data["Ly"], data["ny"]+1)
    Y.astype("float64").tofile(f)
    Z = np.linspace(0, data["Lz"], data["nz"]+1)
    Z.astype("float64").tofile(f)
    # Close config 
    f.close()
    
    # Open data file
    f = open(dataname, "w")
    # nx, ny, nz 
    dims = np.asarray([data["nx"], data["ny"], data["nz"], data["nVar"]])
    dims.astype("int32").tofile(f)
    # dt, time 
    np.asarray([data["dt"]]).astype("float64").tofile(f)
    np.asarray([data["time"]]).astype("float64").tofile(f)
    # varnames 
    for iVar in range(dims[3]):
        varname = data["varnames8"][iVar]
        bin_varname = [ord(varname[i]) for i in range(8)]
        np.asarray(bin_varname).astype("int8").tofile(f)
        print("Writing ", iVar, varname)    
    # Field data
    for fn in data["varnames"]:
        res = data[fn]
        res = res.flatten(order='F')
        res.astype("float64").tofile(f)
    # Close data file 
    f.close()
    return

din = NGA_reader(fn_din, fn_cin)

# 1. Write config & data.init ---------------------------------------
dout = {}
dout["RHO"] = np.zeros((nx_out+1,ny_out+1,nz_out+1),dtype="float64",order='F')
dout["U"]   = np.zeros((nx_out+1,ny_out+1,nz_out+1),dtype="float64",order='F')
dout["V"]   = np.zeros((nx_out+1,ny_out+1,nz_out+1),dtype="float64",order='F')
dout["W"]   = np.zeros((nx_out+1,ny_out+1,nz_out+1),dtype="float64",order='F')
dout["T"]   = np.zeros((nx_out+1,ny_out+1,nz_out+1),dtype="float64",order='F')
dout["P"]   = np.zeros((nx_out+1,ny_out+1,nz_out+1),dtype="float64",order='F')
for spn in spn_out:
    dout[spn] = np.zeros((nx_out+1,ny_out+1,nz_out+1),dtype="float64",order='F')

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
            dout["RHO"][:,j,k] = d_1D
            dout["P"][:,j,k] = P_1D
            dout["T"][:,j,k] = T_1D
            dout["U"][:,j,k] = U_1D
            for spn in spn_out:
                dout[spn][:,j,k] = Y_1D[spn]
else:
    dout["RHO"][:,:,:] = rho_in
    dout["P"][:,:,:] = P_in
    dout["T"][:,:,:] = T_in
    dout["U"][:,:,:] = U_in
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
dout["dx"]          = Lx_out / (nx_out - 1)
if ny_out == 1:
    dout["dy"]      = Ly_out
else:
    dout["dy"]      = Ly_out / (ny_out - 1)
if nz_out == 1:
    dout["dz"]      = Lz_out 
else:
    dout["dz"]       = Lz_out / (nz_out - 1)
dout["nSpec"]       = len(spn_out)
dout["dt"]          = 1E-10
dout["time"]        = 0.0
dout["varnames"]    = ["RHO", "U", "V", "W", "T", "P"]
dout["varnames"]    = dout["varnames"] + spn_out
dout["nVar"]        = len(dout["varnames"])
dout["varnames8"]   = []
for str in dout.keys():
    dout["varnames8"].append(str.ljust(8))

NGA_writer(data=dout, dataname="data.init.flame1D", configname="config.flame1D")
#%%
res = NGA_reader(dataname="data.init.flame1D", configname="config.flame1D")
fig, ax = plt.subplots()
ax.plot(res["x"], res["T"][:,0,0])
#%%

