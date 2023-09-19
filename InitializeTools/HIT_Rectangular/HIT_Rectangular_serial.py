#%%
import sys
import os
os.environ['MPLCONFIGDIR'] = "./tmp"
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def NGA_reader(filename, configname):
    data={}
    
    f = open(configname)
    
    # Simulation name
    intname = np.fromfile(f, dtype='int8', count=64)
    simname = "".join([chr(intname[i]) for i in range(64)])
    data["simname"] = simname
    
    # Coordinate type
    data["coord"] = np.fromfile(f, dtype='int32', count=1)[0]
    #print(data["coord"])
    
    # BC periodicity
    data["xper"] = np.fromfile(f, dtype='int32', count=1)[0]
    data["yper"] = np.fromfile(f, dtype='int32', count=1)[0]
    data["zper"] = np.fromfile(f, dtype='int32', count=1)[0]
    #print(data["xper"], data["yper"], data["zper"])
    
    # Numerical dimension
    dims = np.fromfile(f, dtype='int32', count=3)
    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    
    # Problem dimension
    data["Lx"]=np.fromfile(f, dtype='float64', count=nx+1)[-1]
    data["dx"]=data["Lx"] / nx
    data["Ly"]=data["dx"]
    data["Lz"]=data["dx"]
    
    f.close()
    
    f = open(filename)
    
    # Dimensions of the data
    dims = np.fromfile(f, dtype='int32', count=4)
    data["nx"]=dims[0]
    data["ny"]=dims[1]
    data["nz"]=dims[2]
    data["nVar"]=dims[3]
    data["nSpec"]=0
    nsize=data["nx"]*data["ny"]*data["nz"]
    data["X"] = np.linspace(0, data["Lx"], data["nx"])
    
    # Time
    data["dt"]=np.fromfile(f, dtype='float64', count=1)[0]
    data["time"]=np.fromfile(f, dtype='float64', count=1)[0]
    #print(data["time"])
    
    # Variable names
    data["varnames"] = []
    data["varnames8"] = []
    for iVar in range(data["nVar"]):
        intname = np.fromfile(f, dtype='int8', count=8)
        varname = "".join([chr(intname[i]) for i in range(8)])
        data["varnames"].append(varname.strip())
        data["varnames8"].append(varname)
        
    # Real data
    for fn in data["varnames"]:
        data[fn] = np.fromfile(f, dtype='float64', count=nsize).reshape((dims[0],dims[0],dims[0]))
    
    f.close()
    
    return data

def NGA_writer(data, filename, configname):
    # Write the config file with the information from data
    f = open(configname, "w")
    
    # Simulation name
    simname = data["simname"]
    bin_simname = [ord(simname[i]) for i in range(len(simname))] + [ord(" ") for i in range(64-len(simname))]
    np.asarray(bin_simname).astype("int8").tofile(f)
    
    # Coordinate type
    coord = 0
    np.asarray([coord]).astype("int32").tofile(f)
    
    # BC periodicity
    xper = data["xper"]; 
    np.asarray([xper]).astype("int32").tofile(f)
    yper = data["yper"]; 
    np.asarray([yper]).astype("int32").tofile(f)
    zper = data["zper"]; 
    np.asarray([zper]).astype("int32").tofile(f)
    
    # Numerical dimension
    dims = np.asarray([data["nx"], data["ny"], data["nz"]])
    dims.astype("int32").tofile(f)
    
    # Problem dimension
    X = np.linspace(0, data["Lx"], data["nx"]+1)
    X.astype("float64").tofile(f)
    Y = np.linspace(0, data["Ly"], data["ny"]+1)
    Y.astype("float64").tofile(f)
    Z = np.linspace(0, data["Lz"], data["nz"]+1)
    Z.astype("float64").tofile(f)
    
    # mask?
    mask=np.zeros((data["nx"],data["ny"]))
    mask.astype("int32").tofile(f)
    
    f.close()
    
    # Write the data.init file with the information from data
    f = open(filename, "w")
    
    # Dimensions of the data
    dims = np.asarray([data["nx"], data["ny"], data["nz"], data["nVar"]])
    dims.astype("int32").tofile(f)
    
    # Time
    np.asarray([data["dt"]]).astype("float64").tofile(f)
    np.asarray([data["time"]]).astype("float64").tofile(f)
    
    # Variable names
    for iVar in range(dims[3]):
        varname = data["varnames8"][iVar]
        bin_varname = [ord(varname[i]) for i in range(8)]
        np.asarray(bin_varname).astype("int8").tofile(f)
        
    # Real data
    for fn in data["varnames"]:
        res = data[fn]
        res = res.flatten(order='F')
        res.astype("float64").tofile(f)
    
    f.close()
    return

# Input
nbox   = 5
#fn_din = "/scratch/b/bsavard/zisen347/cases/103_RectTurb/data.init"
fn_din = "./data.hitbox.6"
#fn_cin = "/scratch/b/bsavard/zisen347/cases/103_RectTurb/config"
fn_cin = "./config.hitbox.6"

din = NGA_reader(fn_din, fn_cin)
x_in = din["X"];
nx_in = x_in.shape[0]; 
Lx_in = x_in[-1] - x_in[0]

# Target mesh
Lx_out = nbox * Lx_in;  nx_out = nbox * int(1.0*nx_in);   dx_out = Lx_out / nx_out 
Ly_out = 1.0 * Lx_in;   ny_out = int(1.0*nx_in); dy_out = Ly_out / ny_out
Lz_out = 1.0 * Lx_in;   nz_out = int(1.0*nx_in); dz_out = Lz_out / nz_out
x_out = np.linspace(0, Lx_out, nx_out)
y_out = np.linspace(0, Ly_out, ny_out)
z_out = np.linspace(0, Lz_out, nz_out)
xx_out, yy_out, zz_out = np.meshgrid(x_out, y_out, z_out, indexing='ij')

# Repeat input mesh to cover the target mesh
nrx = int(x_out[-1]/x_in[-1]) + 1
nry = int(y_out[-1]/x_in[-1]) + 1
nrz = int(z_out[-1]/x_in[-1]) + 1
x_rep = np.linspace(x_in[0], x_in[-1]*nrx, nx_in*nrx)
y_rep = np.linspace(x_in[0], x_in[-1]*nry, nx_in*nry)
z_rep = np.linspace(x_in[0], x_in[-1]*nrz, nx_in*nrz)
xx_rep, yy_rep, zz_rep = np.meshgrid(x_rep,y_rep,z_rep, indexing='ij')

dout = {}
field_names = din["varnames"]
for kn in field_names:
    phi_in = np.tile(din[kn], (nrx,nry,nrz))
    interp = RegularGridInterpolator((x_rep, y_rep, z_rep), phi_in)
    dout[kn] = interp((xx_out, yy_out, zz_out))

U_const   = 10.
Wmix      = 2.897E-2
P_const   = 101325.
T_const   = 300.
R_cst     = 8.314462
rho_const = 1.174
P_const   = T_const * (R_cst/Wmix) * rho_const

dout["RHO"] = np.ones((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["U"]   = np.ones((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["V"]   = np.ones((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["W"]   = np.ones((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["T"]   = np.ones((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["P"]   = np.ones((nx_out,ny_out,nz_out),dtype="float64",order='F')
dout["ZMIX"] = np.ones((nx_out,ny_out,nz_out),dtype="float64",order='F')

dout["RHO"][:,:,:] = rho_const
dout["V"][:,:,:] = 0
dout["W"][:,:,:] = 0
dout["P"][:,:,:] = P_const
dout["T"][:,:,:] = T_const
dout["ZMIX"][:,:,:] = 1.0

phi_in = np.tile(din["U"], (nrx,nry,nrz))
interp = RegularGridInterpolator((x_rep, y_rep, z_rep), phi_in)
dout["U"] = interp((xx_out, yy_out, zz_out))
rt=0.5*(1+np.tanh((x_out-x_out[-1]*0.15)/0.4)); plt.plot(rt)
for i, xi in enumerate(rt):
    dout["U"][i,:,:] = dout["U"][i,:,:]*rt[i] + U_const


dout["simname"]     = din["simname"]
dout["coord"]       = din["coord"]
dout["xper"]        = 0 #din["xper"]
dout["yper"]        = din["yper"]
dout["zper"]        = din["zper"]
dout["Lx"]          = Lx_out
dout["Ly"]          = Ly_out
dout["Lz"]          = Lz_out
dout["nx"]          = nx_out
dout["ny"]          = ny_out
dout["nz"]          = nz_out
dout["dx"]          = dx_out
dout["dy"]          = dy_out
dout["dz"]          = dz_out
dout["nVar"]        = din["nVar"]
dout["nSpec"]       = 0
dout["dt"]          = din["dt"]
dout["time"]        = din["time"]
dout["varnames"]    = din["varnames"]
dout["varnames8"]   = din["varnames8"]

print("Density      : ", np.mean(dout["RHO"]))
print("Temperature  : ", np.mean(dout["T"]))
print("Pressure     : ", np.mean(dout["P"]))
print("Velocity U   : ", np.mean(dout["U"]))
print("Velocity V   : ", np.mean(dout["V"]))
print("Velocity W   : ", np.mean(dout["W"]))
print("MW_mix       : ", Wmix)
print("Lx_out       : ", Lx_out)
print("Ly_out       : ", Ly_out)
print("Lz_out       : ", Lz_out)

NGA_writer(data=dout, filename="data.init", configname="config")
print("Writing finished.")
#%%