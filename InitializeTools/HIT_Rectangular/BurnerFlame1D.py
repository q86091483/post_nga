#%%
import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = "./tmp"
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import cantera as ct

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

# Generate a 1D flame solution
Lx      = 0.04
U_in    = 13
P_atm   = 1.
T_in    = 900
eqrt    = 0.4
P_in    = P_atm * ct.one_atm
mtot    = eqrt*2.0 + (1.0 + 3.76)
X_in        = {}
X_in["H2"]  = eqrt*2.0 / mtot
X_in["O2"]  = 1.0 / mtot
X_in["N2"]  = 3.76 / mtot
gas = ct.Solution('h2o2.yaml')
gas.TPX = T_in, P_in, X_in
gas0D = ct.Solution('h2o2.yaml')
gas0D.TPX = T_in, P_in, X_in
#gas.equilibrate("HP")
print("Unburned gas: ", gas())
#%%

# 0. Homogeneous ignition delay time
if (True):
    r = ct.IdealGasConstPressureReactor(gas0D)
    sim = ct.ReactorNet([r])

    t_end = 0.4
    t = 0.0
    told = 0.0
    states = ct.SolutionArray(r.thermo, extra=['t', 'hrr', 'dt'])
    while t < t_end:
        t = sim.step()
        states.append(r.thermo.state,
                    t=t,
                    hrr=np.dot(gas.net_production_rates, gas.partial_molar_enthalpies),
                    dt=t-told)
        told = t
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot(states.t, states.T, 'r', linewidth=3.0)
    ax.set_title("0D")
    ax.set_ylabel("T[K]")
    ax.set_xlabel("t[s]")
#%%
f = ct.BurnerFlame(gas, width=Lx)
mdot = U_in * gas.density
f.burner.mdot = mdot
f.set_refine_criteria(ratio=3.0, slope=0.05, curve=0.1)
f.transport_model = 'mixture-averaged'
loglevel=1
f.solve(loglevel, auto=True)

if "native" in ct.hdf_support():
    output = Path() / "burner_flame.h5"
else:
    output = Path() / "burner_flame.yaml"
output.unlink(missing_ok=True)

f.save(output, name="mix", description="solution with mixture-averaged transport")

f.transport_model = 'multicomponent'
f.set_refine_criteria(ratio=2.0, slope=0.02, curve=0.02)
f.solve(loglevel)  # don't use 'auto' on subsequent solves



f.save('burner_flame.csv', basis="mole", overwrite=True)

fig, ax = plt.subplots()
ax.plot(f.flame.grid, f.T, 'o')
#ax2 = ax.twinx()
#ax2.plot(f.flame.grid, f.velocity, 'ro--')


#%%