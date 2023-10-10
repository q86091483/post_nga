#%%
import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = "./tmp"
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import cantera as ct
#%%
# 0. Initialize parameter ---------------------------------------
Lx1D    = 4.06E-4 * 14.3 * 2.45 * 2.718   #0.0222538
P_atm   = 1.
T_in    = 990   #950
eqrt    = 0.35  #0.4
P_in    = P_atm * ct.one_atm
mech    = "h2o2.yaml"

Lx_out = Lx1D
nx_out = 1182
n1 = 1900; n2 = 100
a1 = np.linspace(0, Lx_out*0.4, n1)[0:-1]; a2 = np.linspace(Lx_out*0.4,Lx_out,n2)
init_grid = np.linspace(0, Lx_out, n1) #np.concatenate((a1, a2))

mtot    = eqrt*2.0 + (1.0 + 3.76)
X_in        = {}
X_in["H2"]  = eqrt*2.0 / mtot
X_in["O2"]  = 1.0 / mtot
X_in["N2"]  = 3.76 / mtot
gas = ct.Solution(mech)
gas.TPX = T_in, P_in, X_in
gas0D = ct.Solution(mech)
gas0D.TPX = T_in, P_in, X_in
print("Unburned gas: ", gas()) 
print("Viscosity", gas.viscosity)

# 1. Homogeneous ignition delay time ----------------------------
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
    indx = np.where(states.T>states.T[0]+400.)[0][0]
    ax.plot(states.t[0:int(2.5*indx)], states.T[0:int(2.5*indx)], 'r', linewidth=3.0)
    ax.set_title("0D")
    ax.set_ylabel("T[K]", fontsize=24)
    ax.set_xlabel("t[s]", fontsize=24)

if np.amax(states.T > (states.T[0]+400)):
    indx = np.where(states.T>states.T[0]+400.)[0][0]
    print("0D ignition delay time: ", states.t[indx])

# 2. One-dimensional premixed flame -----------------------------
loglevel=1
f = ct.FreeFlame(gas, grid=init_grid)
f.set_max_grid_points(domain=f.domains[1], npmax=10000)
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
f.transport_model = 'mixture-averaged'
f.solve(loglevel, auto=True)
f.transport_model = 'mixture-averaged' #'multicomponent'
f.set_refine_criteria(ratio=2.0, slope=0.005, curve=0.005)
f.solve(loglevel)  # don't use 'auto' on subsequent solves
f.save('burner_flame.csv', basis="mole", overwrite=True)

# Write results
res = {}
res["x"]                = f.flame.grid
res["temperature"]      = f.T
res["pressure"]         = f.P
res["density"]          = f.density
res["enthalpy_mass"]    = f.enthalpy_mass
res["velocity"]         = f.velocity
spn_out = ["N2", "H", "O2", "O", "OH", "H2", "H2O", "HO2", "H2O2"]
for spn in spn_out:
    isp = gas.species_index(spn)
    res[spn] = f.Y[isp, :]

fig, ax = plt.subplots()
ax.plot(res["x"], res["temperature"], 'ro-', linewidth=3.0, color="r", label="U")
ax.set_xlabel("x [cm]", fontsize=24)
ax.set_ylabel("T [K]", fontsize=24)
ax2 = ax.twinx()
ax2.plot(res["x"], res["velocity"], 'b--', linewidth=3.0, label="U")
ax2.set_ylim([0, np.amax(res["velocity"])])
ax2.legend()
ax2.set_ylabel("U [m/s]", fontsize=24)

res = pd.DataFrame.from_dict(res)
if np.amax(states.T > (states.T[0]+400)):
    indx = np.where(states.T>states.T[0]+400.)[0][0]
    indx_1D = np.where(res.temperature > res.temperature[0]+400.)[0][0]
    ax.plot([states.t[indx]*f.velocity[0], states.t[indx]*f.velocity[0]], [T_in, T_in+800], "g")
    ax.plot([res.x[indx_1D], res.x[indx_1D]], [T_in, T_in+800], "r--")
    ax.set_title("1D", fontsize=24)

strt = "Solu_H2_" + str(int(T_in)) + "K_" + str(int(P_atm)) + "atm_phi" + str(eqrt) 
res.to_csv(strt+".csv")

iflame = np.argmax(res.temperature > (res.temperature[0]+400.))
xflame = res.x[iflame]
gradT_r = (res.temperature[iflame+1] - res.temperature[iflame]) / (res.x[iflame+1] - res.x[iflame])
gradT_l = (res.temperature[iflame] - res.temperature[iflame-1]) / (res.x[iflame] - res.x[iflame-1])
gradT = 0.5 * (gradT_r + gradT_l)
lf = (res.temperature.values[-1] - res.temperature[0]) / gradT
print("Domain size      : ", Lx1D)
print("U_in             : ", res.velocity[0])
print("Flame location   : ", xflame)
print("Laminar flame thickness [mm]: ", lf*1000)

#%%