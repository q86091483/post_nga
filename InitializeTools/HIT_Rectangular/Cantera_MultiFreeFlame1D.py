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
Lx1D    = 0.017 #0.0222538
P_atm   = 1.
T_in    = 990   #950
eqrt    = 0.35  #0.4
P_in    = P_atm * ct.one_atm
mech    = "h2o2.yaml"

Lx_out = Lx1D
Ly_out = 0
Lz_out = 0
nx_out = 1182
ny_out = 1
nz_out = 1
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
    ax.plot(states.t, states.T, 'r', linewidth=3.0)
    ax.set_title("0D")
    ax.set_ylabel("T[K]", fontsize=24)
    ax.set_xlabel("t[s]", fontsize=24)

if np.amax(states.T > (states.T[0]+400)):
    indx = np.where(states.T>states.T[0]+400.)[0][0]
    print("0D ignition delay time: ", states.t[indx])

#%%
# 2. One-dimensional premixed flame -----------------------------
loglevel=1
f = ct.FreeFlame(gas, grid=init_grid)
f.set_max_grid_points(domain=f.domains[1], npmax=10000)
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
f.transport_model = 'mixture-averaged'
f.solve(loglevel, auto=True)
f.transport_model = 'multicomponent'
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
    ax.plot([states.t[indx]*f.velocity[0], states.t[indx]*f.velocity[0]], [T_in, T_in+800], "m:")
    ax.plot([res.x[indx_1D], res.x[indx_1D]], [T_in, T_in+800], "r-")

strt = "Solu_H2_" + str(int(T_in)) + "K_" + str(int(P_atm)) + "atm_phi" + str(eqrt) 
res.to_csv(strt+".csv")

iflame = np.argmax(res.temperature > (res.temperature[0]+400.))
xflame = res.x[iflame]
gradT_r = (res.temperature[iflame+1] - res.temperature[iflame]) / (res.x[iflame+1] - res.x[iflame])
gradT_l = (res.temperature[iflame] - res.temperature[iflame-1]) / (res.x[iflame] - res.x[iflame-1])
gradT = 0.5 * (gradT_r + gradT_l)
lf = (res.temperature.values[-1] - res.temperature[0]) / gradT
print("U_in", res.velocity[0])
print("Flame location: ", xflame)
print("Laminar flame thickness [mm]: ", lf*1000)

lxs = []; xfs = []; sls = []

sls.append(14.265808363040161) #0.0001
xfs.append(0.0008246445497630332)

sls.append(14.27355137629594) #0.00011
xfs.append(0.0008943654555028963)

sls.append(14.278576937495236) #0.00012
xfs.append(0.0009642969984202211)

sls.append(14.093438699492816) #0.00013
xfs.append(0.0005784623486045287)

sls.append(14.148555593062214) #0.00014
xfs.append(0.000614112690889942)

sls.append(14.187297039665218) #0.00015
xfs.append(0.0006492890995260664)

sls.append(14.214966355666668) #0.00016
xfs.append(0.0006841495523959979)

sls.append(14.234951015387594) #0.00017
xfs.append(0.0007197472353870458)

sls.append(14.250994305555485) #0.00018
xfs.append(0.0007545023696682464)

sls.append(14.260300045266911) #0.00019
xfs.append(0.0007894154818325434)

sls.append(14.26580836417763) #0.00020
xfs.append(0.0008246445497630332)

sls.append(14.270618556780784) #0.00021
xfs.append(0.0008592417061611374)

sls.append(14.27355138014158) #0.00022
xfs.append(0.0008943654555028963)

sls.append(14.276304125488808) #0.00023
xfs.append(0.0009289626119010005)

sls.append(14.278576937801304) #0.00024
xfs.append(0.0009642969984202211)

sls.append(14.280858760488169) #0.00025
xfs.append(0.0009992101105845183)

sls.append(14.282937125188084) #0.00026
xfs.append(0.0010343865192206426)

sls.append(14.28445055231633) #0.00027
xfs.append(0.0010691943127962085)

sls.append(14.28631792478047) #0.00028
xfs.append(0.0011043707214323328)

sls.append(14.288277089913517) #0.00029
xfs.append(0.0011392311743022643)

sls.append(14.290307571863154) #0.00030
xfs.append(0.001174565560821485)

sls.append(14.291661191635693) #0.00031
xfs.append(0.0012088204318062138)

sls.append(14.292997684881346) #0.00032
xfs.append(0.0012444444444444445)

sls.append(14.295194891536001) #0.00033
xfs.append(0.0012789889415481832)

sls.append(14.293897683894516) #0.00034
xfs.append(0.0013141653501843075)

sls.append(14.295083052604761) #0.00035
xfs.append(0.00134913112164297)

sls.append(14.296317499210954) #0.00036
xfs.append(0.0013838862559241705)

sls.append(14.297053933144305) #0.00037
xfs.append(0.0014194049499736705)

sls.append(14.297848096281713) #0.00038
xfs.append(0.00145376513954713)

sls.append(14.297254042195108) #0.00039
xfs.append(0.0014889415481832544)

sls.append(14.295536839082777) #0.00040
xfs.append(0.0015239599789362824)

sls.append(14.295681955356482) #0.00041
xfs.append(0.001558820431806214)

sls.append(14.295734154683052) #0.00042
xfs.append(0.001593522906793049)

sls.append(14.29501120862516) #0.00043
xfs.append(0.0016286334913112165)

sls.append(14.295175628379551) #0.00044
xfs.append(0.0016636124275934705)

sls.append(14.295190496546049) #0.00045
xfs.append(0.0016984597156398105)

sls.append(14.294088657754028) #0.00047
xfs.append(0.001768378093733544)

sls.append(14.293562276286051) #0.00049
xfs.append(0.0018384676145339653)

sls.append(14.292888132099913) #0.00051
xfs.append(0.0019081358609794631)

sls.append(14.29262986443634) #0.00054
xfs.append(0.002013270142180095)

sls.append(14.294717040423212) #0.0006
xfs.append(0.0022235387045813587)

sls.append(14.298664912849079) #0.00065
xfs.append(0.0023977093206951025)

sls.append(14.30032051618185) #0.00071
xfs.append(0.00260781990521327)

sls.append(14.303513825069508) #0.00077
xfs.append(0.0028180621379673516)

sls.append(14.305497515981033) #0.00085
xfs.append(0.003097419694576093)

sls.append(14.308367532348251) #0.0009
xfs.append(0.0032719194312796204)

sls.append(14.32615151128343) #0.01
xfs.append(0.0036216429699842028)

sls.append(14.381545825598774) #0.0011
xfs.append(0.0039722222222222225)

sls.append(14.534663430396948) #0.0012
xfs.append(0.00432306477093207)

sls.append(14.854000687325263) #0.0013
xfs.append(0.004674763033175355)

sls.append(15.35782091335294) #0.00014
xfs.append(0.005027909426013691)

sls.append(15.962690623996997) #0.00015
xfs.append(0.005381121642969984)

#sls.append(16.62893069685717) #0.00016
#xfs.append(0.005735650342285414)

#sls.append(17.335205981478644) #0.00017
#xfs.append(0.006090771458662454)

#sls.append(19.583425811836943) #0.02
#xfs.append(0.007155081621906267)

if len(sls) > 1:
    sls = np.array(sls)
    xfs = np.array(xfs)

    x = sls[1:]
    dx = sls[0:-1] - sls[1:]
    y = (xfs[0:-1] - xfs[1:]) / dx 
    fig, ax = plt.subplots()
    ax.plot(sls, xfs, "o")
    ax2 = ax.twinx()
    ax2.plot(x, y)
#%%