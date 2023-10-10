#%%
"""
A burner-stabilized lean premixed hydrogen-oxygen flame at low pressure.

Requires: cantera >= 3.0
Keywords: combustion, 1D flow, premixed flame, saving output,
          multicomponent transport
"""

from pathlib import Path
import cantera as ct

P_atm   = 1.0
T_in    = 373
eqrt    = 0.4
P_in    = P_atm * ct.one_atm
X_in    = {}
mtot    = eqrt*2.0 + (1.0 + 3.76)
X_in["H2"] = eqrt*2.0 / mtot
X_in["O2"] = 1.0 / mtot
X_in["N2"] = 3.76 / mtot

gas = ct.Solution('h2o2.yaml')
gas.TPX=T_in, P_in, X_in
gas.equilibrate("HP")
print(gas())

#%%
f = ct.BurnerFlame(gas, width=width)
f.burner.mdot = mdot
f.set_refine_criteria(ratio=3.0, slope=0.05, curve=0.1)
f.show()

f.transport_model = 'mixture-averaged'
f.solve(loglevel, auto=True)

if "native" in ct.hdf_support():
    output = Path() / "burner_flame.h5"
else:
    output = Path() / "burner_flame.yaml"
output.unlink(missing_ok=True)

f.save(output, name="mix", description="solution with mixture-averaged transport")

f.transport_model = 'multicomponent'
f.solve(loglevel)  # don't use 'auto' on subsequent solves
f.show()
f.save(output, name="multi", description="solution with multicomponent transport")

f.save('burner_flame.csv', basis="mole", overwrite=True)
fig, ax = plt.subplots()
ax.plot(f.flame.grid, f.T)
#%%