#%%
import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = "./tmp"
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import cantera as ct

spn_out = ["H2", "O2", "H2O", "H", "O", "OH", "HO2", "H2O2", "N2"]
#%%
# 0. Initialize parameter ---------------------------------------
Re_j   = 5000
njet   = 2
P      = 10.0 * ct.one_atm
D_j    = 5E-4; A_j = 0.25 * np.pi * D_j * D_j
intv   = 2.0; 
Lx     = 12 * D_j;          
Ly     = (2*intv+2) * D_j; 
Lz     = 5.6 * D_j; 
A_c    = Ly * Lz
T_j    = 300.;
T_c    = 800.;
X_j    = {}; X_j["H2"] = 1.0; X_j["N2"] = 1 - X_j["H2"] 
X_c    = {}; X_c["O2"] = 0.21; X_c["N2"] = 0.79
mech     = "h2o2.yaml"; 
freq   = 1000

gas_j = ct.Solution(mech)
gas_c = ct.Solution(mech)
id_H2 = gas_j.species_index("H2")
id_O2 = gas_j.species_index("O2")

gas_j.TPX = T_j, P, X_j
gas_c.TPX = T_c, P, X_c

rho_j = gas_j.density
rho_c = gas_c.density

nu_j = gas_j.viscosity
nu_c = gas_c.viscosity

U_j = Re_j * gas_j.viscosity /D_j
U_c = 45

m_j = njet * A_j * U_j * rho_j
m_c = A_c * U_c * rho_c
J = (rho_j * U_j**2) / (rho_c * U_c**2)

gas_mix = ct.Solution(mech)
h_mix   = m_j*gas_j.enthalpy_mass + m_c*gas_c.enthalpy_mass
Y_mix   = (m_j/(m_j+m_c))*gas_j.Y + (m_c/(m_j+m_c))*gas_c.Y
gas_mix.HPY = h_mix, P, Y_mix
equiv = gas_mix.X[id_H2] / (2*gas_mix.X[id_O2])

# Calculate zst
Yst = 4. / (4 + 32)
YO_O2 = gas_c.Y[id_O2]
YF_O2 = gas_j.Y[id_O2]
YF_H2 = gas_j.Y[id_H2]
zst = Yst*YO_O2 / (YF_H2 - Yst*(YF_O2-YO_O2))
print(zst)

print("Density:     {:5.2E}, {:5.2E}".format(rho_j, rho_c))
print("Viscosity:   {:5.2E}, {:5.2E}".format(nu_j, nu_c))
print("Velocity:    {:5.2E}, {:5.2E}".format(U_j, U_c))
print("Sound speed: {:5.2E}, {:5.2E}".format(gas_j.sound_speed, gas_c.sound_speed))
print("Mach number: {:5.2E}, {:5.2E}".format(U_j/gas_j.sound_speed, U_c/gas_c.sound_speed))
print("mdot:        {:5.2E}, {:5.2E}".format(m_j, m_c))
print("Rv:          {:5.2E}".format(U_j/U_c))
print("J:           {:5.2E}".format(J))
print("Gloabl eqv:  {:5.2E}".format(equiv))

print("Domain x:    {:5.2E} - {:5.2E}".format(0-1.5*D_j, Lx-1.5*D_j))
print("Domain y:    {:5.2E} - {:5.2E}".format(-Ly/2., Ly/2.))
print("Domain z:    {:5.2E} - {:5.2E}".format(0.0, Lz))
print("Jet y coord: {:5.2E} ".format(0.5*(intv+1)*D_j))
print("Flow through time: {:5.2E}".format(Lx/U_c))
print("Oscillation:       {:5.2E}".format(1/freq))

#%%
nzbin = 100; zmin = 0.0; zmax = 1.0
zbins = np.linspace(zmin, zmax, nzbin)
gas_s = ct.Solution(mech)
res0D = []
ncol = 4 + len(spn_out)
nrow = nzbin
res_pmf = np.zeros((nrow, ncol)) 
for iz in range(0, nzbin):
  zs = zbins[iz]
  Ys = zs*gas_j.Y + (1-zs)*gas_c.Y
  Hs = zs*gas_j.enthalpy_mass + (1-zs)*gas_c.enthalpy_mass
  gas_s.HPY = Hs, P, Ys
  gas_s.equilibrate("HP")
  res0D.append(gas_s.T)

  res_pmf[iz, 0] = zbins[iz]
  res_pmf[iz, 1] = gas_s.T
  res_pmf[iz, 2] = 0.0
  res_pmf[iz, 3] = gas_s.density
  for isp, spn in enumerate(spn_out):
    igas = gas_s.species_index(spn)
    res_pmf[iz, 4+isp] = gas_s.Y[igas]

s = 'VARIABLES = "X" "temp" "u" "rho"'
for isp, spn in enumerate(spn_out):
        s = s + ' "' + spn + '"'
s = s + "\n"
s = s + "ZONE I=" + str(nrow) + " FORMAT=POINT SPECFORMAT=MASS"
np.savetxt("./initeq.dat", res_pmf, fmt='%.18e', header=s, comments="", delimiter="     ")

Tz = np.array(res0D)
#Tz[6:-4] = Tz[6]
fig, ax4 = plt.subplots()
ax4.plot(zbins, res0D)
ax4.set_ylabel("Tad")

#%%
Nx     = 120
Ny     = 40
Nz     = 56
jet_xs = [0.0, 0.0]
jet_ys = [-1E-3, 1E3]
xmin = -1.5*D_j; xmax = xmin + Lx
ymin = -Ly / 2.; ymax = ymin + Ly
zmin = 0.;       zmax = zmin + Lz
H = Lz
x_ = np.linspace(xmin, xmax, Nx)
y_ = np.linspace(ymin, ymax, Ny)
z_ = np.linspace(zmin, zmax, Nz)
xx, yy, zz = np.meshgrid(x_, y_, z_, indexing='ij')
mf = np.zeros((Nx,Ny,Nz))
temp = np.zeros((Nx,Ny,Nz)) + T_c

ix0 = np.argmax(x_>jet_xs[0])
iy = np.argmax(y_>jet_ys[0])
phi2D = mf[:,iy,:]
figunit_x = 10
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(2*figunit_x*(Lz/Lx),figunit_x))
ax = axs[0]
ax2 = axs[1]
# Jet centerplane trajectory
xc = x_[(ix0-1):]; xc[0] = jet_xs[0]
b = 0.0002 * np.power(xc/H, 0.1)
zc = H * 0.15 * np.power(xc/H, 0.25) * np.exp(-b)
xw = np.argmax(x_>jet_xs[0]-D_j/0.25)
wp = H * 0.18 * np.power(xc/H, 0.25)
wn = H * 0.12 * np.power(xc/H, 0.25)

ax2.plot(xc, zc)
ax2.plot(xc, wp, "g--")
ax2.plot(xc, wn, "g-.")

theta_mix = m_j / (m_j + m_c)
theta_c1 = 1 - 0.2 * np.power(xc/H, 2)
theta_c2 = theta_mix + (1-theta_mix) * np.power(0.6 / (xc/H), 0.5)
theta_c = theta_c1
indx = (xc/H > 1.0); 
theta_c[indx] = theta_c2[indx]

cn = 0.2 * np.power(xc/H, -0.6) * np.exp(4.0 * np.power(xc/H, 2))
rcn = 1 - np.exp(-cn)

cp = 0.5 * np.power(xc/H, 4) 
rcp = 1 - np.exp(-cp)

mf_j = 1.0
mf_c = 0.0
iy = 0
theta_norm = np.zeros_like(mf)
for ix in range(0, Nx):
  for iz in range(0, Nz):
    ic = ix - ix0 +1
    if ix < ix0:
      theta_norm[ix,iy,iz] = 0.0
      continue
    zdis = z_[iz] - zc[ic] 
    if zdis > 0:
      bot = -np.log(2)*zdis**2
      bot = bot / (wp[ic]+1E-30)**2
      theta_norm[ix,iy,iz] = np.exp(bot)
      theta_p = theta_c[ic] * rcp[ic]
      mf[ix,iy,iz] = theta_p + theta_norm[ix,iy,iz] * (theta_c[ic]-theta_p)
    else:
      bot = -np.log(2)*zdis**2
      bot = bot / (wn[ic]+1E-30)**2
      theta_norm[ix,iy,iz] = np.exp(bot)
      theta_n = theta_c[ic] * rcn[ic]
      mf[ix,iy,iz] = theta_n + theta_norm[ix,iy,iz] * (theta_c[ic]-theta_n)
    mf[ix,iy,iz] = mf[ix,iy,iz] * 0.1
    nzbin = len(res0D)
    dZ = 1.0 / nzbin
    ibin = int(mf[ix,iy,iz]/dZ)
    ibin = np.amax((0,ibin))
    ibin = np.amin((nzbin-2,ibin))
    z0 = ibin*dZ; r0 = mf[ix,iy,iz] - z0 
    temp[ix,iy,iz] = res0D[ibin]*(1-r0) + res0D[ibin+1]*r0
for iy in range(1, Ny):
  mf[:,iy,:] = mf[:,0,:]
  temp[:,iy,:] = temp[:,0,:]

phi2D = temp[:,Ny-1,:]
vmin=0.; vmax=0.25
vmin=400; vmax=2500
#im = ax.imshow(phi2D.transpose(), extent=[xmin, xmax, zmin, zmax], origin="lower",
#         vmin = vmin, vmax = vmax, 
#          cmap=cm.viridis) 
#ctrz = ax.contourf(mf[:,5,:].transpose(), extent=[xmin, xmax, zmin, zmax], origin="lower",
#          vmin=vmin, vmax=vmax, levels=np.linspace(vmin,vmax,20),
#          cmap=cm.viridis) 
ctr = ax.contourf(phi2D.transpose(), extent=[xmin, xmax, zmin, zmax], origin="lower",
          vmin=vmin, vmax=vmax, levels=np.linspace(vmin,vmax,20),
          cmap=cm.viridis) 
#ax.contour(ctrz, levels=[zst], colors='r')
ax3 = ax2.twinx()
ax3.plot(xc, theta_c)
ax3.plot(xc, rcn, color="m", linestyle="--")
ax3.plot(xc, rcp, color="m", linestyle="-.")
ax3.set_xlim([xmin,xmax])
ax3.set_ylim([0, 1])
#ax.set_ylim([zmin,zmax])
#%%
# Write to pmf
#%%