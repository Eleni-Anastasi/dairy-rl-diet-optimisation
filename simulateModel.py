from sys import path
path.append(r"libraries")

from libraries import simulationLibrary as lib
import numpy as np

T_P = 83
u_const = [9, 35.6, 18.2, 18.6, 11.2, 3.7, 3.7, 0, 0]   # baseline diet

# model parameters
[p, ops, b] = lib.DefineParameters()

# simulation parameters
ops['N'] = T_P
ops['ts'] = 1
ops['t'] = np.arange(0, ops['N'] * ops['ts'], ops['ts'])

# signals
[y, x, u, d] = lib.DefineSignals(ops, u_const)

# force mode 3 settings exactly like your RL env
b["s"] = 1186
b["q1"] = 1
b["q2"] = 0
b["q3"] = 2
b["c1"] = 0
b["c2"] = 0

mode = 3

for k in range(ops['N']):
    if mode == 3:
        x[:, k+1] = lib.f3(x[:, k], u[:, k], d[:, k], p)
        y[:, k]   = lib.g3(x[:, k], u[:, k], d[:, k], p, b)
        b['c1'] += 1
    else:
        raise ValueError("This baseline comparison is intended for mode 3 only.")

# ---- outputs from g3 ----
# g3: [y1,y2,y3, milk, methane, feed_margin, DVE_req, DVE_intake, OEB_intake, FM_weight]

milk = y[3, :]
methane = y[4, :]
feed_margin = y[5, :]
dve_req = y[6, :]
dve_intake = y[7, :]
oeb_intake = y[8, :]
fm_weight = y[9, :]

# concentrate percentage from fixed baseline diet
u_baseline = np.array(u_const, dtype=float)
conc_idxs = [3, 4, 5, 6, 7, 8]
conc_pct = u_baseline[conc_idxs].sum()

print("=== Baseline diet results ===")
print(f"Sum(feed margin) = €{feed_margin.sum():.2f}")
print(f"Average(feed margin) = €{feed_margin.mean():.2f}")
print(f"Sum(methane) = {methane.sum()/1000:.2f} kg")
print(f"Average(methane) = {methane.mean():.1f} g/day")
print(f"Sum(milk) = {milk.sum():.2f} kg")
print(f"Average(milk) = {milk.mean():.2f} kg/day")
print(f"Concentrate percentage = {conc_pct:.1f}%")
print(f"Average DVE requirement = {dve_req.mean():.2f}")
print(f"Average DVE intake = {dve_intake.mean():.2f}")
print(f"Average OEB intake = {oeb_intake.mean():.2f}")
print(f"Average fresh matter intake = {fm_weight.mean():.2f}")





'''
###############################################################################
## plot
###############################################################################

# plot disturbance and controls
fig, ax = plt.subplots(ops['nu']+ops['nd'])
fig.set_size_inches(12.5, 12.5)

ax[0].plot(ops['t'][0:kk]/24,u[0,0:kk], drawstyle='steps')
ax[0].grid(True)
ax[0].set_xlim(left=ops['t'][0]/24, right=ops['t'][-1]/24)
ax[0].set_ylabel(r'$u_1$ (-)', fontsize=12)
#ax[0].set_ylim(bottom=0.9*min(u[0,1:kk]), top=1.1*max[u[0,1:kk]))

ax[1].plot(ops['t'][0:kk]/24,u[1,0:kk], drawstyle='steps')
ax[1].grid(True)
ax[1].set_xlim(left=ops['t'][0]/24, right=ops['t'][-1]/24)
ax[1].set_ylabel(r'$u_2$ (-)', fontsize=12)
#ax[1].set_ylim(bottom=0.9*min(u[1,1:kk]), top=1.1*max[u[1,1:kk]))

ax[2].plot(ops['t'][0:kk]/24,u[2,0:kk], drawstyle='steps')
ax[2].grid(True)
ax[2].set_xlim(left=ops['t'][0]/24, right=ops['t'][-1]/24)
ax[2].set_ylabel(r'$u_3$ (-)', fontsize=12)
#ax[2].set_ylim(bottom=0.9*min(u[2,1:kk]), top=1.1*max[u[2,1:kk]))

ax[3].plot(ops['t'][0:kk]/24,d[0,0:kk], drawstyle='steps')
ax[3].grid(True)
ax[3].set_xlim(left=ops['t'][0]/24, right=ops['t'][-1]/24)
ax[3].set_ylabel(r'$d_1$ (degC)', fontsize=12)
#ax[3].set_ylim(bottom=0.9*min(d[0,1:kk]), top=1.1*max[d[0,1:kk]))

ax[4].plot(ops['t'][0:kk]/24,d[1,0:kk], drawstyle='steps')
ax[4].grid(True)
ax[4].set_xlim(left=ops['t'][0]/24, right=ops['t'][-1]/24)
ax[4].set_ylabel(r'$d_2$ (kW)', fontsize=12)
#ax[4].set_ylim(bottom=0.9*min(d[1,1:kk]), top=1.1*max[d[1,1:kk]))

ax[5].plot(ops['t'][0:kk]/24,d[2,0:kk], drawstyle='steps')
ax[5].grid(True)
ax[5].set_xlim(left=ops['t'][0]/24, right=ops['t'][-1]/24)
ax[5].set_ylabel(r'$d_3$ (€/kWh)', fontsize=12)
#ax[5].set_ylim(bottom=0.9*min(d[2,1:kk]), top=1.1*max[d[2,1:kk]))

ax[6].plot(ops['t'][0:kk]/24,d[3,0:kk], drawstyle='steps')
ax[6].grid(True)
ax[6].set_xlim(left=ops['t'][0]/24, right=ops['t'][-1]/24)
ax[6].set_ylabel(r'$d_4$ (€/kWh)', fontsize=12)
#ax[6].set_ylim(bottom=0.9*min(d[3,1:kk]), top=1.1*max[d[3,1:kk]))

ax[6].set_xlabel(r'Time (days)',fontsize=12)
'''