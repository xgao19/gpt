#!/usr/bin/env python3
import gpt as g
import numpy as np
import sys

rng = g.random("test")
U = g.qcd.gauge.random(g.grid([16, 16, 16, 16], g.double), rng)
#U = g.load("/home/scior/converter/l6464f21b7130m00119m0322a.1056.nersc")

U_hyp = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3]))

plaq_hyp = g.qcd.gauge.plaquette(U_hyp)

g.message("###########MEMORY REPORT AFTER HYP")
g.mem_report()

g.message(f"HYP smeared Plaquette = {plaq_hyp}")

grid = U_hyp[0].grid

p = {
    "kappa" : 0.12623,
    "csw_r": 1.0372,
    "csw_t": 1.0372,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1, 1, 1, -1],
}

g.message("###########MEMORY REPORT BEFORE DIRAC OPERATOR")
g.mem_report()
w = g.qcd.fermion.wilson_clover(U_hyp, p)
g.message("###########MEMORY REPORT AFTER DIRAC OPERATOR")
g.mem_report()
# create point source
src = g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 0])

# even-odd preconditioned matrix
# build solver using g5m and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-6, "maxiter": 1000})

slv = w.propagator(inv.preconditioned(pc.g5m_ne(), cg))

# propagator
dst = g.mspincolor(grid)
dst @= slv * src

#get # of iterations
# dc_iter = len(cg.history) 

# pion
corr_pion = g.slice(g.trace(g.adj(dst) * dst), 3)  #is this already projected to p=0? I think so, as slice contains a sum
print("Pion two point:")
print(corr_pion)

corr_a4pi = g.slice(g.trace(g.adj(dst) * g.gamma["T"] * dst),3)

print("A4-pi two point:")
print(corr_a4pi)
