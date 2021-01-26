#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

# load configuration
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.double), rng)
V = rng.element(g.lattice(U[0]))
U_transformed = g.qcd.gauge.transformed(U, V)

# Test gauge invariance of plaquette
P = g.qcd.gauge.plaquette(U)
P_transformed = g.qcd.gauge.plaquette(U_transformed)
eps = abs(P - P_transformed)
g.message(f"Plaquette before {P} and after {P_transformed} gauge transformation: {eps}")
assert eps < 1e-13

# Test gauge covariance of staple
rho = np.array(
    [[0.0 if i == j else 0.1 for i in range(4)] for j in range(4)], dtype=np.float64
)
C = g.qcd.gauge.smear.staple_sum(U, rho=rho)
# C_transformed = g.qcd.gauge.smear.staple_sum(U_transformed, rho=rho)
# for mu in range(len(C)):
#     q = g.sum(g.trace(C[mu] * g.adj(U[mu]))) / U[0].grid.gsites
#     q_transformed = (
#         g.sum(g.trace(C_transformed[mu] * g.adj(U_transformed[mu]))) / U[0].grid.gsites
#     )

#     eps = abs(q - q_transformed)
#     g.message(
#         f"Staple q[{mu}] before {q} and after {q_transformed} gauge transformation: {eps}"
#     )
#     assert eps < 1e-14
for mu in range(len(U)):
    q = g.sum(g.det(U[mu])) / U[0].grid.gsites
    r = g.sum(g.det(C[mu])) / U[0].grid.gsites
    g.message(f"avg. Determinant of U[{mu}] = {q}, of C[{mu}] = {r}")

# C_u = []
for mu in range(len(C)):
    # g.projectSU3(C[mu])
    g.projectSU3(U[mu],g.eval(g.adj(C[mu])))

for mu in range(len(C)):
    a = g.sum(g.det(U[mu])) / U[0].grid.gsites
    # b = g.sum(g.det(C_u[mu])) / U[0].grid.gsites
    g.message(f"avg. Determinant of projected C[{mu}] = {a}")#", of C_u[{mu}] = {b}")








# # Test HYP smearing
# U_hyp = U
# P_hyp = []
# for i in range(3):
#     U_hyp = g.qcd.gauge.smear.hyp(U_hyp, alpha = np.array([0.1, 0.2, 0.3]))

#     for mu in range(len(U_hyp)):
#         I = g.identity(U_hyp[mu])
#         eps2 = g.norm2(U_hyp[mu] * g.adj(U_hyp[mu]) - I) / g.norm2(I)
#         g.message(f"Unitarity check of hyp-smeared links: mu = {mu}, eps2 = {eps2}")

#     P_hyp.append(g.qcd.gauge.plaquette(U_hyp))

# g.message(f"Hyp smeared plaquettes {P_hyp}")
# assert sorted(P_hyp) == P_hyp  # make sure plaquettes go towards one

# U_prime, trafo = g.gauge_fix(U)

# # print(U)
# # print(U_prime)

# P_prime = g.qcd.gauge.plaquette(U_prime)

# eps = abs(P - P_prime)
# g.message(f"Plaquette before {P} and after {P_prime} gauge fixing: {eps}")

# k = g.sum(g.det(trafo)) / U[0].grid.gsites

# g.message(f"avg. Determinat of gauge trafo matrices; {k}")

# src = g.mspincolor(U[0].grid)
# g.create.point(src, [0, 0, 0, 0])

# #make the gauge fixed src
# gf_src = g.eval(trafo*src)

# fft =g.eval(g.fft([0,1,2])*gf_src)

# momenta_sq = g.complex(U[0].grid)
# gaussian = g.complex(U[0].grid)

# coord = g.coordinates(U[0].grid)



# # sqr[:] = x[0,:]

# # for x in range(8):
# #     for y in range(8):
# #         for z in range(8):
# #             for t in range(16):
# #                 sqr[x,y,z,t] = x**2+y**2+z**2

# dim = U[0].grid.fdimensions

# for x in coord:
#     a=int(x[0])
#     b=int(x[1])
#     c=int(x[2])
#     d=int(x[3])
#     momenta_sq[a,b,c,d]=4*np.pi*np.pi*((a/dim[0]**2)+(b/dim[1]**2)+(c/dim[2]**2))
# w=0.5
# gaussian[:] = np.exp(-0.5*w**2*momenta_sq[:])

# print("Set done, now multiplying")

# smear_p = g.eval(gaussian*fft)

# smear = g.eval(g.adj(g.fft([0,1,2]))*smear_p)

# smear_2=g.eval(g.adj(trafo)*smear)

# print(smear)

# # Test stout smearing
# U_stout = U
# P_stout = []
# for i in range(3):
#     U_stout = g.qcd.gauge.smear.stout(U_stout, rho=0.1)

#     for mu in range(len(U_stout)):
#         I = g.identity(U_stout[mu])
#         eps2 = g.norm2(U_stout[mu] * g.adj(U_stout[mu]) - I) / g.norm2(I)
#         g.message(f"Unitarity check of stout-smeared links: mu = {mu}, eps2 = {eps2}")

#     P_stout.append(g.qcd.gauge.plaquette(U_stout))

# g.message(f"Stout smeared plaquettes {P_stout}")
# assert sorted(P_stout) == P_stout  # make sure plaquettes go towards one

# # for given gauge configuration, cross-check against previous Grid code
# # this establishes the randomized check value used below
# # U = g.load("/hpcgpfs01/work/clehner/configs/24I_0p005/ckpoint_lat.IEEE64BIG.5000")
# # P = [g.qcd.gauge.plaquette(U),g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.15, orthogonal_dimension=3)),g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.1))]
# # P_comp = [0.588074,0.742136,0.820262]
# # for i in range(3):
# #    assert abs(P[i] - P_comp[i]) < 1e-5
# # g.message(f"Plaquette fingerprint {P} and reference {P_comp}")

# P = [
#     g.qcd.gauge.plaquette(U),
#     g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.15, orthogonal_dimension=3)),
#     g.qcd.gauge.plaquette(g.qcd.gauge.smear.stout(U, rho=0.1)),
# ]
# P_comp = [0.7986848674527128, 0.9132213221481771, 0.9739960794712376]
# g.message(f"Plaquette fingerprint {P} and reference {P_comp}")
# for i in range(3):
#     assert abs(P[i] - P_comp[i]) < 1e-12
