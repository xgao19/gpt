#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

# load configuration
# rng = g.random("test")
# U = g.qcd.gauge.random(g.grid([16, 16, 16, 4], g.double), rng)
U = g.load("/home/scior/converter/l6464f21b7130m00119m0322a.1056.nersc")
# V = rng.element(g.lattice(U[0]))
# U_transformed = g.qcd.gauge.transformed(U, V)

# # reference plaquette
# P = g.qcd.gauge.plaquette(U)

# # test rectangle calculation using parallel transport and copy_plan
# R_1x1, R_2x1 = g.qcd.gauge.rectangle(U, [(1, 1), (2, 1)])
# eps = abs(P - R_1x1)
# g.message(f"Plaquette {P} versus 1x1 rectangle {R_1x1}: {eps}")
# assert eps < 1e-13

# # Test gauge invariance of plaquette
# P_transformed = g.qcd.gauge.plaquette(U_transformed)
# eps = abs(P - P_transformed)
# g.message(f"Plaquette before {P} and after {P_transformed} gauge transformation: {eps}")
# assert eps < 1e-13

# # Test gauge invariance of R_2x1
# R_2x1_transformed = g.qcd.gauge.rectangle(U_transformed, 2, 1)
# eps = abs(R_2x1 - R_2x1_transformed)
# g.message(
#     f"R_2x1 before {R_2x1} and after {R_2x1_transformed} gauge transformation: {eps}"
# )
# assert eps < 1e-13

# # Test gauge covariance of staple
# rho = np.array(
#     [[0.0 if i == j else 0.1 for i in range(4)] for j in range(4)], dtype=np.float64
# )
# C = g.qcd.gauge.staple_sum(U, rho=rho)
# C_transformed = g.qcd.gauge.staple_sum(U_transformed, rho=rho)
# for mu in range(len(C)):
#     a = g.sum(g.det(U[mu])) / U[0].grid.gsites
#     # b = g.sum(g.det(C_u[mu])) / U[0].grid.gsites
#     g.message(f"avg. Determinant of projected C[{mu}] = {a}")#", of C_u[{mu}] = {b}")








# # Test HYP smearing
# U_hyp = U
# P_hyp = []
# for i in range(3):
U_hyp = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.1, 0.2, 0.3]))

#     for mu in range(len(U_hyp)):
#         I = g.identity(U_hyp[mu])
#         eps2 = g.norm2(U_hyp[mu] * g.adj(U_hyp[mu]) - I) / g.norm2(I)
#         g.message(f"Unitarity check of hyp-smeared links: mu = {mu}, eps2 = {eps2}")

#     P_hyp.append(g.qcd.gauge.plaquette(U_hyp))

# g.message(f"Hyp smeared plaquettes {P_hyp}")
# assert sorted(P_hyp) == P_hyp  # make sure plaquettes go towards one

U_prime, trafo = g.gauge_fix(U_hyp)

# # print(U)
# # print(U_prime)

# P_prime = g.qcd.gauge.plaquette(U_prime)

# eps = abs(P - P_prime)
# g.message(f"Plaquette before {P} and after {P_prime} gauge fixing: {eps}")

# k = g.sum(g.det(trafo)) / U[0].grid.gsites

# g.message(f"avg. Determinat of gauge trafo matrices; {k}")

# src = g.mspincolor(U[0].grid)
# g.create.point(src, [0, 8, 0, 0])

# # dst = g.mspincolor(U[0].grid)


# # # print(dst)
# coord = g.coordinates(U[0].grid)

# # #make the gauge fixed src
# gf_src = g.eval(trafo*src)

# test = g.complex(U[0].grid)

# test[:] = gf_src[:,:,:,:,1,1,1,1]
# print(gf_src[0,0,0,0,1,1,1,1][0,0,0,0,0])

# print(test[0,0,0,0])
# print(test[int(coord[0][0]),int(coord[0][1]),int(coord[0][2]),int(coord[0][3])])

# with open("pre.txt", "w") as f:
#     for x in coord:
#         # tmp = test[x[0,x[1],x]]
#         # print(f"{x[0]} \t {x[1]} \t {x[2]} \t {x[3]} \t {test[1,2,3,4]}\n")
#         f.write(f"{x[0]} \t {x[1]} \t {x[2]} \t {x[3]} \t {np.abs(gf_src[int(x[0]),int(x[1]),int(x[2]),int(x[3]),1,1,1,1][0,0,0,0,0])} \n")

# g.message("Output 1 complete")

# interm = g.create.smear.boosted_smearing(trafo, src, w=1.0, boost=[0.0,0.0,2.0])

# back = g.create.smear.boosted_smearing(trafo, interm, w=1.0, boost=[0.0,0.0,-2.0])

# # #do fft for dims 0,1,2
# # fft =g.eval(g.fft([0,1,2])*gf_src)

# # # multiply with shifted gaussian in mom. space. Parameters: destination, source, "width", boost 
# # g.apply_exp_p2(dst, fft, 2.0, [0.0,0.0,1.0])

# # back = g.eval(g.adj(g.fft([0,1,2]))*dst)

# with open("aft.txt", "w") as f:
#     for x in coord:
#         f.write(f"{x[0]} \t {x[1]} \t {x[2]} \t {x[3]} \t {np.abs(back[int(x[0]),int(x[1]),int(x[2]),int(x[3]),1,1,1,1][0,0,0,0,0])} \n")

# momenta_sq = g.complex(U[0].grid)
# gaussian = g.complex(U[0].grid)

# coord = g.coordinates(U[0].grid)

# unit = g.complex(U[0].grid)
# unit[:] = 1

# sqr = coord[:,0]*coord[:,0] + coord[:,1]*coord[:,1] + coord[:,2]*coord[:,2]

# print(gaussian)

# print(sqr)

# gaussian = np.exp(-0.5*sqr)


# test = gaussian*src



# # sqr[:] = x[0,:]

# for x in range(4):
#     for y in range(4):
#         for z in range(4):
#             for t in range(4):
#                 sqr[x,y,z,t] = x**2+y**2+z**2

# print(sqr)

# print(g.norm2(gaussian-sqr))

# dim = U[0].grid.fdimensions

# for x in coord:
#     a=int(x[0])
#     b=int(x[1])
#     c=int(x[2])
#     d=int(x[3])
#     momenta_sq[a,b,c,d]=4*np.pi*np.pi*((a/dim[0]**2)+(b/dim[1]**2)+(c/dim[2]**2))
# w=0.5
# # gaussian[:] = np.exp(-0.5*w**2*momenta_sq[:])

# g.message("Set done")

# print(gaussian)

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
