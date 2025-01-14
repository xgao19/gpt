#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

# This preconditioner performs a block decomposition of the local
# lattice; even and odd blocks are defined from their block id and
# and they are respectively packed together into two major blocks.
# The corresponding block operator is defined on the two clusters of
# blocks: links are properly set to zero to ensure Dirichelet Boundary
# Conditions, such that the operators are automatically defined on
# the blocks.
# The SAP class takes care of performing the block decomposition of
# the gauge field and of defining the corresponding block operators.
# It also contains lists of the coordinates corresponding to the even
# and odd blocks, such that fields can be easily assigned to blocks
# and viceversa.

# This inverter approximates the solution of the Dirac equation
# using the Schwartz Alternating Procedure as described here
#
# M. Luescher, "Solution of the Dirac equation in lattice QCD using a
#               domain decomposition method"
# https://arxiv.org/abs/hep-lat/0310048
#
# It is based on the sap preconditioner class that contains the
# basic functionalities relevant here.
#
# The code below is a first working version; improvements for
# better performances could be achieved by substituting the
# application of op.M with the simpler hopping from even/odd
# blocks.

#
#      ( EE EO )
#  M = ( OE OO )
#
#      ( EE^-1            0     )
#  K = ( -OO^-1 OE EE-1   OO^-1 )
#
#       ( 1 - EO OO^-1 OE EE-1    EO OO^-1  )
#  MK = ( 0                       1         )
#
#  Then K \sum_{n=0}^N (1 - MK)^n -> K (MK)^-1 = M^-1
#
#
#  eps = 1 - MK
#
#        ( EO OO^-1 OE EE-1       -EO OO^-1  )
#      = ( 0                       0         )
#
# This structure maps very well to our defect-correcting inverter:
#
#   outer_mat     =  (1 + defect) inner_mat
#
#   outer_mat^-1  = inner_mat^-1 \sum_{n=0}^N (- defect)^n
#
# with
#
#   inner_mat^-1 = K  and outer_mat = M
#
#   -defect = 1 - outer_mat inner_mat^{-1} = 1 - M K
#
import gpt
from gpt.params import params_convention


class sap_cycle:
    @params_convention(block_size=None)
    def __init__(self, blk_solver, params):
        self.bs = params["block_size"]
        self.blk_solver = blk_solver
        assert self.bs is not None

    def __call__(self, op):

        bop = gpt.qcd.fermion.domain.even_odd_blocks(op, block_size=self.bs)
        solver = [self.blk_solver(f) for f in bop.fermions]

        def inv(dst, src):
            dst[:] = 0
            eta = gpt.copy(src)
            ws = [gpt.copy(src) for _ in range(2)]

            for eo in range(2):
                ws[0][:] = 0

                src_blk = bop.F_domains[eo].lattice(op.otype)
                dst_blk = bop.F_domains[eo].lattice(op.otype)

                bop.F_domains[eo].project(src_blk, eta)

                dst_blk[:] = 0  # for now
                solver[eo](dst_blk, src_blk)

                bop.F_domains[eo].promote(ws[0], dst_blk)

                if eo == 0:
                    op(ws[1], ws[0])

                eta -= ws[1]
                dst += ws[0]

                gpt.message(
                    f"SAP cycle; |rho|^2 = {gpt.norm2(eta):g}; |dst|^2 = {gpt.norm2(dst):g}"
                )

        return gpt.matrix_operator(
            mat=inv,
            inv_mat=op,
            adj_inv_mat=op.adj(),
            adj_mat=None,
            vector_space=op.vector_space,
            accept_guess=(True, False),
        )


# sap_cycle applies K of
#
#      ( EE^-1            0     )
#  K = ( -OO^-1 OE EE-1   OO^-1 )
#
# ws0 = EE^-1 src_e
# ws1 = OE EE^-1 src_e

# eta = src - OE EE^-1 src_e
# dst = OE EE^-1 src_e
#
# ws0 = OO^-1 (src_o - OE EE^-1 src_e)
# ws1 = EO OO^-1 (src_o - OE EE^-1 src_e)
#
