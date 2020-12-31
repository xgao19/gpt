#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np
import gpt as g
from gpt.params import params_convention


# HYP smearing as in Chroma but with SU3 projection from Grid

@params_convention(alpha=None)
def hyp(U, params):
    nd = len(U)

    alpha = params["alpha"]
    assert alpha is not None
    assert alpha.shape == (3,)
    
    U_prime = []
    V1 = []
    
    #lvl 1 smearing
    #ii = -1

    for mu in range(nd):
        for nu in range(nd):
            if mu==nu:
                continue
            #ii+=1
            u_tmp = g.qcd.gauge.smear.staple(U, mu, nu)
            # V_ii= g((1.0-rho)*U[mu]+rho/2.0*u_tmp)
            v1_tmp = g.expr_eval((1.0-alpha[2])*U[mu]+alpha[2]/2.0*u_tmp) 
            g.projectSU3(v1_tmp)
            V1.append(v1_tmp)
    
    # ii=-1

    #lvl 2 smearing
    V2 = []
    for mu in range(nd):
        for nu in range(nd):
            # ii+=1
            first = True
            for eta in range(nd):
                if(eta == mu or eta==nu):
                    continue
                for jj in range(nd):
                    if(jj != mu and jj != nu and jj != eta):
                        sigma = jj
                jj = (nd-1)*mu + sigma
                if(sigma > mu):
                    jj-=1
                kk = (nd-1)*eta + sigma
                if(sigma > eta):
                    kk-=1

                #forward staple:
                #u_tmp += #u_lv1[kk] * shift(u_lv1[jj],FORWARD,rho) * adj(shift(u_lv1[kk],FORWARD,mu));
                if(first):
                    u_tmp = V1[kk]*g.cshift(V1[jj], eta, 1)*g.adj(g.cshift(V1[kk], mu,1))
                    first = False
                else:
                    u_tmp = u_tmp + V1[kk]*g.cshift(V1[jj], eta, 1)*g.adj(g.cshift(V1[kk], mu,1))
                 
                
                #backward staple:
                # u_tmp(x) += u_lv1_dag(x-rho,kk)*u_lv1(x-rho,jj)*u_lv1(x-rho+mu,kk)
                tmp= g.cshift(V1[kk], mu ,1)
                tmp2 = g.adj(V1[kk])* V1[jj] * tmp
                u_tmp = u_tmp + g.cshift(tmp2, eta , -1)
            
            tmp_v2 = g.expr_eval((1.0-alpha[1])*U[mu] + alpha[1]/4.0 * u_tmp)
            g.projectSU3(tmp_v2)
            V2.append(tmp_v2)
    
    #construct hyp links:
    V3 = []
    for mu in range(nd):
        first = True
        for nu in range(nd):
            if( mu == nu):
                continue
            jj = (nd-1)*mu + nu
            if( nu > mu ):
                jj-=1
            kk = (nd-1)*nu + mu
            if( mu > nu ):
                kk -= 1
            #forward staple
            #u_tmp(x) += u_lv2(x,kk)*u_lv2(x+nu,jj)*u_lv2_dag(x+mu,kk)
            if(first):
                u_tmp = V2[kk]*g.cshift(V2[jj], nu, 1)*g.adj(g.cshift(V2[kk], mu, 1))
                first = False
            else:
                u_tmp = u_tmp + V2[kk]*g.cshift(V2[jj], nu, 1)*g.adj(g.cshift(V2[kk], mu, 1))
            
            #backward staple
            #u_tmp(x) += u_lv2_dag(x-nu,kk)*u_lv2(x-nu,jj)*u_lv2(x-nu+mu,kk)
            tmp = g.cshift(V2[kk], mu, 1)
            tmp2 = g.adj(V2[kk]) * V2[jj] * tmp
            u_tmp = u_tmp + g.cshift(tmp2, nu, -1)

        tmp_v3 = g.expr_eval((1-alpha[0])*U[mu] + alpha[0]/6.0*u_tmp)
        g.projectSU3(tmp_v3)

        U_prime.append(tmp_v3)

    return U_prime
                


    