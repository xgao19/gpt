#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                        Mattia Bruno
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
import gpt
import numpy


def copy(dst, src):
    if isinstance(src,list):
        for i in range(len(src)):
            dst[i] @= src[i]
    else:
        dst @= src


def boltzman_factor(h1, h0):
    return numpy.exp(-h1+h0)

# Given the probability density P(f(x)), the metropolis algorithm:
# - computes the initial probability P0 = P(f(x0))
# - makes a proposal for the new fields with w[x1 <- x0]
# - computes the final probability P1 = P(f(x1))
# - performs the accept/reject step, ie accepts with probability = min{1, P(f(x1))/P(f(x0))}
# The class below accepts the following arguments:
# - rng: random number generator
# - w: proposal, a callable function
# - f: the function, argument of the probability density
# - fields: a list of fields
# - prob_ratio: the method to compute the ratio P(f(x1))/P(f(x0)), which implicitly defines P,
#               e.g. boltzman_factor: P(f(x1))/P(f(x0)) = exp(-f(x1) + f(x0))
class metropolis:
    def __init__(self, rng, w, f, fields, prob_ratio = boltzman_factor):
        self.rng = rng
        self.proposal = w
        self.f = f
        self.fields = gpt.core.util.to_list(fields)
        self.prob_ratio = prob_ratio
        
        self.grid = None
        self.copies = []
        for f in self.fields:
            if isinstance(f,list):
                tmp = []
                for ff in f:
                    tmp.append(gpt.lattice(ff))
                    if self.grid is None:
                        self.grid = ff.grid
            else:
                tmp = gpt.lattice(f)
                if self.grid is None:
                    self.grid = f.grid
            self.copies.append(tmp)
            
    def start(self):
        for i in range(len(self.fields)):
            copy(self.copies[i], self.fields[i])
        return self.f(*self.fields)
    
    def restore(self):
        for i in range(len(self.fields)):
            copy(self.fields[i], self.copies[i])
        
    def __call__(self, *vargs):
        f0 = self.start()
        self.proposal(*vargs)
        f1 = self.f(*self.fields)
        
        # decision taken on master node, but for completeness all nodes throw one random number
        rr = self.rng.uniform_real(None, {"min": 0, "max": 1})
        accept = 0
        if gpt.rank() == 0:
            if self.prob_ratio(f1, f0) >= rr:
                accept = 1
        gpt.barrier()
        accept = self.grid.globalsum(accept)

        if accept == 0:
            self.restore()

        return [accept, f1-f0]