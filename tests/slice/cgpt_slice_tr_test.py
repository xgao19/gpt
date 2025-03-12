import gpt as g 
import os
import sys
import numpy as np
import math


U = g.qcd.gauge.unit(g.grid([16,16,16,16], g.double))
grid = U[0].grid
g.mem_report(details=False)

srcD = g.mspincolor(grid)
g.mem_report(details=False)

rng = g.random("test")

rng.cnormal(srcD)        
psrc = g.create.point(srcD, [0,0,0,0])
 
prop_bw = psrc
 
prop_f = psrc

g.message("slice_tr start")
result = g.slice_tr(prop_bw * prop_f,3)
g.message("slice_tr stop")

g.message("g.slice(g.trace(...)) start")
check = g.slice(g.trace(prop_bw * prop_f),3)
g.message("g.slice(g.trace(...)) stop")

diff = np.array(result) - np.array(check)

assert np.linalg.norm(diff) < 1e-8
g.message("Slice-trace test passed")
g.message(f"Norm(diff) = {np.linalg.norm(diff)}")