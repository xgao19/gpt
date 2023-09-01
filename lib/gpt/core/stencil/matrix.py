#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import cgpt


def parse(c):
    if isinstance(c, tuple):
        assert len(c) == 4
        return {"target": c[0], "accumulate": c[1], "weight": c[2], "factor": c[3]}
    return c


class matrix:
    def __init__(self, lat, points, code):
        self.points = points
        self.code = [parse(c) for c in code]
        self.obj = cgpt.stencil_matrix_create(lat.v_obj[0], lat.grid.obj, points, self.code)

    def __call__(self, *fields):
        cgpt.stencil_matrix_execute(self.obj, list(fields))

    def __del__(self):
        cgpt.stencil_matrix_delete(self.obj)
