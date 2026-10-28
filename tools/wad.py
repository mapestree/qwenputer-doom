"""Minimal Doom WAD reader: one map's VERTEXES, LINEDEFS, SIDEDEFS, SECTORS, THINGS."""
import struct
from collections import namedtuple

Line = namedtuple("Line", "v1 v2 flags special tag right left")
Side = namedtuple("Side", "sector")
Sector = namedtuple("Sector", "floor ceil special tag")
Thing = namedtuple("Thing", "x y angle type flags")


def read_map(path, name="E1M1"):
    b = open(path, "rb").read()
    kind, n, off = struct.unpack_from("<4sii", b, 0)
    lumps = [struct.unpack_from("<ii8s", b, off + 16 * i) for i in range(n)]
    names = [l[2].rstrip(b"\0").decode() for l in lumps]
    i0 = names.index(name)
    lump = {names[i]: b[lumps[i][0]:lumps[i][0] + lumps[i][1]] for i in range(i0 + 1, i0 + 11)}
    it = lambda fmt, data: list(struct.iter_unpack(fmt, data))
    verts = it("<hh", lump["VERTEXES"])
    lines = [Line(*t) for t in it("<HHHHHHH", lump["LINEDEFS"])]
    sides = [Side(t[5]) for t in it("<hh8s8s8sH", lump["SIDEDEFS"])]
    sectors = [Sector(t[0], t[1], t[5], t[6]) for t in it("<hh8s8shHH", lump["SECTORS"])]
    things = [Thing(*t) for t in it("<hhHHH", lump["THINGS"])]
    return verts, lines, sides, sectors, things
