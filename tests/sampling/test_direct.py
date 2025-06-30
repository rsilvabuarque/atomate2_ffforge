from pymatgen.core import Structure, Lattice, Element
from atomate2_ffforge.sampling.direct_sampler import sample_direct


def toy_struct(a=4.0):
    lat = Lattice.cubic(a)
    return Structure(lat, [Element("Li")], [[0, 0, 0]])


def test_direct_determinism():
    # generate 100 dummy Li bcc cells with varying lattice params
    structs = [toy_struct(a=3.5 + 0.01 * i) for i in range(100)]
    idx1 = sample_direct(structs, "battery", dft_budget=20, k=1, seed=0)
    idx2 = sample_direct(structs, "battery", dft_budget=20, k=1, seed=0)
    assert idx1 == idx2
    assert len(idx1) == 20
    # no duplicates when k=1
    assert len(set(idx1)) == 20
