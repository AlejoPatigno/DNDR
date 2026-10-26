"""Build the explicitly indexed alanine-dipeptide NPZ used by data.load_data (keys xyz[N,A,3] in nm,
trajectory[N], phi_indices[4], psi_indices[4], bonds[B,2]).

Two sources:

  1) mdshare (FU Berlin, 3 x 250 ns, 1 ps per frame, 10 heavy atoms) -- no topology tools needed:
       python prepare_alanine.py --mdshare --out /kaggle/working/alanine.npz
     Uses alanine-dipeptide-3x250ns-heavy-atom-positions.npz and ...-backbone-dihedrals.npz, from
     --search folders if they are there (e.g. uploaded as a Kaggle Dataset), else downloaded from
     http://ftp.imp.fu-berlin.de/pub/cmb-data/ (Internet on).  The phi/psi atom quadruplets are NOT
     guessed: they are the unique bonded quadruplets whose recomputed dihedrals reproduce the
     published dihedral file (checked on every frame); bonds are read from the geometry.

  2) your own topology + trajectories (requires mdtraj):
       python prepare_alanine.py --topology ala.pdb --trajectories a.xtc b.xtc c.xtc --out alanine.npz
"""
import argparse
from pathlib import Path
import urllib.request
import numpy as np

MDSHARE_URL = "http://ftp.imp.fu-berlin.de/pub/cmb-data/"
POSITIONS = "alanine-dipeptide-3x250ns-heavy-atom-positions.npz"
DIHEDRALS = "alanine-dipeptide-3x250ns-backbone-dihedrals.npz"


def _dihedral(x, ids):
    from data import dihedral
    return dihedral(x, ids)


def _wrap(a):
    return np.angle(np.exp(1j*a))


def _fetch(name, folder, search=()):
    for d in search:
        hits = sorted(Path(d).rglob(name)) if Path(d).exists() else []
        if hits:
            return hits[0]
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder)/name
    if not path.exists():
        try:
            import mdshare
            return Path(mdshare.fetch(name, working_directory=str(folder)))
        except ImportError:
            pass
        print(f"downloading {MDSHARE_URL+name} ...", flush=True)
        tmp = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(MDSHARE_URL+name, tmp)
        tmp.rename(path)
    return path


def bonds_from_geometry(xyz, cutoff=.17):
    """Heavy-atom covalent bonds: median distance < cutoff nm (C-C 0.153, C-N 0.133-0.147, C=O 0.123;
    nonbonded heavy-atom pairs are > 0.22 nm)."""
    d = np.median(np.linalg.norm(xyz[:, :, None]-xyz[:, None], axis=-1), axis=0)
    i, j = np.triu_indices(len(d), 1)
    keep = d[i, j] < cutoff
    return np.column_stack([i[keep], j[keep]])


def match_quadruplet(xyz, target, bonds, tol=1e-3):
    """Bonded chains a-b-c-d whose dihedral equals `target` (radians) on every frame given."""
    nb = {k: set() for k in range(xyz.shape[1])}
    for a, b in bonds:
        nb[int(a)].add(int(b))
        nb[int(b)].add(int(a))
    chains = [(a, b, c, d) for b in nb for c in nb[b] for a in nb[b]-{c} for d in nb[c]-{b, a}]
    out = []
    for q in chains:
        err = np.abs(_wrap(_dihedral(xyz, q)-target))
        if np.nanmax(err) < tol and tuple(reversed(q)) not in out:
            out.append(q)
    return out


def prepare_mdshare(out, folder="/kaggle/working/mdshare", search=("/kaggle/input",), stride=1):
    pos_path = _fetch(POSITIONS, folder, search)
    dih_path = _fetch(DIHEDRALS, folder, search)
    with np.load(pos_path) as fh:
        pos = [fh[f"arr_{i}"][::stride].astype(np.float32) for i in range(3)]
    with np.load(dih_path) as fh:
        dih = [fh[f"arr_{i}"][::stride].astype(np.float64) for i in range(3)]
    atoms = pos[0].shape[1]//3
    if pos[0].shape[1] % 3 or any(p.shape[1] != 3*atoms for p in pos):
        raise ValueError(f"positions must be [frames, 3*atoms]; got {[p.shape for p in pos]}")
    xyz = [p.reshape(len(p), atoms, 3) for p in pos]
    if any(len(a) != len(b) for a, b in zip(xyz, dih)) or any(d.shape[1] != 2 for d in dih):
        raise ValueError("positions and dihedrals do not have matching frames / two angles")
    bonds = bonds_from_geometry(xyz[0][:2000])
    if len(bonds) != atoms-1:
        raise ValueError(f"expected a tree of {atoms-1} heavy-atom bonds, found {len(bonds)}: {bonds.tolist()}")
    probe = np.concatenate([x[:5000] for x in xyz])
    probe_d = np.concatenate([d[:5000] for d in dih])
    found = []
    for col, name in ((0, "phi"), (1, "psi")):
        q = match_quadruplet(probe, probe_d[:, col], bonds)
        if len(q) != 1:
            raise ValueError(f"{name}: {len(q)} quadruplets reproduce column {col} of {DIHEDRALS} "
                             "(expected exactly 1; is the dihedral file in radians / same frames?)")
        found.append(np.array(q[0]))
    phi, psi = found
    worst = max(float(np.nanmax(np.abs(_wrap(_dihedral(x, q)-d[:, c])))) for x, d in zip(xyz, dih)
                for c, q in ((0, phi), (1, psi)))
    if worst > 1e-3:
        raise ValueError(f"recomputed dihedrals differ from the published ones by up to {worst:.2e} rad")
    group = np.concatenate([np.full(len(x), i) for i, x in enumerate(xyz)])
    np.savez_compressed(out, xyz=np.concatenate(xyz), trajectory=group, phi_indices=phi, psi_indices=psi,
                        bonds=bonds, frame_ps=np.float32(stride), source=np.array(f"mdshare:{POSITIONS}"))
    print(f"Saved {out}: {len(group)} frames {[len(x) for x in xyz]}, {atoms} heavy atoms, "
          f"phi={phi.tolist()}, psi={psi.tolist()}, bonds={bonds.tolist()}, "
          f"max |dihedral - published| = {worst:.1e} rad, units=nm")
    return out


def prepare(topology, trajectories, out, stride=10, phi_column=0, psi_column=0):
    import mdtraj as md
    blocks, groups, phi, psi, bonds = [], [], None, None, None
    for i, path in enumerate(trajectories):
        for chunk in md.iterload(path, top=topology, chunk=10000, stride=stride):
            if chunk.unitcell_vectors is not None:
                chunk.make_molecules_whole(inplace=True)
            t = chunk.atom_slice(chunk.topology.select("not element H"))
            ph, _ = md.compute_phi(t, periodic=False)
            ps, _ = md.compute_psi(t, periodic=False)
            if phi is None:
                phi, psi = ph[phi_column], ps[psi_column]
                bonds = np.array([(a.index, b.index) for a, b in t.topology.bonds], dtype=int)
            elif not np.array_equal(phi, ph[phi_column]) or not np.array_equal(psi, ps[psi_column]):
                raise ValueError("Inconsistent atom ordering between trajectories.")
            blocks.append(t.xyz)
            groups.append(np.full(len(t), i))
    if len(trajectories) < 3 or not blocks:
        raise ValueError("At least three nonempty independent trajectories required.")
    np.savez_compressed(out, xyz=np.concatenate(blocks), trajectory=np.concatenate(groups),
                        phi_indices=phi, psi_indices=psi, bonds=bonds)
    print(f"Saved {out}; phi={phi}, psi={psi}, units=nm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mdshare", action="store_true", help="build from the mdshare heavy-atom NPZ files")
    p.add_argument("--mdshare-dir", default="/kaggle/working/mdshare")
    p.add_argument("--search", nargs="*", default=["/kaggle/input"], help="folders searched before downloading")
    p.add_argument("--topology")
    p.add_argument("--trajectories", nargs="+")
    p.add_argument("--out", default="alanine.npz")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--phi-column", type=int, default=0)
    p.add_argument("--psi-column", type=int, default=0)
    a = p.parse_args()
    if a.mdshare:
        prepare_mdshare(a.out, a.mdshare_dir, a.search, a.stride or 1)
    else:
        if not a.topology or not a.trajectories:
            p.error("use --mdshare, or give --topology and --trajectories")
        prepare(a.topology, a.trajectories, a.out, a.stride or 10, a.phi_column, a.psi_column)
