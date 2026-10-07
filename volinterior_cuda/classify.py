"""One-pass six-connected free-space classification on CPU or CUDA."""
import numpy as np


def connectivity_partition(surface, center_index, *, backend="cuda", return_labels=False):
    if backend == "cuda":
        import cupy as xp
        from cupyx.scipy import ndimage
        host = xp.asnumpy
    elif backend == "cpu":
        xp = np
        from scipy import ndimage
        host = np.asarray
    else:
        raise ValueError("backend must be cpu or cuda")
    surface = xp.asarray(surface, dtype=bool)
    if surface.ndim != 3 or len(center_index) != 3:
        raise ValueError("expected a 3-D mask and a three-component seed index")
    if any(i < 0 or i >= surface.shape[k] for k, i in enumerate(center_index)):
        raise ValueError("COM seed lies outside the grid")
    structure = xp.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, :] = structure[1, :, 1] = structure[:, 1, 1] = True
    components, number = ndimage.label(~surface, structure=structure)
    sizes = host(xp.bincount(components.ravel(), minlength=int(number) + 1))
    boundary = host(xp.unique(xp.concatenate([
        components[0].ravel(), components[-1].ravel(), components[:, 0].ravel(),
        components[:, -1].ravel(), components[:, :, 0].ravel(), components[:, :, -1].ravel()])))
    enclosed = np.ones(len(sizes), dtype=bool)
    enclosed[0] = False
    enclosed[boundary] = False
    seed = int(components[center_index].item())
    closed = bool(enclosed[seed])
    interior = int(sizes[enclosed].sum())
    selection = int(sizes[0])
    central = int(sizes[seed]) if closed else 0
    counts = {"interior_voxels": interior, "selection_voxels": selection,
              "exterior_voxels": int(surface.size) - interior - selection,
              "closure_pass": closed, "center_in_material": seed == 0,
              "center_connected_to_boundary": bool(seed and not closed),
              "central_lumen_voxels": central, "other_interior_voxels": interior - central,
              "interior_components": int(enclosed.sum()),
              "failure_reason": "" if closed else ("center_in_material" if seed == 0 else "center_connected_to_boundary")}
    labels = None
    if return_labels:
        mask = xp.asarray(enclosed)[components]
        labels = host(xp.where(surface, -5, xp.where(mask, 0, 5)).astype(xp.int8))
    return counts, labels
