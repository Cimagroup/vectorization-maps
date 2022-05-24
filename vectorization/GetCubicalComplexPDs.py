import numpy as np
from gudhi import CubicalComplex

__all__ = ["GetCubicalComplexPDs"]

def GetCubicalComplexPDs(img, img_dim):
    cub_filtration = CubicalComplex(
        dimensions=img_dim, top_dimensional_cells=img)
    cub_filtration.persistence()
    pds = [cub_filtration.persistence_intervals_in_dimension(0),
           cub_filtration.persistence_intervals_in_dimension(1)]

    pds[0] = pds[0][pds[0][:, 0] != np.inf]
    pds[0] = pds[0][pds[0][:, 1] != np.inf]
    pds[1] = pds[1][pds[1][:, 0] != np.inf]
    pds[1] = pds[1][pds[1][:, 1] != np.inf]

    return pds