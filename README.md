# meshdist

A package that projects to a mesh (by projecting to the faces corresponding to the top-k vertices)

# Dependencies
- torch
- scipy.spatial: KDTree

# Usage

```
import meshdist

query_projected = meshdist.project_to_mesh(vertices, faces, query)
```
