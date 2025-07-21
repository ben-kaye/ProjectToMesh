# meshdist

A package that projects to a mesh (by projecting to the faces corresponding to the top-k vertices)

Yes I did just implement this without barycentric coordinates..!

## Dependencies
- torch
- scipy.spatial: KDTree

## Usage

```
import meshdist

query_projected = meshdist.project_to_mesh(vertices, faces, query)
```

## Projecting to faces

```
import meshdist.d3d as d3d

distances, projections = d3d.project_to_tri(tris, points)
```
