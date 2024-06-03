import torch
import plotly.graph_objects as go
import meshdist
def plot_triangle():
    fig = go.Figure()

    vertices = torch.tensor([[0.0, 1.0, 1.0],
                            [0.3, 0.8, 1.12],
                            [0.2, 0.6, 1.0],
                            [0.7, 1.2, 1.1],]
                            )
    faces = torch.tensor([[0, 1, 2],  # ])
                          [0, 1, 3]])

    plot_tris = vertices[torch.cat((faces, faces[..., [0]]), dim=-1)]

    # query_point = torch.tensor([[0.2, 0.8, 1.3]])
    # dist_, point_ = dist_to_tri_single_args(vertices[faces][0], query_point[0])

    query_point = torch.tensor([[0.2, 0.8, 1.3],
                                [0.05, 0.75, 1.3],
                                [1.05, 1.75, 1.3],
                                [0.23, 1.03, 1.1]
                                ])

    # dist_, point_ = dist_to_tri_single_args(vertices[faces][0], query_point[0])

    # plot_query = torch.cat((point_[None], query_point), dim=0)
#
    dists, points = dist_to_mesh(vertices[faces], query_point)

    for tri in plot_tris:
        fig.add_trace(

            go.Scatter3d(
                x=tri[:, 0],
                y=tri[:, 1],
                z=tri[:, 2],
                mode='lines+markers',
                name='face',

            ))

    for k, (point, qp) in enumerate(zip(points, query_point)):
        dist_, single_point = dist_to_tri_single_args(
            vertices[faces][0], qp)
        plot_query_gt = torch.stack((single_point, qp), dim=0)

        plot_query = torch.stack((point, qp), dim=0)

        fig.add_trace(go.Scatter3d(
            x=plot_query[:, 0],
            y=plot_query[:, 1],
            z=plot_query[:, 2],
            mode='lines+markers',
            name=f'query k={k}'
        ))
    fig.show()


def draw_triangle(vertices, faces):
    tris = vertices[faces]
    a, b, c = (tris[:, k, :] for k in range(3))
    ab = b - a
    bc = c - b
    ca = a - c

    plane_normal = torch.linalg.cross(ab, -ca)
    plane_normal = plane_normal/plane_normal.norm(dim=1)[:, None]

    means = (a + b + c)/3
    normals = torch.stack((means, means+0.3*plane_normal), dim=1)

    plot_tris = vertices[torch.cat((faces, faces[..., [0]]), dim=-1)]

    fig = go.Figure()

    for tri, normal in zip(plot_tris, normals):
        fig.add_trace(

            go.Scatter3d(
                x=tri[:, 0],
                y=tri[:, 1],
                z=tri[:, 2],
                mode='lines+markers',
                name='face',

            ))
        fig.add_trace(
            go.Scatter3d(
                x=normal[:, 0],
                y=normal[:, 1],
                z=normal[:, 2],
                mode='lines+markers',
                showlegend=False,
            )
        )

    return fig


if __name__ == '__main__':
    vertices = torch.tensor([[0.0, 1.0, 1.0],
                            [0.3, 0.8, 1.12],
                            [0.2, 0.6, 1.0],
                            [0.7, 1.2, 1.1],]
                            )
    faces = torch.tensor([[0, 1, 2],  # ])
                          [0, 1, 3]])

    # draw_triangle(vertices, faces).show()
    plot_triangle()
