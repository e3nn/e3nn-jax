import jax.numpy as jnp
import plotly.graph_objects as go
import e3nn_jax as e3nn


def get_cmap(x):
    if x == "bwr":
        return [[0, "rgb(0,50,255)"], [0.5, "rgb(200,200,200)"], [1, "rgb(255,50,0)"]]
    if x == "plasma":
        return [
            [0, "#9F1A9B"],
            [0.25, "#0D1286"],
            [0.5, "#000000"],
            [0.75, "#F58C45"],
            [1, "#F0F524"],
        ]


alpha = jnp.linspace(0, 2 * jnp.pi, 200)
beta = jnp.linspace(0, jnp.pi, 200)

alpha, beta = jnp.meshgrid(alpha, beta, indexing="ij")
vectors = e3nn.angles_to_xyz(alpha, beta)

signal = e3nn.spherical_harmonics(
    "8e", vectors, normalize=True, normalization="component"
).array
signal = signal[:, :, 8]

data = [
    go.Surface(
        x=jnp.abs(signal) * vectors[:, :, 0],
        y=jnp.abs(signal) * vectors[:, :, 1],
        z=jnp.abs(signal) * vectors[:, :, 2],
        surfacecolor=signal,
        showscale=False,
        cmin=-1.5,
        cmax=1.5,
        colorscale=get_cmap("bwr"),
    )
]

axis = dict(
    showbackground=False,
    showticklabels=False,
    showgrid=False,
    zeroline=False,
    title="",
    nticks=3,
    range=[-3, 3],
)

layout = dict(
    width=512,
    height=512,
    scene=dict(
        xaxis=dict(**axis),
        yaxis=dict(**axis),
        zaxis=dict(**axis),
        aspectmode="manual",
        aspectratio=dict(x=4, y=4, z=4),
        camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=5),
            projection=dict(type="orthographic"),
        ),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
)

fig = go.Figure(data=data, layout=layout)
fig.write_image("icon.png")
