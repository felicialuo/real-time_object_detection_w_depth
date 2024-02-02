import numpy as np
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista

@st.cache_resource
def stpv_readfile(dummy: str = "grid"):

    plotter = pv.Plotter()

    # axes
    axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
    axes.origin = (0, 0, 0)
    plotter.add_actor(axes.actor)

    # read mesh
    mesh = pv.read('../output/rs_recording/mesh/000650.ply')
    # rotate mesh
    # mesh = mesh.rotate_y(180, point=axes.origin, inplace=False)
    plotter.add_mesh(mesh, show_edges=False, edge_color="k")

    # read pointcloud
    pcd = pv.read('../output/office_polycam_pcd.ply')
    plotter.add_mesh(pcd, show_edges=False, edge_color="k")
    
    plotter.background_color = "white"
    plotter.view_isometric()
    plotter.window_size = [1000, 800]
    return plotter

stpyvista(stpv_readfile())