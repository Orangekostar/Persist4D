from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
# import open3d
import trimesh 
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree


def load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    labels = None
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array(
            [data["red"], data["green"], data["blue"]], dtype=np.uint8
        ).T
    if "label" in data.dtype.names:
        labels = np.array(data["label"], dtype=np.uint32)
    return coords, feats, labels

def load_ply_keys(filepath, keys):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array(
            [data["red"], data["green"], data["blue"]], dtype=np.uint8
        ).T
    
    if len(keys) > 1: 
        labels = []    
        for label_key in keys:
            if label_key in data.dtype.names:
                labels.append(np.array(data[label_key], dtype=np.uint32))
            else: 
                labels.append(None)
    else: 
        labels = np.array(data[keys[0]])
    return coords, feats, labels

def load_ply_with_normals_keys(filepath, keys):
    # Load mesh using trimesh
    mesh = trimesh.load_mesh(str(filepath), process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)
    
    coords, feats, labels = load_ply_keys(filepath, keys)
    assert np.allclose(coords, vertices), "different coordinates"
    if feats is not None:
        feats = np.hstack((feats, normals))
    else:
        feats = normals
        
    return coords, feats, labels



def load_ply_with_normals(filepath):
    # Load mesh using trimesh
    mesh = trimesh.load_mesh(str(filepath), process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)
    
    coords, feats, labels = load_ply(filepath)
    assert np.allclose(coords, vertices), "different coordinates"
    if feats is not None:
        feats = np.hstack((feats, normals))
    else:
        feats = normals
        
    return coords, feats, labels


def load_obj_with_normals(filepath):
    # Load mesh using trimesh
    mesh = trimesh.load_mesh(str(filepath), process=False)
    coords = np.array(mesh.vertices, dtype=np.float32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Convert to float and normalize
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'): #visuals stored as a texture, convert to vertex colors 
        colors = mesh.visual.to_color().vertex_colors[:, :3] / 255.0  # Convert to float and normalize
    else:
        colors = np.zeros((len(coords), 3), dtype=np.float32)
    feats = np.hstack((colors, normals))
    
    return coords, feats

def mapping_labels_to_target(target_vertices, source_vertices):
    tree = KDTree(source_vertices)
    distances, indices = tree.query(target_vertices)
    return indices 



def write_point_cloud_in_ply(
    filepath: Path,
    coords: np.ndarray,
    feats: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    dtypes: Optional[List[Tuple[str, str]]] = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("label", "<u2"),
    ],
    comments: Optional[List[str]] = [""],
):
    combined_coords = tuple([coords])
    if feats is not None:
        combined_coords += tuple([feats])
    else:
        dtypes = dtypes[:3] + dtypes[-1:]
    if labels is not None:
        combined_coords += tuple([labels[:, np.newaxis]])
    else:
        dtypes = dtypes[:-1]
    combined_coords = np.hstack(combined_coords)
    ply_data = np.empty(len(coords), dtype=dtypes)
    for i, dtype in enumerate(dtypes):
        ply_data[dtype[0]] = combined_coords[:, i]
    ply_data = PlyData(
        [PlyElement.describe(ply_data, "vertex", comments=comments)]
    )
    ply_data.write(filepath)
