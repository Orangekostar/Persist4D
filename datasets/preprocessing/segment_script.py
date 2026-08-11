import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from fire import Fire
import trimesh
from loguru import logger
import re

def obj_to_ply(obj_path, ply_path):
    """Convert OBJ file to PLY format while preserving normals."""
    mesh = trimesh.load(obj_path)
    mesh.vertex_normals  # ensure we have vertex normals
    mesh.export(ply_path, file_type='ply')
    return ply_path

def run_segmentation(ply_path, output_path, k_thresh=0.01, min_verts=20):
    """Run the segmentator binary on the PLY file."""
    try:
        cmd = ["./third_party/ScanNet/Segmentator/segmentator", str(ply_path), str(k_thresh), str(min_verts)]
        subprocess.run(cmd, check=True)
        
        # The segmentator outputs a JSON file with the same name as input
        seg_json = ply_path.with_suffix('.0.010000.segs.json')
        
        # Move to desired output location
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(seg_json), str(output_path))
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Segmentation failed for {ply_path}: {e}")
        return False

def process_3rscan_test(
    data_dir: str,
    save_dir: str,
    metadata_file: str,
):
    """Process all test scenes in 3RScan dataset."""
    save_dir = Path(save_dir)
    data_dir = Path(data_dir)
    
    # Load dataset metadata
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # Create temporary directory for PLY files
    temp_dir = Path("./temp_ply")
    temp_dir.mkdir(exist_ok=True)
    
    # Process each test scene
    for scan in metadata:
        if scan['type'] != 'test':
            continue
            
        # Process reference scan
        ref_name = scan['reference']
        mesh_path = data_dir / ref_name / 'mesh.refined.v2.obj'
        
        if not mesh_path.exists():
            logger.warning(f"Mesh file not found: {mesh_path}")
            continue
            
        # Convert to PLY
        ply_path = temp_dir / f"{ref_name}.ply"
        obj_to_ply(mesh_path, ply_path)
        
        # Run segmentation
        output_path = save_dir / f"{ref_name}.refined.0.010000.segs.json"
        success = run_segmentation(ply_path, output_path)
        
        if success:
            logger.info(f"Successfully processed {ref_name}")
        
        # Process rescans
        for rescan in scan['scans']:
            name = rescan['reference']
            mesh_path = data_dir / name / 'mesh.refined.v2.obj'
            
            if not mesh_path.exists():
                logger.warning(f"Mesh file not found: {mesh_path}")
                continue
                
            ply_path = temp_dir / f"{name}.ply"
            obj_to_ply(mesh_path, ply_path)
            
            output_path = save_dir / f"{name}.refined.0.010000.segs.json"
            success = run_segmentation(ply_path, output_path)
            
            if success:
                logger.info(f"Successfully processed {name}")
    
    # Cleanup
    for ply_file in temp_dir.glob("*.ply"):
        ply_file.unlink()
    temp_dir.rmdir()

def process_scannet_test(
    data_dir: str,
    save_dir: str,
    git_repo: str,
):
    """Process all test scenes in ScanNet dataset."""
    save_dir = Path(save_dir)
    data_dir = Path(data_dir)
    git_repo = Path(git_repo)
    
    # Read test split file
    with open(git_repo / "Tasks" / "Benchmark" / "scannetv2_test.txt") as f:
        test_scenes = f.read().strip().split("\n")
    
    # Create temporary directory for processing
    temp_dir = Path("./temp_ply")
    temp_dir.mkdir(exist_ok=True)
    
    # Process each test scene
    for scene in test_scenes:
        mesh_path = data_dir / "test" / scene / f"{scene}_vh_clean_2.ply"
        
        if not mesh_path.exists():
            logger.warning(f"Mesh file not found: {mesh_path}")
            continue

                    # Copy to temporary directory
        temp_ply = temp_dir / f"{scene}_vh_clean_2.ply"
        try:
            import shutil
            shutil.copy2(mesh_path, temp_ply)
        except Exception as e:
            logger.error(f"Failed to copy {mesh_path} to temporary directory: {e}")
            continue
        
        # The mesh is already in PLY format, so we can use it directly
        output_path = save_dir / f"{scene}_vh_clean_2.0.010000.segs.json"
        success = run_segmentation(temp_ply, output_path)

        # Clean up temporary PLY file
        temp_ply.unlink()
        
        if success:
            logger.info(f"Successfully processed {scene}")

def process_test_set(
    dataset: str = "3rscan",
    data_dir: str = None,
    save_dir: str = "./test_segments",
    metadata_file: str = None,
    git_repo: str = None,
):
    """Process test set for either 3RScan or ScanNet dataset.
    
    Args:
        dataset: Either '3rscan' or 'scannet'
        data_dir: Path to dataset directory
        save_dir: Path to save segmentation outputs
        metadata_file: Path to 3RScan metadata JSON (only for 3RScan)
        git_repo: Path to ScanNet git repo (only for ScanNet)
    """
    if dataset.lower() == "3rscan":
        if not metadata_file:
            raise ValueError("metadata_file is required for 3RScan processing")
        process_3rscan_test(data_dir, save_dir, metadata_file)
    
    elif dataset.lower() == "scannet":
        if not git_repo:
            raise ValueError("git_repo is required for ScanNet processing")
        process_scannet_test(data_dir, save_dir, git_repo)
    
    else:
        raise ValueError("dataset must be either '3rscan' or 'scannet'")

if __name__ == "__main__":
    Fire(process_test_set)
