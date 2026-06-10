import numpy as np
import yaml
from pathlib import Path
from fire import Fire
from loguru import logger
from tqdm import tqdm
import copy
from collections import defaultdict

# Change constants
CHANGE_LABELS = [    
    "static",
    "rigid", 
    "nonrigid", # non rigid cannot be added 
    "ambiguities", 
    "added", 
    "removed"
]

CHANGE_IDS = {label: idx for idx, label in enumerate(CHANGE_LABELS)}


class ScanNetAugmentation:
    def __init__(
        self,
        processed_data_dir: str = "./data/processed/scannet",
        save_dir: str = "./data/processed/scannet_sequence",
        modes: tuple = ("train", "validation"),
        enable_movement: bool = True,
        enable_removal: bool = True,
        enable_addition: bool = True,
        enable_duplication: bool = True,
        aug_params: dict = None,
    ):
        self.processed_dir = Path(processed_data_dir)
        self.save_dir = Path(save_dir)
        self.modes = modes
        
        # Default augmentations based on enabled flags ( for now leave this as the config)
        self.base_augmentations = [
            aug for aug, enabled in [
                (self._apply_duplication, enable_duplication)
            ] if enabled
        ]
        self.change_augmentations = [
            aug for aug, enabled in [
                (self._apply_movement, enable_movement),
                (self._apply_removal, enable_removal),
                (self._apply_addition, enable_addition),
            ] if enabled
        ]
        
        # Simplified scene-level augmentations for duplicate scenes
        self.scene_augmentations = [
            self._apply_instance_point_variation,    # Point-level jitter and deletions
            self._apply_instance_color_shift,        # Slight color shift to all instances
            self._apply_scene_alignment,             # Rotation and small translation
        ]
        ''' NOTE:
        No Overlap with Training Pipeline: Avoided augmentations that duplicate standard training transforms:
            No random rotation (already in training: RandomRotate)
            No random scale (already in training: RandomScale)
            No elastic distortion (already in training: ElasticDistortion)
            No chromatic jitter (already in training: ChromaticJitter, ChromaticTranslation)
            No random dropout (already in training: RandomDropout)
        '''
        
        
        # Default parameters
        default_params = {
            'num_augmented': 10,
            'seed': 59,
            
            # instance augmentations
            'movement_pct': [0, 0.4],
            'removal_pct': [0, 0.1], 
            'addition_pct': [0, 0.1],
            'ambiguity_pct': [0, 0.05],
            'duplicates': [1,4],

            'rotation': 0.5,
            'z_shift_max': 0.2,
            'excluded_labels' : [0, 1, 2, 22],
            
            # Simplified scene augmentations
            'point_jitter_std': 0.005,              # 5mm point jitter
            'point_dropout_prob': 1.0,               # all instances drop points
            'point_dropout_range': [0.00, 0.10],    # Drop 0-10% of points
            'point_duplicate_prob': 1.0,             # all instances duplicate points
            'point_duplicate_range': [0.00, 0.10],  # Duplicate 0-10% of points
            'normal_jitter_std': 0.1,                # Normal vector jitter std
            'color_shift_range': 10,                 # ±10 RGB units
            'scene_rotation_std': 0.02,              # ~1 degree rotation std
            'scene_translation_std': 0.05,           # 5cm translation std
        }
        self.params = {**default_params, **(aug_params or {})}
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, sequence_type='sliding', sequence_length=4):
        """Main pipeline"""

        self._copy_original_data()
        
        logger.info("Loading data...")
        self.databases = self._load_databases()
        
        logger.info("Creating augmented scenes...")
        aug_databases = self._create_augmented_scenes()
        
        logger.info("Creating temporal sequences...")
        sequences = self._create_sequences(aug_databases, sequence_type, sequence_length)
        
        logger.info("Saving results...")
        self._save_results(aug_databases, sequences, sequence_type, sequence_length)
        
        logger.info(f"Done! Created {len(sequences)} sequences with {sum(len(db) for db in aug_databases.values())} scenes")

    def _copy_original_data(self):
        """Copy all files and label databases from processed_dir to save_dir, preserving structure"""
        if self.processed_dir.resolve() == self.save_dir.resolve():
            return

        for item in self.processed_dir.rglob("*"):
            rel_path = item.relative_to(self.processed_dir)
            dst_path = self.save_dir / rel_path
            if item.is_dir():
                dst_path.mkdir(parents=True, exist_ok=True)
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if not dst_path.exists():
                    if item.suffix == ".npy":
                        np.save(dst_path, np.load(item))
                    else:
                        with open(item, "rb") as fsrc, open(dst_path, "wb") as fdst:
                            fdst.write(fsrc.read())


    def _load_databases(self):
        """Load existing scene databases"""
        databases = {}
        for mode in self.modes:
            db_path = self.processed_dir / f"{mode}_database.yaml"
            with open(db_path, 'r') as f:
                databases[mode] = yaml.safe_load(f)
        return databases

    def _create_augmented_scenes(self):
        """Create augmented versions of each scene"""
        aug_databases = {}
        new_index = 0 # For renumbering sub-scenes
        
        for mode in self.modes:
            aug_databases[mode] = []
            
            # Create augmented versions
            for original_info in tqdm(self.databases[mode], desc=f"Augmenting {mode}"):
                # Copy originals renumber sub-scenes
                scene_info = self._copy_scene(original_info, new_index, mode)
                scene_info = self._augment_scene(scene_info, aug_id=0, augmentations=self.base_augmentations)
                aug_databases[mode].append(scene_info)
                new_index += 1
                
                for aug_id in range(1, self.params['num_augmented'] + 1):
                    augmented = self._augment_scene(scene_info, aug_id, augmentations=self.change_augmentations)
                    aug_databases[mode].append(augmented)
        
        return aug_databases

    def _augment_scene(self, scene_info, aug_id, augmentations=[]):
        """Create single augmented scene"""
        rng = np.random.default_rng(self.params['seed'] + aug_id * 1000 + scene_info['scene'])
        
        # Load scene data, gt data will be automatically rewritten to reflect updates 
        scene_data = copy.deepcopy(np.load(scene_info['filepath'])) #TODO: load from baseline for ambiguity
        
        coords = scene_data[:, :3]
        semantic_labels = scene_data[:, 10].astype(int)
        instance_labels = scene_data[:, 11].astype(int)
        scene_bbox = np.array([np.min(coords, axis=0), np.max(coords, axis=0)])
        
        # get valid instances for change (not background, ceiling, floor, walls etc)
        valid_instance_mask = (instance_labels >= 0) & ~np.isin(semantic_labels, self.params['excluded_labels'])
        instances = np.unique(instance_labels[valid_instance_mask])
        
        #         
        scene_type = scene_info.get('scene_type', 'unknown')
        same_type_scenes = [s for s in self.databases[scene_info['mode']] 
                           if s.get('scene_type') == scene_type and s['scene'] != scene_info['scene']]

        if len(instances) == 0:
            # Nothing to augment; carry through as-is for this sub_scene
            scene = {
                "info": copy.deepcopy(scene_info),
                "data": scene_data,
                "bbox": scene_bbox,
                "instances": [],
                'similar_scenes': same_type_scenes
            }
            scene['info']['sub_scene'] = aug_id
            return self._save_augmented_scene(scene)
                
        # group scene information 
        scene = {
            "info": copy.deepcopy(scene_info)      , 
            "data": scene_data, 
            "bbox": scene_bbox, 
            "instances": instances.tolist(),
            'similar_scenes': same_type_scenes
        }
        scene['info']['sub_scene'] = aug_id
        
        # Apply instance-level augmentations first
        for aug_func in augmentations:
            scene = aug_func(scene, rng)
        
        # Apply scene-level augmentations (noise, partial scans)
        for aug_func in self.scene_augmentations:
            scene = aug_func(scene, rng)

        # Clean change labels after scene augmentations
        scene = self._clean_change_labels(scene)
        
        # update global information for used instances
        scene_info['used_ids'] = scene['info']['used_ids']
        
        return self._save_augmented_scene(scene)

    def _apply_movement(self, scene, rng):
        """Move random instances"""
        num_move = int(rng.uniform(*self.params['movement_pct']) * len(scene['instances']))
        if num_move == 0:
            return scene
            
        moved_ids = rng.choice(scene['instances'], size=num_move, replace=False)
        
        for instance_id in moved_ids:
            mask = scene['data'][:, 11] == instance_id
            placed_instance = self._place_instance(scene, scene['data'][mask], rng)
            if placed_instance is not None: 
                scene['data'][mask] = placed_instance
                scene['info']['changes']['rigid'].append(int(instance_id))
        
        return scene
    
    def _apply_removal(self, scene, rng):
        """Remove random instances"""
        num_remove = int(rng.uniform(*self.params['removal_pct']) * len(scene['instances']))
        
        if num_remove > 0:
            removed_ids = rng.choice(scene['instances'], size=min(num_remove, len(scene['instances'])), replace=False)
            
            for instance_id in removed_ids:
                mask = scene['data'][:, 11] == instance_id
                scene['data'] = scene['data'][~mask]  # remove instance points 
                scene['info']['changes']['removed'].append(int(instance_id))
                scene['instances'].remove(instance_id)
        
        return scene

    def _apply_addition(self, scene, rng):
        """Add random instances from other scenes of same type"""
        num_add = int(rng.uniform(*self.params['addition_pct']) * len(scene['instances']))
        if num_add == 0 or len(scene['similar_scenes']) == 0 :
            return scene
        
        for i in range(num_add):
            # Sample random instance from random scene
            source_scene = rng.choice(scene['similar_scenes'])
            instance_data = self._sample_instance(source_scene, rng)
            
            if instance_data is not None:
                placed_instance = self._place_instance(scene, instance_data, rng)
                if placed_instance is not None:
                    scene, new_id = self._insert_instance(scene, placed_instance)
                    scene['info']['changes']['added'].append(new_id)
        
        return scene


    def _apply_duplication(self, scene, rng):
        """Create ambiguity groups by duplicating instances"""
        
        num_groups = int(rng.uniform(*self.params['ambiguity_pct']) * len(scene['instances']))
        
        if num_groups == 0:
            return scene 
        
        duplicate_ids = rng.choice(scene['instances'], size=min(num_groups, len(scene['instances'])), replace=False)
        
        for instance_id in duplicate_ids:
            
            group = [int(instance_id)]
            num_duplicates = rng.integers(*self.params['duplicates'])
            for _ in range(num_duplicates):
                mask = scene['data'][:, 11] == instance_id # mask needs to update as instances are inserted
                placed_instance = self._place_instance(scene, scene['data'][mask], rng)
                if placed_instance is not None:
                    scene, new_id = self._insert_instance(scene, placed_instance)
                    group.append(new_id)
            if len(group) > 1:
                scene['info']['changes']['ambiguities'].append(group)
        
        return scene

    def _sample_instance(self, scene_info, rng):
        """Sample random instance from scene"""
        try:
            scene_data = np.load(scene_info['filepath'])
            instance_labels = scene_data[:, 11].astype(int)
            semantic_labels = scene_data[:, 10].astype(int)
            
            valid_instance_mask = (instance_labels >= 0) & ~np.isin(semantic_labels, self.params['excluded_labels'])
            instances = np.unique(instance_labels[valid_instance_mask])
            
            if len(instances) == 0:
                return None
            
            # select and center chosen instance
            instance_id = rng.choice(instances)
            mask = instance_labels == instance_id
            instance_data = copy.deepcopy(scene_data[mask])
            centroid = np.mean(instance_data[:, :3], axis=0)
            instance_data[:, :3] -= centroid
            
            return instance_data
        except:
            return None

    def _place_instance(self, scene, instance_data, rng): 
        coords = instance_data[:, :3]
        centroid = np.mean(coords, axis=0)
        
        if np.allclose(centroid[2], 0, atol=1e-3):
            # If the instance is already centered make the offset similar to scene mean
            z_offset = scene['data'][:, 2].mean()
        else: 
            z_offset = centroid[2]
        
        #TODO: only sometimes rotate
        angle = rng.uniform(0, 2 * np.pi)  # 0 to 360 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # 3D rotation matrix around Z-axis
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])     
        coords = coords - centroid
        coords = coords @ rotation_matrix.T
        # coords = coords + centroid
        
        # Calculate allowable centroid placement region
        instance_min, instance_max = np.min(coords, axis=0), np.max(coords, axis=0)
        allowable_min = scene['bbox'][0] - instance_min
        allowable_max = scene['bbox'][1] - instance_max
        
        # Check if instance fits and apply z constraints
        if np.any(allowable_max < allowable_min):
            return None
        
        z_shift = self.params['z_shift_max'] * (scene['bbox'][1, 2] - scene['bbox'][0, 2])
        allowable_min[2] = max(allowable_min[2], z_offset - z_shift)
        allowable_max[2] = min(allowable_max[2], z_offset + z_shift)
        
        if allowable_max[2] <= allowable_min[2]:
            return None
        
        coords += rng.uniform(allowable_min, allowable_max)
        
        instance_data[:, :3] = coords 
        return instance_data
        
        
    def _insert_instance(self, scene, instance_data):
        """Process instance insert"""
        
        new_id = max(scene['info']['used_ids']) + 1
        scene['info']['used_ids'].append(new_id)
        scene['instances'].append(new_id)
        
        # assign new segments and shift to prevent duplciates 
        new_segments = np.unique(instance_data[:, 9], return_inverse=True)[1]
        new_segments += (np.max(scene['data'][:, 9].astype('int64')) + 1)
    
        instance_data[:, 9] = new_segments
        instance_data[:, 11] = np.repeat(new_id, len(instance_data))
        
        scene['data'] = np.vstack([scene['data'], instance_data])
        
        return scene, new_id
    
    def _apply_instance_point_variation(self, scene, rng):
        """Add point-level jitter, random deletions, duplications, and normal jitter to all instances"""
        instances = np.unique(scene['data'][:, 11].astype(int))
        
        for instance_id in instances:
            if instance_id < 0:  # Skip background
                continue
                
            mask = scene['data'][:, 11] == instance_id
            instance_points = scene['data'][mask].copy()
            
            if len(instance_points) == 0:
                continue
    
            
            # Random point dropout with varying amounts
            if rng.random() < self.params['point_dropout_prob']:
                dropout_pct = rng.uniform(*self.params['point_dropout_range'])
                # Ensure at least one point remains for each instance
                max_drops = max(0, len(instance_points) - 1)
                num_to_drop = min(int(len(instance_points) * dropout_pct), max_drops)
                if num_to_drop > 0:
                    drop_indices = rng.choice(len(instance_points), size=num_to_drop, replace=False)
                    instance_points = np.delete(instance_points, drop_indices, axis=0)
            
            # Random point duplication with varying amounts
            if rng.random() < self.params['point_duplicate_prob']:
                duplicate_pct = rng.uniform(*self.params['point_duplicate_range'])
                num_to_duplicate = int(len(instance_points) * duplicate_pct)
                if num_to_duplicate > 0:
                    duplicate_indices = rng.choice(len(instance_points), size=num_to_duplicate, replace=True)
                    duplicated_points = instance_points[duplicate_indices].copy()
                    # Add small jitter to duplicated points
                    duplicate_jitter = rng.normal(0, self.params['point_jitter_std'] * 0.5, duplicated_points[:, :3].shape)
                    duplicated_points[:, :3] += duplicate_jitter
                    instance_points = np.vstack([instance_points, duplicated_points])
            
            
            # Add jitter to coordinates
            jitter = rng.normal(0, self.params['point_jitter_std'], instance_points[:, :3].shape)
            instance_points[:, :3] += jitter
            
            # Add normal vector jitter if normals exist
            if instance_points.shape[1] > 9:  # Has normals (columns 6:9)
                normal_jitter = rng.normal(0, self.params['normal_jitter_std'], instance_points[:, 6:9].shape)
                instance_points[:, 6:9] += normal_jitter
                # Renormalize normals
                normal_norms = np.linalg.norm(instance_points[:, 6:9], axis=1, keepdims=True)
                instance_points[:, 6:9] = instance_points[:, 6:9] / (normal_norms + 1e-8)
            
            # Update scene data
            scene['data'] = scene['data'][~mask]  # Remove original instance
            if len(instance_points) == 0:
                # Fallback: keep one original point to preserve instance presence
                original_points = scene['data'][mask]
                if original_points.shape[0] > 0:
                    instance_points = original_points[:1].copy()
            if len(instance_points) > 0:
                scene['data'] = np.vstack([scene['data'], instance_points])
        
        return scene

    def _apply_instance_color_shift(self, scene, rng):
        """Apply slight color shift to all instances"""
        instances = np.unique(scene['data'][:, 11].astype(int))
        
        for instance_id in instances:
            if instance_id < 0:  # Skip background
                continue
                
            mask = scene['data'][:, 11] == instance_id
            if mask.any():
                # Apply same color shift to all points in instance
                color_shift = rng.uniform(-self.params['color_shift_range'], 
                                        self.params['color_shift_range'], 3)
                scene['data'][mask, 3:6] = np.clip(
                    scene['data'][mask, 3:6] + color_shift, 0, 255
                )
        
        return scene

    def _apply_scene_alignment(self, scene, rng):
        """Apply rotation and small translation to entire scene"""
        coords = scene['data'][:, :3]
        
        # Small rotation around Z-axis
        rotation_angle = rng.normal(0, self.params['scene_rotation_std'])
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        
        # Small translation
        translation = rng.normal(0, self.params['scene_translation_std'], 3)
        
        # Apply transformation around scene center
        center = coords.mean(0)
        coords_transformed = (coords - center) @ rotation_matrix.T + center + translation
        scene['data'][:, :3] = coords_transformed
        
        return scene
    
    def _clean_change_labels(self, scene):
        """Clean change labels after scene-level augmentations"""
        remaining_instances = set(np.unique(scene['data'][:, 11].astype(int)))
        changes = scene['info']['changes']
        
        # Clean moved instances - if completely removed, they can't be moved
        changes['rigid'] = [iid for iid in changes['rigid'] if iid in remaining_instances]
        
        # Clean added instances - if completely removed, they're no longer added
        changes['added'] = [iid for iid in changes['added'] if iid in remaining_instances]
        
        return scene
    
    def _copy_scene(self, scene_info, aug_id, mode):
        """Copy original scene to augmented directory"""
        new_info = copy.deepcopy(scene_info)
        new_info['augmentation_source'] = f"scene{scene_info['scene']:04}_{scene_info['sub_scene']:02}"
        new_info['scene'] = aug_id
        new_info['sub_scene'] = 0 # Original sub-scene is 0
        new_info['mode'] = mode
        
        # Copy scene file
        scene_data = np.load(new_info['filepath'])
        new_path = self.save_dir / mode / f"{new_info['scene']:04}_{new_info['sub_scene']:02}.npy"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(new_path, scene_data)
        new_info['filepath'] = str(new_path)
        new_info['used_ids'] = np.unique(scene_data[:, 11].astype(int)).tolist()
        
        # Copy instance GT if exists
        if 'instance_gt_filepath' in new_info:
            gt_data = np.loadtxt(new_info['instance_gt_filepath'], dtype=np.int32)
            gt_path = self.save_dir / "instance_gt" / mode / f"scene{new_info['scene']:04}_{new_info['sub_scene']:02}.txt"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(gt_path, gt_data, fmt="%d")
            new_info['instance_gt_filepath'] = str(gt_path)
        
        new_info['changes'] = {'rigid': [], 'removed': [], 'added': [], 'ambiguities': []}
        return new_info

    def _save_augmented_scene(self, scene):
        """Save augmented scene"""
        info = scene['info']
        info['file_len'] =  len(scene['data'])
        
        # Save scene file     
        new_path = self.save_dir / info['mode'] / f"{info['scene']:04}_{info['sub_scene']:02}.npy"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(new_path, scene['data'])
        info['filepath'] = str(new_path)
        
        # Update color stats
        features = scene['data'][:, 3:6]
        info["color_mean"] = [float((features[:, i] / 255).mean()) for i in range(3)]
        info["color_std"] = [float(((features[:, i] / 255) ** 2).mean()) for i in range(3)]
        
        # Save instance GT
        if info['mode'] in ['train', 'validation']:
            gt_data = scene['data'][:, 10].astype(int) * 1000 + scene['data'][:, 11].astype(int) + 1
            gt_path = self.save_dir / "instance_gt" / info['mode'] / f"scene{info['scene']:04}_{info['sub_scene']:02}.txt"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(gt_path, gt_data.astype(np.int32), fmt="%d")
            info["instance_gt_filepath"] = str(gt_path)
        
        return info

    def _generate_change_labels(self, scene1, scene2):
        """Generate change labels between two scenes with precedence rules and order-aware added/removed"""
        
        # Initialize change labels
        changes = {}
        
        # Apply changes with precedence: ambiguity > (added/removed) > movement > static
        
        # 1. Ambiguity
        changes['ambiguities'] = scene1['changes']['ambiguities'] # all subscenes will have the same ambiguity groups
        
        assigned = {iid for group in changes['ambiguities'] for iid in group}
        
        # 2. Order-aware Added/Removed (trumps movement)
        changes['added'] = []
        for iid in scene1['changes']['removed'] + scene2['changes']['added']:
            if iid not in assigned:
                assigned.add(iid)
                changes['added'].append(iid)
                
        changes['removed'] = []
        for iid in scene1['changes']['added'] + scene2['changes']['removed']:
            if iid not in assigned:
                assigned.add(iid)
                changes['removed'].append(iid)

        changes['rigid'] = []
        for iid in scene1['changes']['rigid'] + scene2['changes']['rigid']:
            if iid not in assigned:
                changes['rigid'].append(iid)
        
        return changes

    def _create_sequences(self, aug_databases, sequence_type, sequence_length):
        """Create temporal sequences"""
        rng = np.random.default_rng(self.params['seed'])
        
        # Build mapping: scene id -> list of sub-scenes (with mode and changes)
        scenes_by_id = {}
        for mode in self.modes:
            for s in aug_databases[mode]:
                entry = {
                    'scene': s['scene'],
                    'sub_scene': s['sub_scene'],
                    'mode': mode,
                    'changes': s['changes']
                }
                scenes_by_id.setdefault(s['scene'], []).append(entry)

        sequences = {}
        for scene_id, entries in scenes_by_id.items():
            # Identify base sub-scene (prefer sub_scene == 0)
            base = next((e for e in entries if e['sub_scene'] == 0))

            others = [e for e in entries if e is not base]
            if not others:
                continue

            shuffled = rng.permutation(others).tolist()

            # Pair base with each other; randomize which appears first
            for other in shuffled:
                if rng.random() < 0.5:
                    scene1, scene2 = base, other
                else:
                    scene1, scene2 = other, base
                seq_name = f"scene{scene1['scene']:04}_{scene1['sub_scene']:02}-scene{scene2['scene']:04}_{scene2['sub_scene']:02}"
                sequences[seq_name] = {
                    'scene': scene1['scene'],
                    'sub_scenes': [scene1['sub_scene'], scene2['sub_scene']],
                    'type': scene1['mode'],
                    **self._generate_change_labels(scene1, scene2)
                }

        return sequences

    def _get_scene_mode(self, scene_info, aug_databases):
        """Find which mode a scene belongs to"""
        scene, sub_scene = scene_info
        for mode in self.modes:
            for scene_data in aug_databases[mode]:
                if scene_data['scene'] == scene and scene_data['sub_scene'] == sub_scene:
                    return mode
        return 'train'

    def _save_results(self, aug_databases, sequences, sequence_type, sequence_length):
        """Save all results"""
        # Save augmented databases
        for mode, database in aug_databases.items():
            with open(self.save_dir / f"{mode}_database.yaml", 'w') as f:
                yaml.dump(database, f, default_flow_style=False)
        
        # Save sequence database
        with open(self.save_dir / f"sequence_database_{sequence_type}_{sequence_length}.yaml", 'w') as f:
            yaml.dump(sequences, f, default_flow_style=False)
        
        # Save change label database
        change_colors = {
            0: [128, 128, 128], 1: [255, 0, 0], 2: [0, 255, 0], 
            3: [0, 0, 255], 4: [255, 255, 0], 5: [255, 0, 255]
        }
        change_db = {
            i: {"color": change_colors[i], "name": label, "validation": True}
            for i, label in enumerate(CHANGE_LABELS)
        }
        with open(self.save_dir / "change_label_database.yaml", 'w') as f:
            yaml.dump(change_db, f, default_flow_style=False)
        
        # Save change ground truth files
        self._save_change_gt(sequences)

    def _save_change_gt(self, sequences):
        """Save change ground truth for sequences"""
        for seq_name, seq_info in tqdm(sequences.items(), desc="Saving change GT"):
            if seq_info['type'] == 'test':
                continue
                
            # Load instance GT for each stage
            all_instance_gt = []
            for sub_scene in seq_info['sub_scenes']:
                gt_path = self.save_dir / "instance_gt" / seq_info['type'] / f"scene{seq_info['scene']:04}_{sub_scene:02}.txt"
                if gt_path.exists():
                    gt_data = np.loadtxt(gt_path, dtype=np.int32)
                    all_instance_gt.append(gt_data % 1000 - 1)  # Extract instance IDs
            
            if not all_instance_gt:
                continue
                
            instance_gt = np.concatenate(all_instance_gt)
            num_stages = len(seq_info['sub_scenes']) - 1
            
            if num_stages > 0:
                change_gt = np.zeros((len(instance_gt), num_stages), dtype=np.int32)
                
                # Apply change labels
                for stage in range(num_stages):
                    # Ambiguity (highest priority)
                    for group in seq_info.get('ambiguities', []):
                        mask = np.isin(instance_gt, group)
                        change_gt[mask, stage] = CHANGE_IDS['ambiguities']

                    non_ambiguous = change_gt[:, stage] != CHANGE_IDS['ambiguities']
                    
                    # Other changes (apply flat lists across all stages)
                    for change_type in ['rigid', 'removed', 'added']:
                        instances = seq_info.get(change_type, [])
                        if len(instances) == 0:
                            continue
                        mask = np.isin(instance_gt, instances)
                        change_gt[mask & non_ambiguous, stage] = CHANGE_IDS[change_type]
            else:
                change_gt = np.zeros(len(instance_gt), dtype=np.int32)
            
            # Save change GT
            change_gt_path = self.save_dir / "change_gt" / seq_info['type'] / f"{seq_name}.txt"
            change_gt_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(change_gt_path, change_gt, fmt="%d")


def main(
    processed_data_dir: str = "./data/processed/scannet",
    save_dir: str = "./data/processed/scannet_sequence",
    sequence_type: str = 'sliding',
    sequence_length: int = 2,
):
    """
    ScanNet Instance Shuffling Augmentation
    
    Creates temporal sequences by augmenting ScanNet scenes with:
    - Movement: Translate instances within scene bounds
    - Removal: Remove instances from scenes  
    - Addition: Add instances from other scenes of same type
    - Ambiguity: Create groups of ambiguous/duplicate instances
    
    For duplicate scenes, applies simple augmentations:
    - Point-level jitter and random deletions per instance
    - Slight color shift per instance
    - Small rotation and translation for scene alignment
    
    Each original scene generates num_augmented+1 total scenes (original + augmented),
    which are then grouped into temporal sequences of specified length.
    """

    
    augmenter = ScanNetAugmentation(
        processed_data_dir=processed_data_dir,
        save_dir=save_dir,
    )
    
    augmenter.run(sequence_type=sequence_type, sequence_length=sequence_length)


if __name__ == "__main__":
    Fire(main)