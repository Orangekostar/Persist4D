import re
from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from loguru import logger
import json
from itertools import permutations
from tqdm import tqdm

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_obj_with_normals, load_ply_with_normals_keys, mapping_labels_to_target

from datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    CLASS_LABELS_200,
    VALID_CLASS_IDS_200,
)

from datasets.RScan.RScan_constants import (
    get_scannet200_label, 
    VALID_CHANGE_IDS, 
    CHANGE_LABELS, 
    RIO_CHANGE_COLOR_MAP
)


class RScanPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "/oak/stanford/groups/iarmeni/easteine/3RScan",
        save_dir: str = "/scratch/users/easteine/mask3d-3RScan",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
        metadata_file: str = None,
        scannet200: bool = True,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)
        
        # if globalID map to scannet200
        #TODO: drop other nyu 40 labels to match with scannet labels 
        self.label_key = "globalId" if scannet200 else "NYU40"
        self.scannet200 = scannet200    

        self.create_label_database()
        self.create_change_label_database()
        self.write_metric_spec()
        
        if metadata_file is None:
            metadata_file = self.data_dir / "3RScan.json"
            

        # Load dataset metadata containing train/val/test splits and scan sequences
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        self.scenes = self.__generate_scene_subscene_mapping()
        
        for mode in self.modes:
            mode_scenes = {k: v for k, v in self.scenes.items() if v['mode'] == mode}

            filepaths = []
            #TODO: for now just use the mesh vertices as the pointclouds 
            # add each reference scene and rescan as individual scenes 
            # assign integer ids to reference scans and rescan 
            for scene in mode_scenes.keys():
                path = self.data_dir / scene / 'mesh.refined.v2.obj'
                filepaths.append(path) 
                
            # no sorting (3RScan are labelled with random keys)
            self.files[mode] = (filepaths)

    def __generate_scene_subscene_mapping(self) -> None:

        scan_id_mapping = {}

        scan_id = 0  # Global scan ID counter
        for scan in self.metadata:
            # add each reference scene and rescan as individual scenes 
            # assign integer ids to reference scans and rescan 
            rescan_id = 0
            ref_name = scan['reference']
            scan_id_mapping[ref_name] = {
                'mode' : scan['type'], 
                'scan_id' : scan_id,
                'rescan_id' : rescan_id, 
                'metadata_key': [ref_name],
                'transformation' : np.eye(4)
            }
            
            # add all additional rescans 
            for rescan in scan['scans']:
                rescan_id += 1
                name = rescan['reference'] 
                scan_id_mapping[name] = {
                    'mode' : scan['type'], 
                    'scan_id' : scan_id,
                    'rescan_id' : rescan_id, 
                    'metadata_key': [ref_name, name], 
                    'transformation' : np.array(rescan['transform']).reshape(4,4).T if 'transform' in rescan else np.eye(4),
                }
                
            scan_id +=1
        return scan_id_mapping
    
    def create_sequences(self, sequence_type='sliding', sequence_length=2, seed=45):
        sequences = []
        
        # set only one rng with a seed for evaluation reproducibility
        rng = np.random.default_rng(seed)
        
        all_scenes = np.array([self._parse_scene_subscene(name) for name in self.scenes.keys()])
        
        # create a boolean mask for each scene 
        _, group_idx = np.unique(all_scenes[:, 0], return_inverse=True)
        scene_masks = np.eye(np.max(group_idx) + 1)[group_idx].T.astype(bool)

        for group in scene_masks:
            scene_names = all_scenes[group]

            if sequence_type == 'sliding':
                # IMPORTANT: Do not "wrap" scenes that don't have enough unique scans.
                # Otherwise, for e.g. sequence_length=3 and only 2 rescans, modulo indexing
                # would repeat scans and incorrectly create fake t=3 windows.
                if len(scene_names) < sequence_length:
                    continue
                scene_names = rng.permutation(scene_names).tolist()
                for i, name in enumerate(scene_names): 
                    
                        # Create a cyclic sliding window of <sequence_length> scans.
                        sequences.append([
                            scene_names[(i + offset) % len(scene_names)] 
                            for offset in range(sequence_length)
                        ])
            elif sequence_type == 'exhaustive':
                # all possible combinations in the scene (n, l)
                #TODO: when sequence_length is too large wrap scans
                sequences.extend([list(combo) for combo in permutations(scene_names.tolist(), sequence_length)])
            elif sequence_type == 'single': 
                # ignores set sequence_length and group scans in the same scene together in a random order 
                scene_names = rng.permutation(scene_names).tolist()
                sequences.append(scene_names)
            else:
                raise NotImplementedError

        return sequences 
    
    
    def process_sequences(self, sequence_type='sliding', sequence_length=2):
        sequence_database = {}
        
        # create and store the predefined sequences for each scene 
        sequences = self.create_sequences(sequence_type, sequence_length)
    
        # store the temporal instance information (ambiguity, change type)
        for sequence in tqdm(sequences, unit="file"):
            name = '-'.join(f'scene{self.__generate_name(*scene)}' for scene in sequence)
            
            sequence_database[name] = {}
            sequence_database[name]['scene'] = sequence[0][0]
            sequence_database[name]['sub_scenes'] = [seq[1] for seq in sequence]
            sequence_database[name]['type'] = self.metadata[sequence[0][0]].get('type', '') # train, val, test
            
            # get ambiguity information from reference scan
            ambiguities_data = self.metadata[sequence[0][0]].get('ambiguity', [])
            # first get ambiguity pairs 
            ambiguity_pairs = {
                pair['instance_source']: pair['instance_target'] 
                for pair in (ambiguities_data[0] if ambiguities_data else [])
            }
            # then make sure to include all ambiguity options in the loop
            ambiguity_groups = []
            all_ids = set(list(ambiguity_pairs.keys()) + list(ambiguity_pairs.values()))
            visited = set()

            for id in all_ids:
                if id in visited:
                    continue
                    
                # Start a new group with BFS
                current_group = []
                queue = [id]
                visited.add(id)
                
                while queue:
                    current = queue.pop(0)
                    current_group.append(current)
                    
                    # Check connections in both directions
                    target = ambiguity_pairs.get(current)
                    if target and target not in visited:
                        queue.append(target)
                        visited.add(target)
                        
                    # Also find any sources that point to current
                    for source, target in ambiguity_pairs.items():
                        if target == current and source not in visited:
                            queue.append(source)
                            visited.add(source)
                
                ambiguity_groups.append(sorted(current_group))
            
            sequence_database[name]['ambiguities'] = ambiguity_groups
            
            # compare the sets of two scenes (reference and rescan or two rescans) to determine the changes
            nonrigid, rigid, removed, added = [], [], [], []
            
            # start with initial scene as base 
            prev_scene = sequence[0]
            prev_nonrigid_changes, prev_rigid_changes, prev_removed_changes, prev_transform = self.__get_change_metadata(*prev_scene)
            
            
            # filter by objects which moved wrt to reference but not wrt to each other in the sequence 
            for scene in sequence[1:]:
                # Get changes for current scene/subscene as sets
                nonrigid_changes, rigid_changes, removed_changes, _ = self.__get_change_metadata(*scene)
                
                # Union with current sets (automatic deduplication)
                nonrigid.append(sorted(prev_nonrigid_changes | nonrigid_changes))
                
                # for each unique id within removed changes and prev removed changes 
                # if the id is present in both discard 
                # if the id is only present in the prev version add to added 
                # if the the id is present only in the second add to removed 
                removed.append(sorted(removed_changes - prev_removed_changes))
                added.append(sorted(prev_removed_changes - removed_changes))
                
                # if test mode, cannot check transformations, simply use the union 
                if sequence_database[name]['type'] == 'test':
                    rigid.append(sorted(rigid_changes | prev_rigid_changes))
                else: 
                    # add a rigid change if either scan if the reference scan
                    if prev_scene[1] == 0:
                        # TODO: add transformations as they are
                        rigid.append(sorted(list(rigid_changes.keys())))
                    elif scene[1] == 0:
                        # TODO: Invert transformations 
                        rigid.append(sorted(list(prev_rigid_changes.keys())))
                    else: # if both rescans
                        # add change if unique 
                        combine_rigid = rigid_changes.copy()
                        for key, prev_rigid in prev_rigid_changes.items():
                            if key in combine_rigid:
                                # get the centroid of the first scene 
                                to_instance_frame = np.linalg.inv(prev_transform @ self.__get_instance_frame(*prev_scene, key))
                                # add the rigid change if the transformation matrix is different (else its the same change)
                                relative_matrix = self.__relative_matrix(prev_rigid['transform'], rigid_changes[key]['transform'], 
                                                                 rigid_changes[key]['symmetry'], to_instance_frame)
                                # check if the relative transformation is significant compared to the original transformations from the reference scan
                                if self.__is_diff(to_instance_frame @ relative_matrix, threshold=0.5*self.__mag(to_instance_frame @ prev_rigid['transform'])):
                                    combine_rigid[key] = relative_matrix
                                else: 
                                    combine_rigid.pop(key)
                            else: 
                                combine_rigid[key] = np.linalg.inv(prev_rigid['transform']) #TODO: invert this matrix 
                        
                        # only pass on the change if it occures for now without the relative transformation 
                        rigid.append(sorted(list(combine_rigid.keys())))
                    
                prev_scene = scene
                prev_nonrigid_changes, prev_rigid_changes, prev_removed_changes = nonrigid_changes, rigid_changes, removed_changes
            


            # removed and added based on annotation only for now ()
            # sorted(list(nonrigid_set))
            sequence_database[name]['nonrigid'] = nonrigid
            sequence_database[name]['rigid'] = rigid
            sequence_database[name]['removed'] = removed
            sequence_database[name]['added'] = added
            
            # process the change data into a per vertex label and record filepath 
            sequence_database[name]['filepath'] = str(self.process_change_files(name, sequence_database[name])) 
            
            

        self._save_yaml(
            self.save_dir / f"sequence_database_{sequence_type}_{sequence_length}.yaml", sequence_database
        )
        return sequence_database
    
    def process_change_files(self, sequence_name, sequence_info):
        """process_file.

        Args:
            sequence_name 
            mode: typically train, test or validation

        Returns:
            filebase: info about file
        """
    
        # only dataset with temporal information included 
        if sequence_info['type'] == 'test':
            return 
        
        instance_gt = []
        for subscene in sequence_info['sub_scenes']:
            # load instance gt txt file 
            filename = self.save_dir / 'instance_gt' / sequence_info['type'] / \
                        f"scene{self.__generate_name(sequence_info['scene'],subscene)}.txt"
            # get only the instance ids
            instance_gt.append(np.loadtxt(filename, dtype=np.int32) % 1000)
        instance_gt = np.concatenate(instance_gt, axis=0)
        
        # change map n-1 layers 
        change_layers = len(sequence_info['sub_scenes'])-1
        
        if change_layers > 0: 
            change_gt = np.zeros((instance_gt.shape[0], change_layers), dtype=np.int32)
            
            # ambiguities first (apply to all stages)
            # label as ambiguity even if it also experienced a rigid change 
            ambiguity_mask = np.isin(instance_gt, sum(sequence_info['ambiguities'], []))
            change_gt[ambiguity_mask] = VALID_CHANGE_IDS[CHANGE_LABELS.index('ambiguities')]
            
            for label_idx, label in enumerate(CHANGE_LABELS):
                if label not in ['static', 'ambiguities']:
                    for stage in range(change_layers):
                        # set change label where change is documented in the sequence db 
                        change_gt[np.isin(instance_gt, sequence_info[label][stage]), stage] = VALID_CHANGE_IDS[label_idx]
                        
            # save the txt file with the change labels
                    # Save instance ground truth
            processed_gt_filepath = (
                self.save_dir
                / "change_gt"
                / sequence_info['type']
                / f"{sequence_name}.txt"
            )
            if not processed_gt_filepath.parent.exists():
                processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(processed_gt_filepath, change_gt.astype(np.int32), fmt="%d")
        else:
            # if a single scan
            change_gt = np.zeros((instance_gt.shape[0]), dtype=np.int32)
            processed_gt_filepath = (
                self.save_dir
                / "change_gt"
                / sequence_info['type']
                / f"{sequence_name}.txt"
            )
            if not processed_gt_filepath.parent.exists():   
                processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(processed_gt_filepath, change_gt.astype(np.int32), fmt="%d")
            
        return processed_gt_filepath

    def write_metric_spec(self):
        """Emit data/processed/{rio|rio200}/{name}.yaml for stmetrics."""
        from datasets.preprocessing.metric_spec_writer import (
            NYU40_SUBSET_18_IDS,
            NYU40_SUBSET_18_LABELS,
            resolve_200_subset,
            write_metric_spec,
        )

        if self.scannet200:
            from datasets.RScan.RScan_splits import CLASS_LABELS_200_VALIDATION
            from datasets.scannet200.scannet200_constants import (
                CLASS_LABELS_200,
                VALID_CLASS_IDS_200,
                VALID_PANOPTIC_IDS,
            )
            from datasets.scannet200.scannet200_splits import (
                COMMON_CATS_SCANNET_200,
                HEAD_CATS_SCANNET_200,
                TAIL_CATS_SCANNET_200,
            )

            class_labels, valid_class_ids = resolve_200_subset(
                full_labels=CLASS_LABELS_200,
                full_ids=VALID_CLASS_IDS_200,
                candidate_labels=CLASS_LABELS_200_VALIDATION,
                panoptic_ids=VALID_PANOPTIC_IDS,
            )
            path = write_metric_spec(
                name="rio200",
                class_labels=class_labels,
                valid_class_ids=valid_class_ids,
                aux_labels=CHANGE_LABELS,
                valid_aux_ids=VALID_CHANGE_IDS,
                categories={
                    "head": HEAD_CATS_SCANNET_200,
                    "common": COMMON_CATS_SCANNET_200,
                    "tail": TAIL_CATS_SCANNET_200,
                },
                out_dir=self.save_dir,
            )
        else:
            path = write_metric_spec(
                name="rio",
                class_labels=NYU40_SUBSET_18_LABELS,
                valid_class_ids=NYU40_SUBSET_18_IDS,
                aux_labels=CHANGE_LABELS,
                valid_aux_ids=VALID_CHANGE_IDS,
                out_dir=self.save_dir,
            )
        logger.info(f"wrote metric spec → {path}")

    def create_change_label_database(self):

        # Do not overwrite an existing change label database when (re)building a sequence DB.
        # This prevents accidentally switching label sets just by running the builder.
        existing = self.save_dir / "change_label_database.yaml"
        if existing.exists():
            return self._load_yaml(existing)

        change_label_database = {}
        for row_id, class_id in enumerate(VALID_CHANGE_IDS):
            change_label_database[class_id] = {
                "color": RIO_CHANGE_COLOR_MAP[class_id],
                "name": CHANGE_LABELS[row_id],
                "validation": True,
            }
        self._save_yaml(
            self.save_dir / "change_label_database.yaml", change_label_database
        )
        return change_label_database

    def create_label_database(self):

        # Do not overwrite an existing label database when (re)building a sequence DB.
        # This prevents accidentally switching between NYU40/ScanNet200 mappings.
        existing = self.save_dir / "label_database.yaml"
        if existing.exists():
            return self._load_yaml(existing)

        if self.scannet200:
            label_database = {}
            for row_id, class_id in enumerate(VALID_CLASS_IDS_200):
                label_database[class_id] = {
                    "color": SCANNET_COLOR_MAP_200[class_id],
                    "name": CLASS_LABELS_200[row_id],
                    "validation": True,
                }
            self._save_yaml(
                self.save_dir / "label_database.yaml", label_database
            )
            return label_database
        elif self.label_key == 'NYU40':
            #self.key == 'NYU40':
            # use onlt the nyu40 labels from scannet for inference validation purposes only 
            if (self.save_dir / "label_database.yaml").exists():
                return self._load_yaml(self.save_dir / "label_database.yaml")
            
            #TEMP: hardcode scannet directory and git to create the yaml 
            scannet_data = Path("datasets/scannet200/scannetv2-labels.combined.tsv")
            git_repo = Path("third_party/ScanNet")
            df = pd.read_csv(
                scannet_data, sep="\t"
            )
            df = (
                df[~df[["nyu40class", "nyu40id"]].duplicated()][
                    ["nyu40class", "nyu40id"]
                ]
                .set_index("nyu40id")
                .sort_index()[["nyu40class"]]
                .rename(columns={"nyu40class": "name"})
                .replace(" ", "_", regex=True)
            )
            df = pd.concat([pd.DataFrame([{"name": "empty"}]), df], ignore_index=True)
            df["name"] = df["name"].replace(
                {"refridgerator": "refrigerator"}
            )
            df["validation"] = False

            with open(
                git_repo
                / "Tasks"
                / "Benchmark"
                / "classes_SemVoxLabel-nyu40id.txt"
            ) as f:
                for_validation = f.read().split("\n")
            for category in for_validation:
                index = int(re.split(" +", category)[0])
                df.loc[index, "validation"] = True

            # doing this hack because otherwise I will have to install imageio
            with open(git_repo / "BenchmarkScripts" / "util.py") as f:
                util = f.read()
                color_list = eval("[" + util.split("return [\n")[1])

            df["color"] = color_list

            label_database = df.to_dict("index")
            self._save_yaml(
                self.save_dir / "label_database.yaml", label_database
            )
            return label_database
        else: 
            raise NotImplementedError

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file

            Required fields for all datasets:
                - filepath (str): Path to the processed numpy file labelled with int ids 
                - raw_filepath (str): Original path to the input file
                - file_len (int): Number of points in the point cloud
                - color_mean (List[float]): Mean RGB values [R, G, B] of the point cloud
                - color_std (List[float]): Standard deviation of RGB values [R, G, B]
                - instance_gt_filepath (str): Path to instance ground truth file (non-test modes only)
            
            Additional fields for 3RScan:
                - scene (int): Scene number 
                - sub_scene (int): Subscene number
                - raw_instance_filepath (str): Path to instance JSON (train/validation only)
                - raw_segmentation_filepath (str): Path to segmentation JSON file
                - raw_label_filepath (str): Path to labels file (train/validation only)

        """
        scene_name = filepath.parts[-2]
        scene, sub_scene = self._parse_scene_subscene(scene_name) 
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "sub_scene": sub_scene,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        # reading both files and checking that they are fitting
        coords_orig, features = load_obj_with_normals(filepath)
        file_len = len(coords_orig)
        filebase["file_len"] = file_len

        # apply calibration transformation 
        transform = self.scenes[scene_name]['transformation']
        coords = (transform[:3, :3] @ coords_orig.T + transform[:3, 3:]).T

        # coordinates and normals 
        points = np.hstack((coords, features))

        if mode in ["train", "validation"]:
            # reading labels file for mapping
            label_filepath = filepath.parent / "labels.instances.annotated.v2.ply"
            filebase["raw_label_filepath"] = label_filepath

            # Extract globalID which will then be mapped to scannet200 labels 
            label_coords, label_colors, labels = load_ply_with_normals_keys(
                label_filepath,
                keys = [self.label_key]
            )
            if self.scannet200:
                labels = np.array([get_scannet200_label(label) for label in labels])
            else: 
                labels = np.array(labels)

            # determine index mapping before calibration transformation
            # instance file spatially aligned with before calibration obj file 
            mapping_indices = mapping_labels_to_target(coords_orig, label_coords)

            # getting instance and segment info
            instance_info_filepath = next(
                Path(filepath).parent.glob("*semseg.v2.json")
            )
            segment_indexes_filepath = next(
                Path(filepath).parent.glob("*mesh.refined.0.010000.segs.v2.json")
            )
            instance_db = self._read_json(instance_info_filepath)
            segments = self._read_json(segment_indexes_filepath)
            segments = np.array(segments["segIndices"])
            filebase["raw_instance_filepath"] = instance_info_filepath
            filebase["raw_segmentation_filepath"] = segment_indexes_filepath

            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]

            points = np.hstack((points, segment_ids[..., None][mapping_indices]))

            # adding instance label
            labels = labels[:, np.newaxis]
            empty_instance_label = np.full(labels.shape, -1)
            labels = np.hstack((labels, empty_instance_label))
            for instance in instance_db["segGroups"]:
                segments_occupied = np.array(instance["segments"])
                occupied_indices = np.isin(segments, segments_occupied)
                labels[occupied_indices, 1] = instance["id"]

            points = np.hstack((points, labels[mapping_indices]))

            # do not shift the semantic label or instance label 
            gt_data = points[:, -2] * 1000 + points[:, -1]

        else: 
            # get preprocessed segments
            segments_test = "data/raw/rio_test_segments"
            segment_indexes_filepath = f"{scene_name}.refined.0.010000.segs.json"
            segments = self._read_json(
                f"{segments_test}/{segment_indexes_filepath}"
            )
            segments = np.array(segments["segIndices"])
            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]
            points = np.hstack((points, segment_ids[..., None]))

        # Save processed point cloud
        processed_filepath = (
            self.save_dir / mode / f"{self.__generate_name(scene, sub_scene)}.npy"
        )
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        if mode == "test":
            return filebase

        # Save instance ground truth
        processed_gt_filepath = (
            self.save_dir
            / "instance_gt"
            / mode
            / f"scene{self.__generate_name(scene, sub_scene)}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)


        # Calculate color statistics if train or val 
        filebase["color_mean"] = [
            float((features[:, 0] / 255).mean()),
            float((features[:, 1] / 255).mean()),
            float((features[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((features[:, 0] / 255) ** 2).mean()),
            float(((features[:, 1] / 255) ** 2).mean()),
            float(((features[:, 2] / 255) ** 2).mean()),
        ]

        return filebase

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/rio200/train_database.yaml",
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)        


    @logger.catch
    def fix_bugs_in_labels(self):
        # fix labels only in scannet
        pass

    def _parse_scene_subscene(self, name):
        return int(self.scenes[name]['scan_id']), int(self.scenes[name]['rescan_id'])
    
    def __generate_name(self, scene, sub_scene):
        return f"{scene:04}_{sub_scene:02}"
    
    def __get_instance_frame(self, scene, sub_scene, instance_id): 
        # get original 3Rscan name 
        if sub_scene == 0:
            scan_id = self.metadata[scene]['reference']
        else:
            scan_id = self.metadata[scene]['scans'][sub_scene-1]['reference']
        
        filename = self.data_dir / scan_id / 'semseg.v2.json'
        # read the json file
        instance_frame = np.eye(4)
        with open(filename, 'r') as f:
            data = json.load(f)
            # get centroid where data['segGroups'][i]['ObjectId] == instance_id    
            instance_data = next((item for item in data['segGroups'] if item['objectId'] == instance_id), None)
            centroid = np.array(instance_data['obb']['centroid']) if instance_data else np.zeros(3)
        
        instance_frame[:3, 3] = centroid
        return instance_frame

        
        
    
    def __get_change_metadata(self, scene, sub_scene):
        if sub_scene == 0: 
            # scan is reference scan 
            return set(), set(), set(), np.eye(4)
        else: 
            non_rigid_changes = set(self.metadata[scene]['scans'][sub_scene-1].get('nonrigid', []))
            removed_changes = set(self.metadata[scene]['scans'][sub_scene-1].get('removed', []))
            rigid_objects = self.metadata[scene]['scans'][sub_scene-1].get('rigid', [])
            if self.metadata[scene]['type'] == 'test':
                rigid_changes = set(rigid_objects)
                global_transform = np.eye(4)
            else:
                rigid_changes = {}
                global_transform = np.array(self.metadata[scene]['scans'][sub_scene-1]['transform']).reshape(4, 4).T
                for rigid_obj in rigid_objects:
                    transform = global_transform @ np.array(rigid_obj['transform']).reshape(4, 4).T
                    rigid_changes.update({rigid_obj['instance_reference']: {'transform': transform, 'symmetry': rigid_obj['symmetry']}})
                # rigid_changes = {item['instance_reference'] for item in self.metadata[scene]['scans'][sub_scene-1].get('rigid', [])}
            return non_rigid_changes, rigid_changes, removed_changes, global_transform
    
    def __is_diff(self, matrix, threshold=0.014725531394103791):
        # smallest diff from no transformation for known rigid transforms 0.014725531394103791
        return self.__mag(matrix) > threshold
    
    def __mag(self, matrix):
        return np.linalg.norm(matrix- np.eye(matrix.shape[0]), 'fro')

    
    def __relative_matrix(self, matrix1, matrix2, symmetry=0, to_instance_frame=np.eye(4)):
        # Determine if the matrices are significantly different 
        
        def __solve_relative(matrix1, matrix2):
            # from 1->2
            # point1to2 = np.dot(point1, relative1to2[:3, :3].T) + relative1to2[:3, 3]
            return np.dot(np.linalg.inv(matrix1) , matrix2 )

        def __create_z_rotation_matrix(angle_rad):
            #Create a 3D rotation matrix for rotation around z-axis
            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)
            
            rot_matrix = np.eye(4)
            
            # Apply rotation in the xy plane (around z-axis)
            if rot_matrix.shape[0] >= 3:  # If matrix is at least 3x3
                rot_matrix[0, 0] = cos_theta
                rot_matrix[0, 1] = -sin_theta
                rot_matrix[1, 0] = sin_theta
                rot_matrix[1, 1] = cos_theta
                
            return rot_matrix
        
        # symmetry around the z axis, can have values 0, 1, 2, 3 
        # If no symmetry, just check direct difference
        if symmetry == 0:
            return __solve_relative(matrix1, matrix2)
        
        # C_2 (two-fold) symmetry - check original and 180° rotation
        elif symmetry == 1:
            rot_180 = __create_z_rotation_matrix(np.pi)  # 180 degrees in radians
            
            # Check if the original and rotated matrices are significantly different
            relative = __solve_relative(matrix1, matrix2)
            relative_rotated = rot_180 @ to_instance_frame @ relative
            
            # return the matrix with the smallest norm
            return np.linalg.inv(to_instance_frame) @ relative if self.__mag(relative) < self.__mag(relative_rotated) else relative_rotated
        
        # C_4 (four-fold) symmetry - check original, 90°, 180°, and 270° rotations
        elif symmetry == 2:
            # Define all rotations for C₄ symmetry
            rotations = [
                __create_z_rotation_matrix(angle) 
                for angle in [0, np.pi/2, np.pi, 3*np.pi/2]  # 0°, 90°, 180°, 270°
            ]
            
            relative = __solve_relative(matrix1, matrix2)
            
            # Calculate all relative transformations
            relatives = [rot @ to_instance_frame @ relative for rot in rotations]
            
            # Find and return the transformation with the smallest norm
            norms = [self.__mag(rel) for rel in relatives]
            return np.linalg.inv(to_instance_frame) @ relatives[np.argmin(norms)]
            
        elif symmetry == 3: # c_inf, symmetry is inf rotational symmetry  
            # only consider the relative translation (assume x,y axis have no rotation)
            relative = __solve_relative(matrix1, matrix2)
            translation_only = np.eye(4)
            translation_only[:3, 3] = np.dot(relative[:3, :3] - np.eye(3), np.linalg.inv(to_instance_frame)[:3, 3]) + relative[:3, 3]
            return translation_only
        else: 
            return __solve_relative(matrix1, matrix2)

                 

        
    def __apply_rio_transform(self, points, transformation): 
        # for aligned calibrated with reference points --> original per stage coordinates
        # for original rescan points instance --> reference scan instance
        # for calibrated rescan instance -> reference instance
        return np.dot(points - transformation[:3, 3:].T, transformation[:3, :3])
    
        # from the algined rescan back to original coordiantes, equivalent 
        # global_inv = np.linalg.inv(global_transform)
        # rescan_points = np.dot(instance_points, global_inv[:3, :3].T) + global_inv[:3, 3].T
        # rescan_points = np.dot(instance_points- global_transform[:3, 3].T, global_transform[:3, :3])
        # for rigid change instance from original rescan position to global alignment
        # aligned_points = np.dot(rescan_points - instance_transform[:3, 3].T, instance_transform[:3, :3])
        
        # transform1to2 = np.linalg.inv(global_transform2 @ instance_transform2 ) @ global_transform1 @ instance_transform1 
        # point1to2 = np.dot(instance_points_in1 - transform1to2[:3, 3].T, transform1to2[:3, :3])


if __name__ == "__main__":
    Fire(RScanPreprocessing)
