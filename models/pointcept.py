import torch
import torch.nn as nn
import torch_scatter
from packaging import version
import os 
from addict import Dict
from huggingface_hub import hf_hub_download
import copy

from models.scatter import gem


class PointceptBackbone(nn.Module):
    """
    Shared backbone class for Pointcept models (Sonata and Concerto).
    """
    PLANES = [48, 96, 192, 384, 512, 384, 192, 96, 96]

    def __init__(self, name, repo_id, model_lib=None, custom_config={}, **kwargs):
        super().__init__()
        
        # Determine which library to use
        if model_lib is None:
            raise ValueError("model_lib parameter must be specified (either 'sonata' or 'concerto')")
        
        # Import the appropriate library
        if model_lib == 'sonata':
            import sonata
            self.model_lib = sonata
            self.model_class = sonata.model.PointTransformerV3
        elif model_lib == 'concerto':
            import concerto
            self.model_lib = concerto
            self.model_class = concerto.model.PointTransformerV3
        else:
            raise ValueError(f"Unsupported model_lib: {model_lib}. Must be 'sonata' or 'concerto'")
        
        if name in self.model_lib.model.MODELS:
            print(f"Loading checkpoint from HuggingFace: {name} ...")
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{name}.pth",
                repo_type="model",
                revision="main",
                local_dir=os.path.expanduser(f"~/.cache/{self.model_lib.__name__}/ckpt"),
            )
        elif os.path.isfile(name):
            print(f"Loading checkpoint in local path: {name} ...")
            ckpt_path = name
        else:
            # Try downloading from HuggingFace even if not in MODELS list
            # (e.g., concerto_tiny exists on HF but may not be in the library's MODELS list)
            try:
                print(f"Model {name} not in MODELS list, attempting to load from HuggingFace: {name} ...")
                ckpt_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{name}.pth",
                    repo_type="model",
                    revision="main",
                    local_dir=os.path.expanduser(f"~/.cache/{self.model_lib.__name__}/ckpt"),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Model {name} not found. "
                    f"Available models in library: {self.model_lib.model.MODELS}. "
                    f"Also tried downloading from HuggingFace repo {repo_id} but failed: {str(e)}"
                )

        ckpt = self._load_checkpoint(ckpt_path)
        if custom_config is not None:
            for key, value in custom_config.items():
                ckpt["config"][key] = value

        # update channels based on the checkpoint
        self.PLANES[:5] = ckpt["config"]["enc_channels"]  
        # if decoder channels are provided, use them to update the PLANES
        if "dec_channels" in ckpt["config"]:
            self.PLANES[-4:] = ckpt["config"]["dec_channels"]

        self.model = self.model_class(**ckpt["config"])
        self._load_state_dict(ckpt)
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model params: {n_parameters / 1e6:.2f}M")

        # Read optional behavior flags with safe defaults to preserve previous behavior
        # Read flags; support both new and legacy key names
        serials = kwargs.get("decoder_serializations", ["standard"]) 
        self.decoder_serializations = self._select_decoder_serializations(serials)
        self.parallel_decoder = kwargs.get("parallel_decoder", False)

        if self.parallel_decoder:
            self.feat_merge_type = kwargs.get("feat_merge_type", "mean")

            # Prepare feature merge function and scatter aggregator
            if self.feat_merge_type == "gem":
                p_init = kwargs.get("p_init", 3)
                eps = kwargs.get("eps", 1e-6)
                self.p = nn.Parameter(torch.ones(1)*p_init)
                self.eps = eps
                self.merge_feat = lambda x: gem(x, p=self.p, eps=self.eps)
            else:
                self.merge_feat = lambda x: x.mean(dim=0)  
            
    def _load_checkpoint(self, ckpt_path):
        """Load checkpoint with proper version handling"""
        if version.parse(torch.__version__) >= version.parse("2.4"):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        return ckpt
    
    def _load_state_dict(self, ckpt):
        """Load state dict with handling for missing keys"""
        # Filter out decoder weights if loading encoder-only
        missing_keys, unexpected_keys = self.model.load_state_dict(ckpt["state_dict"], strict=False)
        
        if missing_keys:
            print(f"Missing keys (expected for encoder-only): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    def _select_decoder_serializations(self, serials):
        selected = []

        if "standard" in serials:
            selected.append(self.standard)
        if "temporal_overlay" in serials:
            selected.append(self.temporal_overlay)
        if "temporal_shuffle" in serials:
            selected.append(self.temporal_shuffle)
        if "random" in serials:
            selected.append(self.random_shuffle)
        if "zeros" in serials:
            selected.append(self.zeros_serialization)

        if not selected:
            selected = [self.standard]

        return selected

    def forward(self, x):
        # x is split into samples of individual temporal stages by the voxelizer 
        # pointcept models will treat each stage point cloud as a separate batch sample
        # we join the temporal stages later back with their original batch indices stored in "batch_idx"
        
        # run inference 
        point = self.encoder(x)
        point, aux_feat, coords = self.decoder(point)
        
        return point, aux_feat, coords
    
    def encoder(self, x):
        """
        Run only the encoder part of the model.
        Returns the encoded features for external fusion.
        """
        # Move tensors to the correct device (DDP compatible)
        device = next(self.model.parameters()).device
        for key in x.keys():
            if isinstance(x[key], torch.Tensor):
                x[key] = x[key].to(device, non_blocking=True)
        
        # Run encoder - this will depend on the internal structure of PointTransformerV3
        if len(self.decoder_serializations) == 1 and self.decoder_serializations[0] is self.standard:
            return self.model(x)
        else:
            # modify serializations 
            return self.forward_override(x)
    
    def decoder(self, point): 
        full_res_point = self.format(point)
        
        # decoder outputs the largest resolution first
        aux_feat = [self.format(point)]
        coords = [self.decompose_raw_coords(point)]
        while "unpooling_parent" in point.keys():
            assert "unpooling_parent" in point.keys()
            parent = point.pop("unpooling_parent")
            aux_feat.append(self.format(parent))
            coords.append(self.decompose_raw_coords(parent))
            point = parent
        
        # expected from smaller layers first 
        aux_feat.reverse()
        coords.reverse()
        
        return full_res_point, aux_feat, coords
        
    # return coords of the hierarchical features 
    def decompose_coords(self, point):
        """
        Decomposes coordinates per batch 
        """
        # depends on dimensionality so we cannot simply concatenate the 3D points with temporal stage
        # point.coord will be shape (N, D+1), first column is batch info 
        full_grid_coords = point.coord[:, 1:].int()
        # first three columns will always be X,Y,Z, T is optional 
        full_grid_coords[:, :3] = point.grid_coord
        # get offset indices of the batch splits (skip last one as it is the end of the last batch)
        batch_offset = self.model_lib.utils.batch2offset((point.coord[:,0]).int())[:-1]
        return torch.tensor_split(full_grid_coords, batch_offset.detach().cpu(), dim=0)
    
    def decompose_raw_coords(self, point):
        """
        Decomposes coordinates per batch 
        """
        # get offset indices of the batch splits (skip last one as it is the end of the last batch)
        batch_offset = self.model_lib.utils.batch2offset((point.coord[:,0]).int())[:-1]
        return torch.tensor_split(point.coord[:, 1:], batch_offset.detach().cpu(), dim=0)
    
    def decompose_features(self, point):
        """
        Decomposes features per batch 
        """
        # get offset indices of the batch splits (skip last one as it is the end of the last batch)
        batch_offset = self.model_lib.utils.batch2offset((point.coord[:,0]).int())[:-1]
        return torch.tensor_split(point.feat, batch_offset.detach().cpu(), dim=0)
    
    def get_C(self, point): 
        full_grid_coord = point.coord.int()
        full_grid_coord[:, 1:4] = point.grid_coord
        return full_grid_coord
    
    def get_F(self, point): 
        return point.feat
    
    def get_t(self, point):
        if point.coord.shape[1] == 5: 
            # if the point cloud has temporal stages, return the last column as temporal stage
            return point.coord[:, -1].int()
        else:
            # if the point cloud does not have temporal stages, return a dummy tensor
            return torch.zeros(point.coord.shape[0], dtype=torch.int32, device=point.coord.device)
        
    
    def format(self, point):
        # only store information required for future use 
        point_dict = Dict(
            feat=point.feat,
            coord=point.coord,
            grid_coord=point.grid_coord,
            batch=point.batch,
            F=self.get_F(point), 
            C=self.get_C(point), 
            decomposed_coordinates=self.decompose_coords(point),
            decomposed_features=self.decompose_features(point), 
            t = self.get_t(point),
            true_offsets = self.model_lib.utils.batch2offset((point.coord[:,0]).int())
        )
        if point.pooling_inverse is not None:
            point_dict["pooling_inverse"] = point.pooling_inverse
            point_dict["idx_ptr"] = point.idx_ptr
        point = self.model_lib.structure.Point(point_dict)
        return point
    
    def sparse_from_sample(self, features, sample_sparse_tensor):    
        # set new features but copy structure (coordinates and batch information) from sample
        point_dict = Dict(
            feat=features,
            coord=sample_sparse_tensor.coord,
            grid_coord=sample_sparse_tensor.grid_coord,
            batch=sample_sparse_tensor.batch,
        )
        
        point = self.model_lib.structure.Point(point_dict)

        return self.format(point)
    
    def temporal_pool_merge(self, point, stride=2, reduce="mean"):
        """
        Merge masks using pooling across temporal stages, merging features and coordinates.
        """

        # Get or compute grid coordinates
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[grid_coord] or [coord, grid_size] should be included in the Point"
            )
        
        if "t" in point.keys():
            # if the point cloud has temporal stages, add them to the grid coordinates
            # this is useful for temporal pooling 
            grid_coord = torch.cat([grid_coord, point.t.view(-1, 1)], dim=1)
        else:
            raise AssertionError(
                "[t] should be included in the Point"
            )
        
        # only pool in the temporal dimension
        grid_coord[:, -1] = torch.div(grid_coord[:, -1], stride, rounding_mode="trunc")
        
        # Combine grid coordinates with true batch information (not pointcept batches which split temporal stages)
        batch = point.coord[:, 0].int()
        grid_coord = grid_coord | batch.view(-1, 1) << 48
        
        # Find unique grid coordinates and clustering information
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        
        # Extract grid coordinates (remove batch bits)
        grid_coord = grid_coord & ((1 << 48) - 1)
        
        # Get indices for segment operations
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        # Build pooled point dictionary, only features 
        pooled_features =  torch_scatter.segment_csr(
                point.feat[indices], idx_ptr, reduce=reduce
            )
        
        # unpool the max pooled features back to the original points
        unpooled_features = pooled_features[cluster]

        return self.sparse_from_sample(
            unpooled_features, point
        )
    
    def pooling(self, point, stride=2, reduce="mean"): 
        
        # Get or compute grid coordinates
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[grid_coord] or [coord, grid_size] should be included in the Point"
            )
        
        grid_coord = torch.div(grid_coord, stride, rounding_mode="trunc")
        
        # Combine grid coordinates with batch information
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        
        # Find unique grid coordinates and clustering information
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        
        # Extract grid coordinates (remove batch bits)
        grid_coord = grid_coord & ((1 << 48) - 1)
        
        # Get indices for segment operations
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        
        # Build pooled point dictionary
        point_dict = {
            "feat": torch_scatter.segment_csr(
                point.feat[indices], idx_ptr, reduce=reduce
            ),
            "coord": torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            "grid_coord": grid_coord,
            "batch": point.batch[head_indices],
        }

        # Handle optional attributes
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * stride
            
        # Copy non-spatial attributes directly
        for attr in ["condition", "context", "name", "split"]:
            if attr in point.keys():
                point_dict[attr] = getattr(point, attr)
                
        new_point = self.model_lib.structure.Point(point_dict)
        
        # traceable pooling 
        # new_point.pooling_inverse = cluster
        # new_point.pooling_parent = point
        
        new_point = self.format(new_point)
        
        return new_point

    def forward_override(self, point):
        point = self.model_lib.structure.Point(point)
        point = self.model.embedding(point)

        point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)
        point.sparsify()

        point = self.model.enc(point)
        if not self.model.enc_mode:
            points = []

            if self.parallel_decoder:
                for serialization in self.decoder_serializations:
                    dec_point = self.model.dec(self.change_hierarchical_serialization(point, serialization))
                    points.append(dec_point)
                point = self.merge_features(points)
            else:
                point = self.add_hierarchical_serializations(point) # add serializaztions to all pooling parents 
                point = self.add_serializations(point) # add serializations to this layer 
                point = self.model.dec(point)

        return point

    # change all heirachical serializations to the given serialization
    def change_hierarchical_serialization(self, point, serialization):
        # Recursively modify pooling_parent in place
        if "pooling_parent" in point.keys():
            # Get reference (don't copy!)
            parent = point["pooling_parent"]
            # Recursively modify parent's hierarchy FIRST (bottom-up)
            parent = self.change_hierarchical_serialization(parent, serialization)
            # Apply serialization to parent itself
            parent = serialization(parent)
        
        return point

    # add serializations to each heirachical layer 
    def add_hierarchical_serializations(self, point):
        # Recursively modify pooling_parent IN PLACE
        if "pooling_parent" in point.keys():
            # Get reference (don't copy!)
            parent = point["pooling_parent"]
            
            # Recursively modify parent's hierarchy FIRST (bottom-up)
            parent = self.add_hierarchical_serializations(parent)
            
            # Apply serializations to parent itself
            point["pooling_parent"] = self.add_serializations(parent)
        
        return point 

    # add new serializations to existing point and randomize the order 
    def add_serializations(self, point):
        serialized_order = [] 
        serialized_inverse = []

        # Collect per-serialization orders without relying on shared in-place state
        for serialization in self.decoder_serializations:
            _p = serialization(point)
            serialized_order.append(_p.serialized_order)
            serialized_inverse.append(_p.serialized_inverse)

        point.serialized_order = torch.cat(serialized_order, dim=0)
        point.serialized_inverse = torch.cat(serialized_inverse, dim=0)
        
        if self.model.shuffle_orders:
            perm = torch.randperm(point.serialized_inverse.shape[0])
            point.serialized_order = point.serialized_order[perm]
            point.serialized_inverse = point.serialized_inverse[perm]

        return point

    def merge_features(self, points):
        if not points:
            return None
        
        base_point = points[0]
        # Merge features at current level
        merged_feat = self.merge_feat(torch.stack([p.feat for p in points], dim=0))
        base_point.feat = merged_feat
        
        # Recursively merge unpooling_parent chain
        if "unpooling_parent" in base_point.keys():
            # Collect all unpooling_parents from each point
            parent_points = []
            for p in points:
                if "unpooling_parent" in p.keys():
                    parent_points.append(p["unpooling_parent"])
                else:
                    # If one doesn't have unpooling_parent, they should all not have it
                    return base_point
            
            # Recursively merge parents
            merged_parent = self.merge_features(parent_points)
            # Set merged parent back
            base_point["unpooling_parent"] = merged_parent
        
        return base_point

    # keep original serialization with no changes
    def standard(self, point):
        return point

    def zeros_serialization(self, point):
        point.serialized_order = torch.zeros_like(point.serialized_order)
        return point
    
    def temporal_overlay(self, point): 
        indiv_scene_batches = point.batch
        # Combine grid coordinates with true batch information (not pointcept batches which split temporal stages)
        point.batch = point.coord[:, 0].int()
        point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)
        point.batch = indiv_scene_batches
        return point

    def random_shuffle(self, point): 
        # totally randomize across the whole batch to sanity check 
        rand_idx = torch.randperm(point.serialized_order.shape[1])
        point.serialized_order = point.serialized_order[:, rand_idx]
        point.serialized_inverse = self.serialized_inverse(point.serialized_order)
        return point

    def temporal_shuffle(self, point):
        NotImplementedError("Temporal shuffle not implemented yet")
        return point   

    def serialized_inverse(self, order):
        return torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, order.shape[1], device=order.device).repeat(
                order.shape[0], 1
            ),
        )

    def pool_gt(self, pooled_point, gt_attr, reduce="mean"):
        """
        Args:
            pooled_point: Point object after GridPooling with traceable=True.
                        Must have: pooling_inverse, idx_ptr 
            gt_attr: Ground truth attribute tensor of shape (N_original, ...)
            reduce: Reduction operation - "mean", "max", "min", or "sum".
                Use "max" for categorical labels (majority vote), "mean" for continuous values.
        
        Returns:
            Pooled GT attribute of shape (num_clusters, ...)
        """
        
        # Pool using torch_scatter
        if not isinstance(gt_attr, list):
            # Extract indices and idx_ptr from pooled_point if provided
            idx_ptr = pooled_point.idx_ptr
            # Reconstruct from pooling_inverse (cluster assignment)
            _, indices = torch.sort(pooled_point.pooling_inverse)
            out = torch_scatter.segment_csr(gt_attr[indices], idx_ptr, reduce=reduce)
        else: 
            # Find max number of instances across all batches for padding
            max_instances = max(gt.shape[1] if len(gt.shape) > 1 else 1 for gt in gt_attr)
            
            # Pre-allocate concatenated tensor for efficiency
            total_points = sum(gt.shape[0] for gt in gt_attr)
            device = pooled_point.feat.device
            
            gt_cat = torch.zeros(total_points, max_instances, device=device, dtype=int)
            original_shapes = []
            
            # Fill pre-allocated tensor in place
            row_offset = 0
            for gt in gt_attr:
                num_points = gt.shape[0]
                num_instances = gt.shape[1] if len(gt.shape) > 1 else 1
                original_shapes.append(num_instances)
                # Copy existing data and pad with zeros (which are already there)
                gt_cat[row_offset:row_offset+num_points, :num_instances] = gt
                row_offset += num_points
            
            # Pool all masks at once using global idx_ptr
            _, indices = torch.sort(pooled_point.pooling_inverse)
            pooled_all = torch_scatter.segment_csr(gt_cat[indices], pooled_point.idx_ptr, reduce=reduce)

            
            # Split pooled results by batch and trim to original instance counts
            pooled_batch_info = torch.cat([torch.zeros(1, device=pooled_point.true_offsets.device, dtype=pooled_point.true_offsets.dtype), pooled_point.true_offsets])
            out = []
            for i in range(len(gt_attr)):
                start_idx = int(pooled_batch_info[i].item())
                end_idx = int(pooled_batch_info[i+1].item()) if i+1 < len(pooled_batch_info) else pooled_all.shape[0]
                pooled_batch = pooled_all[start_idx:end_idx]
                # Trim to original number of instances for this batch
                out.append(pooled_batch[:, :original_shapes[i]])
            
        return out





class PointceptBackboneEncOnly(PointceptBackbone):
    """
    Shared encoder-only backbone class for Pointcept models (Sonata and Concerto).
    Uses feature upcasting as the decoder with optional dimensionality reduction.
    """
    PLANES = [48, 96, 192, 384, 512, 512, 384, 192, 96, 96]  # Match regular decoder
    
    def __init__(self, name, repo_id, model_lib=None, decrease_planes=True, custom_config={}, **kwargs):
        # decoder config ignored when encoder mode is present 
        custom_config["enc_mode"] = True
        self.decrease_planes = decrease_planes
        
        super().__init__(name, repo_id, model_lib=model_lib, custom_config=custom_config, **kwargs)

        if self.decrease_planes:
            # Feature adapters for each resolution level to match decoder dimensions
            self.adapters = nn.ModuleList([
                nn.Linear(896, 384),  
                nn.Linear(1088, 192),  
                nn.Linear(1088, 96),   
                nn.Linear(1088, 96),   # Final decoder stage
            ])
    
    def decoder(self, point):
        # use feature upcasting as the decoder
        aux_feat = [self.format(point)]
        
        # store the raw coordinates of the pooled pointclouds split by batch
        coords = [self.decompose_raw_coords(point)]
        
        # map output to original scales 
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            aux_feat.append(self.format(parent))
            coords.append(self.decompose_raw_coords(parent))
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            aux_feat.append(self.format(parent))
            coords.append(self.decompose_raw_coords(parent))
            point = parent
            
        if self.decrease_planes:
            point.feat = self.adapters[-1](point.feat)
            for i in range(1, len(aux_feat)):
                aux_feat[i].feat = self.adapters[i-1](aux_feat[i].feat)
                aux_feat[i] = self.format(aux_feat[i])
            point = self.format(point)

        return point, aux_feat, coords
