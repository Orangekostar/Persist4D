import torch
import torch.nn as nn

from models.position_embedding import PositionEmbeddingCoordsSine, PositionalEncoding3D
from models.layers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from models.scatter import AdaptiveScatter
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch.amp import autocast


class ReScene(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        use_np_features,
        voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal,
        D, 
        num_changes, 
        temporal_masking,
        use_changes_loss,
        save_segment_info,
        return_query_features=False,
    ):
        super().__init__()

        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type
        self.D = D
        self.num_changes = num_changes
        self.temporal_masking = temporal_masking
        self.save_segment_info = bool(save_segment_info)
        self.return_query_features = bool(return_query_features)

        self.backbone = config.backbone
        self.num_levels = len(self.hlevels)
        sizes = self.backbone.PLANES[-5:]

        self.mask_features_head = nn.Linear(
            in_features=self.backbone.PLANES[-1], 
            out_features=self.mask_dim,         
            bias=True
        )

        self.scatter_fn = AdaptiveScatter(scatter_type=self.scatter_type, feat_dim=self.mask_dim, p=3, eps=1e-6)

        assert (
            not use_np_features
        ) or non_parametric_queries, "np features only with np queries"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2 * self.mask_dim,
                hidden_dims=[2 * self.mask_dim],
                output_dim=2 * self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)
        
        # Only initialize change_embed_head if changes loss is enabled
        # This prevents unused parameters in DDP when changes loss is disabled
        if use_changes_loss:
            self.change_embed_head = nn.Linear(hidden_dim, self.num_changes)
        else:
            self.change_embed_head = None

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=self.mask_dim,
                gauss_scale=self.gauss_scale,
                normalize=self.normalize_pos_enc,
                d_in=self.D
            )
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=self.mask_dim,
                normalize=self.normalize_pos_enc,
                d_in=self.D,
            )
        else:
            assert False, "pos enc type not known"

        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_squeeze_attention.append(
                    nn.Linear(sizes[hlevel], self.mask_dim)
                )

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)
        self.decoder_norm = nn.LayerNorm(hidden_dim)


    def initialize_queries(self, pcd_features, coords):
        sampled_coords = None
        batch_size = len(coords)

        if self.non_parametric_queries:
            fps_idx = [
                furthest_point_sample(
                    pcd_features.decomposed_coordinates[i][None, ...].float(),
                    self.num_queries,
                )
                .squeeze(0)
                .long()
                for i in range(len(pcd_features.decomposed_coordinates))
            ]

            sampled_coords = torch.stack(
                [
                    coords[-1][i][fps_idx[i].long(), :]
                    for i in range(len(fps_idx))
                ]
            )

            mins = torch.stack(
                [
                    coords[-1][i].min(dim=0)[0]
                    for i in range(len(coords[-1]))
                ]
            )
            maxs = torch.stack(
                [
                    coords[-1][i].max(dim=0)[0]
                    for i in range(len(coords[-1]))
                ]
            )

            query_pos = self.pos_enc(
                sampled_coords.float(), input_range=[mins, maxs]
            )  # Batch, Dim, queries
            query_pos = self.query_projection(query_pos)
            # Convert from (Batch, Dim, queries) to (Batch, queries, Dim)
            query_pos = query_pos.permute((0, 2, 1))

            if not self.use_np_features:
                queries = torch.zeros_like(query_pos)  # query_pos is already (batch, queries, features)
            else:
                queries = torch.stack(
                    [
                        pcd_features.decomposed_features[i][fps_idx[i].long(), :]
                        for i in range(len(fps_idx))
                    ]
                )
                queries = self.np_feature_projection(queries)

        elif self.random_queries:
            query_pos = (
                torch.rand(
                    batch_size,
                    self.num_queries,  # Swap order: queries first
                    self.mask_dim,     # features second
                    device=pcd_features.device,
                )
                - 0.5
            )

            queries = torch.zeros_like(query_pos)

        elif self.random_query_both:
            if not self.random_normal:
                query_pos_feat = (
                    torch.rand(
                        batch_size,
                        self.num_queries,  # queries first
                        2 * self.mask_dim, # features second
                        device=pcd_features.device,
                    )
                    - 0.5
                )
            else:
                query_pos_feat = torch.randn(
                    batch_size,
                    self.num_queries,  # queries first
                    2 * self.mask_dim, # features second
                    device=pcd_features.device,
                )

            queries = query_pos_feat[:, :, : self.mask_dim]  # (batch, num_queries, features)
            # (batch, num_queries, features)
            query_pos = query_pos_feat[:, :, self.mask_dim :]  
        else:
            # PARAMETRIC QUERIES
            queries = self.query_feat.weight.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            # (batch, num_queries, features)
            query_pos = self.query_pos.weight.unsqueeze(0).repeat(
                batch_size, 1, 1
            )

        return queries, query_pos, sampled_coords

    def sample_and_batch_features(self, decomposed_feat, curr_sample_max=None, is_eval=False, extra = []):
        # stack and sample features 
        device = decomposed_feat[0].device

        curr_sample_size = max(
            [pcd.shape[0] for pcd in decomposed_feat]
        )

        if min([pcd.shape[0] for pcd in decomposed_feat]) == 1:
            raise RuntimeError(
                "only a single point gives nans in cross-attention"
            )

        if not (self.max_sample_size or is_eval) and curr_sample_max is not None:
            curr_sample_size = min(
                curr_sample_size, curr_sample_max
            )

        rand_idx = []
        mask_idx = []
        for k in range(len(decomposed_feat)):
            pcd_size = decomposed_feat[k].shape[0]
            if pcd_size <= curr_sample_size:
                # we do not need to sample
                # take all points and pad the rest with zeroes and mask it
                idx = torch.zeros(
                    curr_sample_size,
                    dtype=torch.long,
                    device=device,
                )

                midx = torch.ones(
                    curr_sample_size,
                    dtype=torch.bool,
                    device=device,
                )

                idx[:pcd_size] = torch.arange(
                    pcd_size, device=device
                )

                midx[:pcd_size] = False  # attend to first points
            else:
                # we have more points in pcd as we like to sample
                # take a subset (no padding or masking needed)
                idx = torch.randperm(
                    decomposed_feat[k].shape[0], device=device
                )[:curr_sample_size]
                midx = torch.zeros(
                    curr_sample_size,
                    dtype=torch.bool,
                    device=device,
                )  # attend to all

            rand_idx.append(idx)
            mask_idx.append(midx)

        batched_feat = torch.stack(
            [
                decomposed_feat[k][rand_idx[k], :]
                for k in range(len(rand_idx))
            ]
        )

        # batch all other decomposed items if any are provided
        batched_outputs = []
        for decomposed_item in extra:
            batched = torch.stack(
                [
                    decomposed_item[k][rand_idx[k], :]
                    for k in range(len(rand_idx))
                ]
            )
            batched_outputs.append(batched)

        m = torch.stack(mask_idx)

        return (batched_feat, *batched_outputs, m)

    def unstack_batched(self, batched_feat, batch_mapping):
        return [f[~m] for f, m in zip(batched_feat.unbind(0), batch_mapping.unbind(0))]


    def get_pos_encs(self, coords):
        # coords: list of raw coordinates at each hierarchical level per batch sample 
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i]:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast('cuda', enabled=False):
                    tmp = self.pos_enc(
                        coords_batch[None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False
    ):  
        if not self.train_on_segments:
            point2segment=None # ensure no point2segment is used

        x.raw_coordinates = raw_coordinates
        x.point2segment = point2segment
        
        # backbone and positional encodings
        pcd_features, aux, coords = self.backbone(x)
        pos_encodings_pcd = self.get_pos_encs(coords)
        
        # mask feature head and aggregation to segments if needed
        agg_feat, agg_coords = self.aggregate_features(pcd_features, point2segment)
        batched_features, batch_map = self.sample_and_batch_features(agg_feat)

        # query initialization
        queries, query_pos, sampled_coords = self.initialize_queries(
            pcd_features=pcd_features,
            coords=coords
        )

        predictions_class = []
        predictions_changes = []
        predictions_mask = []
        segment_features = [agg_feat]

        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                output_class, output_change, output_logits = self.mask_module(queries, batched_features)
                output_mask = self.unstack_batched(output_logits, batch_map)

                # list of attn masks per batch sample
                attn_masks = self.attn_mask(
                    output_logits, 
                    batch_map, 
                    sparse_coords=pcd_features, # associate features with original coordinates for pooling 
                    num_pooling_steps=len(aux) - hlevel - 1, 
                    point2segment=point2segment # map to full size if trained on segments
                )

                batched_aux, batched_attn, batched_pos_enc, padding_mask = self.sample_and_batch_features(
                    aux[hlevel].decomposed_features,
                    self.sample_sizes[hlevel],
                    is_eval,
                    extra = [attn_masks, pos_encodings_pcd[hlevel][0]] # also batched attn and pos enc
                )

                # reset queries that attended to all positions
                batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == batched_attn.shape[1]] = False
                attn_mask = batched_attn.repeat_interleave(
                        self.num_heads, dim=0
                    ).permute((0, 2, 1))  # Shape: (num_heads*batch, seq_len, num_queries) -> (num_heads*batch, num_queries, seq_len)

                # linear layer 
                src_pcd = self.lin_squeeze[decoder_counter][i](batched_aux)
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output = self.cross_attention[decoder_counter][i](
                    queries,
                    src_pcd,
                    memory_mask=attn_mask,
                    memory_key_padding_mask=padding_mask,  # Use dedicated padding mask for better performance
                    pos=batched_pos_enc,
                    query_pos=query_pos,
                )

                output = self.self_attention[decoder_counter][i](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ) 

                predictions_class.append(output_class)
                predictions_changes.append(output_change)
                predictions_mask.append(output_mask)

        # final predictions
        output_class, output_change, output_logits = self.mask_module(queries, batched_features)
        output_mask = self.unstack_batched(output_logits, batch_map)

        predictions_class.append(output_class)
        predictions_changes.append(output_change)
        predictions_mask.append(output_mask)

        # Build output dictionary
        output_dict = {
            "pred_logits": predictions_class[-1],
            "pred_changes": predictions_changes[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class, predictions_mask, predictions_changes
            ),
            "sampled_coords": sampled_coords.detach().cpu().numpy()
            if sampled_coords is not None
            else None,
            "backbone_features": pcd_features,
            "segment_features": segment_features,
        }

        if self.return_query_features:
            output_dict["query_features"] = self.decoder_norm(queries)

        # Optionally export segment-level GT info + temporal stage aggregated to segments.
        if self.save_segment_info:
            targets = x.gt_targets
            seg_gt_classes = []
            seg_gt_instance_ids = []
            seg_temporal_stages = []

            for b in range(len(point2segment)):
                p2s = point2segment[b]

                device = p2s.device
                p2s = p2s.to(device=device, dtype=torch.long)

                t = targets[b]
                ts = t["temporal_stages"].to(device=device)
                gm = t["masks"].to(device=device)
                gl = t["labels"].to(device=device).to(torch.long)
                gids = t["ids"].to(device=device).to(torch.long)

                seg_temporal_stages.append(self.scatter_fn.max_scatter(ts, p2s))

                gm_bool = gm > 0.5
                has_inst = gm_bool.any(dim=0)
                inst_idx = gm_bool.to(torch.float32).argmax(dim=0).to(torch.long)

                point_cls = gl[inst_idx]
                point_inst_idx = inst_idx

                seg_cls = self.scatter_fn.mode_scatter(
                    point_cls, p2s, num_categories=int(self.num_classes), ignore_index=-1
                )
                seg_inst_idx = self.scatter_fn.mode_scatter(
                    point_inst_idx, p2s, num_categories=int(gm.shape[0]), ignore_index=-1
                )

                seg_gt_classes.append(seg_cls)
                seg_gt_instance_ids.append(gids[seg_inst_idx])

            output_dict["segment_gt_classes"] = seg_gt_classes
            output_dict["segment_gt_instance_ids"] = seg_gt_instance_ids
            output_dict["segment_temporal_stages"] = seg_temporal_stages

        return output_dict

    def aggregate_features(self, pcd_features, point2segment=None):
        # mask feature head 
        mask_features = self.mask_features_head(pcd_features.F)
        mask_features = self.backbone.sparse_from_sample(mask_features, pcd_features)
        
        # accumulate mask features per segment if needed 
        if self.train_on_segments:
            mask_segments = []
            segment_coordinates = []
            for i, (mask_feature, mask_coords) in enumerate(zip(mask_features.decomposed_features, mask_features.decomposed_coordinates)):
                mask_segments.append(self.scatter_fn(mask_feature, point2segment[i]))
                segment_coordinates.append(self.scatter_fn.mean_scatter(mask_coords, point2segment[i]))
            
            return mask_segments, segment_coordinates
        else: 
            return mask_features.decomposed_features, mask_features.decomposed_coordinates
        
    def mask_module(self, query_feat, features):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
        
        # Only compute change predictions if change_embed_head is initialized
        if self.change_embed_head is not None:
            outputs_change = self.change_embed_head(query_feat)
        else:
            outputs_change = None

        # [batch_size, num_points, num_queries]
        output_logits = torch.bmm(features, mask_embed.transpose(-1, -2))

        return outputs_class, outputs_change, output_logits

    def attn_mask(self, logits, mask=None, sparse_coords=None, num_pooling_steps=0, point2segment=None, threshold=0.5):
        
        # if no batch mapping is provided return the mask as is 
        if mask is None:
            return logits.detach().sigmoid() < threshold

        if point2segment is not None:
            # Apply point2segment mapping for each batch item
            output_masks = []
            for i in range(len(point2segment)):
                output_masks.append(logits[i][~mask[i]][point2segment[i]])
            output_masks = torch.cat(output_masks)

        else: 
            output_masks = logits[~mask]

        if sparse_coords is not None:
            attn_mask = self.backbone.sparse_from_sample(output_masks, sparse_coords)
            # pool attn mask to current hierarchical level 
            for _ in range(num_pooling_steps):
                attn_mask = self.backbone.pooling(attn_mask)

            # pool across time if configured
            if self.temporal_masking: 
                attn_mask = self.backbone.temporal_pool_merge(attn_mask)

            attn_mask = self.backbone.sparse_from_sample((attn_mask.F.detach().sigmoid() < threshold), attn_mask).decomposed_features
        else:
            attn_mask = output_masks.detach().sigmoid() < threshold

        return attn_mask
    
    def space_n_time_m(self, n, m):
        return n if self.D == 3 else [n, n, n, m]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_change):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b, "pred_changes": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_change[:-1])
        ]

