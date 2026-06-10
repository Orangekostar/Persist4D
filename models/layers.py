import torch.nn as nn
import torch.nn.functional as F
import functools
import torch

try:
    import flash_attn
except ImportError:
    flash_attn = None

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        batch_first=True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first= batch_first)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        batch_first=True,
        return_updated_features=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_updated_features = return_updated_features

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        # Stage 1: Queries attend to features
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        query_out = tgt + self.dropout(tgt2)
        query_out = self.norm(query_out)

        if self.return_updated_features:
            # Stage 2: Features attend to updated queries (for backward compatibility with Flash version)
            # This aggregates information from queries back to features
            feat2 = self.multihead_attn(
                query=self.with_pos_embed(memory, pos),
                key=self.with_pos_embed(query_out, query_pos),
                value=query_out,
                attn_mask=None,  # No mask for reverse attention
                key_padding_mask=None,
            )[0]
            memory_out = memory + self.dropout(feat2)
            memory_out = self.norm(memory_out)
            return memory_out
        else:
            return query_out

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        # Stage 1: Queries attend to features
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        query_out = tgt + self.dropout(tgt2)

        if self.return_updated_features:
            # Stage 2: Features attend to updated queries (for backward compatibility with Flash version)
            # This aggregates information from queries back to features
            memory2 = self.norm(memory)
            feat2 = self.multihead_attn(
                query=self.with_pos_embed(memory2, pos),
                key=self.with_pos_embed(query_out, query_pos),
                value=query_out,
                attn_mask=None,  # No mask for reverse attention
                key_padding_mask=None,
            )[0]
            memory_out = memory + self.dropout(feat2)
            return memory_out
        else:
            return query_out

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class CrossAttentionLayerFlash(nn.Module):
    """
    Cross-attention layer with flash attention optimization for patched features.
    Adapted from FeatureRefinement in sonata.py.
    All queries attend to patched features.
    """
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        normalize_before=False,
        patch_size=1024,
        enable_flash=True,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.normalize_before = normalize_before
        self.patch_size = patch_size
        self.enable_flash = enable_flash and (flash_attn is not None)
        self.scale = (d_model // nhead) ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize parameters
        for m in [self.q_proj, self.kv_proj, self.proj]:
            nn.init.xavier_uniform_(m.weight)
    
    @torch.no_grad()
    def _offset2bincount(self, offset):
        """Convert offset (cumulative lengths) to bincount (individual lengths)."""
        if len(offset) == 0:
            return torch.tensor([], dtype=torch.long, device=offset.device)
        if len(offset) == 1:
            return offset
        return offset[1:] - offset[:-1]

    @torch.no_grad()
    def _get_padding_and_inverse(self, seq_lengths, device, patch_size=None):
        """
        Get padding, unpadding, and cu_seqlens for features.
        Adapted from FeatureRefinement.get_padding_and_inverse.
        """
        if patch_size is None:
            patch_size = self.patch_size
        
        B = len(seq_lengths)
        # Convert to offset format (cumulative)
        offset = torch.cat([torch.tensor([0], device=device), torch.cumsum(seq_lengths, dim=0)])
        bincount = self._offset2bincount(offset)
        
        bincount_pad = (
            torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc") * patch_size
        )
        # Only pad when num of elements larger than patch_size
        mask_pad = bincount > patch_size
        bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
        
        _offset = nn.functional.pad(offset, (1, 0))
        _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
        pad = torch.arange(_offset_pad[-1], device=device)
        unpad = torch.arange(_offset[-1], device=device)
        cu_seqlens = []
        
        for i in range(B):
            unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
            if bincount[i] != bincount_pad[i]:
                # Repeat last patch_size elements for padding
                pad[
                    _offset_pad[i + 1] - patch_size + (bincount[i] % patch_size) : _offset_pad[i + 1]
                ] = pad[
                    _offset_pad[i + 1] - 2 * patch_size + (bincount[i] % patch_size) : _offset_pad[i + 1] - patch_size
                ]
            pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i],
                    _offset_pad[i + 1],
                    step=patch_size,
                    dtype=torch.int32,
                    device=device,
                )
            )
        
        cu_seqlens_tensor = nn.functional.pad(
            torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
        )
        return pad, unpad, cu_seqlens_tensor, bincount_pad

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        """
        Forward pass: queries (tgt) attend to patched features (memory).
        
        Args:
            tgt: Query tensor (B, Q, C)
            memory: Feature tensor to attend to (B, N, C) - will be patched
            memory_mask: Optional attention mask (ignored for flash attention)
            memory_key_padding_mask: Optional padding mask (ignored for flash attention)
            pos: Positional encoding for memory
            query_pos: Positional encoding for queries
        """
        if not self.enable_flash:
            raise NotImplementedError("Non-flash attention not implemented for CrossAttentionLayerFlash")
        
        B, Q, C = tgt.shape
        N = memory.shape[1]
        
        # Store residual
        residual = memory  # Features are updated, not queries
        
        # Normalize before if needed
        if self.normalize_before:
            memory = self.norm(memory)
        
        # Add positional encodings
        q_input = tgt + query_pos if query_pos is not None else tgt
        kv_input = memory + pos if pos is not None else memory
        
        # Project Q, K, V
        q = self.q_proj(q_input)  # (B, Q, C)
        kv = self.kv_proj(kv_input)  # (B, N, 2*C)
        k, v = kv.chunk(2, dim=-1)  # Each: (B, N, C)
        
        # Patch features (k, v) for flash attention
        seq_lengths = torch.full((B,), N, dtype=torch.long, device=memory.device)
        pad, unpad, cu_seqlens_kv, bincount_pad = self._get_padding_and_inverse(seq_lengths, memory.device)
        
        # Compute offset_pad for batch boundaries
        offset_pad = torch.cat([torch.tensor([0], device=memory.device), torch.cumsum(bincount_pad, dim=0)])
        
        # Reshape queries: (B, Q, C) -> (B*Q, H, D)
        q_flat = q.reshape(B * Q, self.nhead, self.head_dim)  # (B*Q, H, D)
        cu_seqlens_q = torch.arange(0, (B + 1) * Q, Q, dtype=torch.int32, device=memory.device)
        
        # Pad and reshape keys/values per batch
        k_padded_list = []
        v_padded_list = []
        cu_seqlens_k = [0]  # Start with 0 for batch 0
        
        for b in range(B):
            n_pad_b = bincount_pad[b].item()  # Use the computed padded size
            
            # Ensure n_pad_b is valid
            if n_pad_b <= 0:
                n_pad_b = N  # Fallback to original size
            n_pad_b = min(n_pad_b, N * 2)  # Reasonable upper bound
            
            # Get the padded indices for this batch using the offset boundaries
            # pad[offset_pad[b]:offset_pad[b+1]] contains indices relative to batch b's start
            pad_start = offset_pad[b].item()
            pad_end = offset_pad[b + 1].item()
            pad_b = pad[pad_start:pad_end]  # These are already relative to batch start after adjustment
            
            # Ensure pad_b contains valid indices (0 to N-1)
            pad_b = torch.clamp(pad_b, 0, N - 1)
            
            # Ensure pad_b has exactly n_pad_b elements
            if len(pad_b) < n_pad_b:
                # Repeat the last indices to pad
                num_repeat = n_pad_b - len(pad_b)
                if num_repeat > 0 and len(pad_b) > 0:
                    pad_b = torch.cat([pad_b, pad_b[-num_repeat:]])
                elif len(pad_b) == 0:
                    # If pad_b is empty, create repeated indices
                    pad_b = torch.zeros(n_pad_b, dtype=torch.long, device=memory.device)
            elif len(pad_b) > n_pad_b:
                # Truncate if somehow too long
                pad_b = pad_b[:n_pad_b]
            
            # Pad k, v for this batch
            k_b = k[b, pad_b, :]  # (len(pad_b), C)
            v_b = v[b, pad_b, :]  # (len(pad_b), C)
            
            # Use actual length for reshaping
            actual_len = k_b.shape[0]
            
            # Verify C matches expected dimensions
            if C != self.nhead * self.head_dim:
                raise ValueError(
                    f"Dimension mismatch: C={C}, nhead={self.nhead}, head_dim={self.head_dim}, "
                    f"expected C={self.nhead * self.head_dim}"
                )
            
            # Reshape to (actual_len, H, D) for flash attention
            # Use view instead of reshape for better error messages
            k_b_flat = k_b.view(actual_len, self.nhead, self.head_dim)
            v_b_flat = v_b.view(actual_len, self.nhead, self.head_dim)
            
            k_padded_list.append(k_b_flat)
            v_padded_list.append(v_b_flat)
            
            # Track cumulative length for cu_seqlens_k (one entry per batch, not per patch)
            # cu_seqlens_k[b+1] = cumulative length up to and including batch b
            cu_seqlens_k.append(cu_seqlens_k[-1] + actual_len)
        
        # Concatenate all padded k, v
        k_padded = torch.cat(k_padded_list, dim=0)  # (sum(N_pad), H, D)
        v_padded = torch.cat(v_padded_list, dim=0)  # (sum(N_pad), H, D)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=memory.device)  # Shape: (B+1,)
        
        # Compute max_seqlen_k from actual batch lengths
        max_seqlen_k = max([k_b.shape[0] for k_b in k_padded_list]) if k_padded_list else self.patch_size
        
        # Flash attention: all queries attend to patched features
        attn_out = flash_attn.flash_attn_varlen_func(
            q_flat.half(),
            k_padded.half(),
            v_padded.half(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=Q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
        ).to(q.dtype)  # (B*Q, H, D)
        
        # Reshape back to (B, Q, C)
        query_out = attn_out.reshape(B, Q, C)
        
        # Apply output projection
        query_out = self.proj(query_out)  # (B, Q, C)
        
        # Update features by having them attend to updated queries
        # This aggregates information from queries back to features
        q_features = memory + pos if pos is not None else memory
        kv_queries = query_out + query_pos if query_pos is not None else query_out
        
        # Project for reverse attention: features attend to query outputs
        q_f = self.q_proj(q_features)  # (B, N, C)
        kv_f = self.kv_proj(kv_queries)  # (B, Q, 2*C)
        k_f, v_f = kv_f.chunk(2, dim=-1)  # Each: (B, Q, C)
        
        # Reshape for flash attention (no patching needed, Q is small)
        q_f_flat = q_f.reshape(B * N, self.nhead, self.head_dim)  # (B*N, H, D)
        k_f_flat = k_f.reshape(B * Q, self.nhead, self.head_dim)  # (B*Q, H, D)
        v_f_flat = v_f.reshape(B * Q, self.nhead, self.head_dim)  # (B*Q, H, D)
        
        # Create cu_seqlens
        cu_seqlens_q_f = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=memory.device)
        cu_seqlens_k_f = torch.arange(0, (B + 1) * Q, Q, dtype=torch.int32, device=memory.device)
        
        # Flash attention: features attend to query outputs
        feat_out = flash_attn.flash_attn_varlen_func(
            q_f_flat.half(),
            k_f_flat.half(),
            v_f_flat.half(),
            cu_seqlens_q=cu_seqlens_q_f,
            cu_seqlens_k=cu_seqlens_k_f,
            max_seqlen_q=N,
            max_seqlen_k=Q,
            dropout_p=0.0,  # No dropout on aggregation pass
            softmax_scale=self.scale,
        ).to(q_f.dtype)  # (B*N, H, D)
        
        feat_out = feat_out.reshape(B, N, C)  # (B, N, C)
        
        # Apply output projection
        out = self.proj(feat_out)
        
        # Add residual and normalize
        if not self.normalize_before:
            out = self.norm(residual + out)
        else:
            out = residual + out
        
        return out


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, dropout=0.0, activation="relu"):
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        super().__init__()
        if hidden_features is None:
            hidden_features = [out_features]
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # first layer
        self.layers.append(nn.Linear(in_features, hidden_features[0]))
        self.norms.append(norm_fn(hidden_features[0]))
        
        # Hidden layers: hidden_features[i] -> hidden_features[i+1]
        for i in range(len(hidden_features) - 1):
            self.layers.append(nn.Linear(hidden_features[i], hidden_features[i+1]))
            self.norms.append(norm_fn(hidden_features[i+1]))
        
        # Final layer: last hidden -> out_features
        self.final_layer = nn.Linear(hidden_features[-1], out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self._reset_parameters()

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            
        x = self.final_layer(x)
        return x

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)