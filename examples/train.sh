#!/usr/bin/bash

# --- Training ---
python main_instance_segmentation.py \
    general.experiment_name="my_experiment" \
    general.project_name="rescene" \
    general.train_mode=true \
    data/datasets=mix \
    model=rescene \
    backbone=concerto \
    general.freeze="backbone_encoder" \
    general.gpus=8 \
    data.batch_size=4 \
    serialization=mixed \
    loss/contrastive=infoNCE



# --- Validation only (no training) ---
# python main_instance_segmentation.py \
#     general.experiment_name="my_experiment" \
#     general.project_name="rescene" \
#     general.train_mode=false \
#     general.checkpoint="checkpoints/my_checkpoint.ckpt" \
#     data/datasets=mix \
#     model=rescene \
#     backbone=concerto \
#     general.gpus=1 \
#     serialization=mixed

# --- Temporal modifiers (append any of these to training or validation) ---
# serialization=mixed                                   # standard + temporal_overlay serializations
# serialization=st_only                                 # temporal_overlay only
# model.temporal_masking=true                           # mask decoder queries across time
# loss/contrastive=infoNCE                              # contrastive loss between temporal frames
# loss.losses='["labels","masks","changes"]'            # add change detection head
