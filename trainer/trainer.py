import gc
import hashlib
import json
from pathlib import Path
import statistics
from torch_scatter import scatter_mean
from collections import defaultdict
from collections.abc import Mapping
from sklearn.cluster import DBSCAN
import warnings

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import pickle


_SINGLE_POINT_CROSS_ATTENTION_ERROR = (
    "only a single point gives nans in cross-attention"
)


def _p2_general_flag(config, name):
    general = getattr(config, "general", None)
    return bool(getattr(general, name, False))


def _safe_length(value):
    if value is None:
        return 0
    try:
        return len(value)
    except (AttributeError, TypeError):
        return 0


def _point_cloud_is_empty(data):
    return any(
        _safe_length(getattr(data, component, None)) == 0
        for component in ("features", "coordinates")
    )


def _batch_preflight_failure(
    data,
    target,
    max_batch_size=None,
    require_labels=True,
):
    if not isinstance(target, (list, tuple)):
        return "invalid target list"
    if _safe_length(target) == 0:
        return "empty target list"
    for target_idx, target_item in enumerate(target):
        if not isinstance(target_item, Mapping):
            return f"target[{target_idx}] must be a mapping"
        required_fields = ["point2segment"]
        if require_labels:
            required_fields.insert(0, "labels")
        for field in required_fields:
            if field not in target_item:
                return f"target[{target_idx}] missing required field '{field}'"
            value = target_item[field]
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                return (
                    f"target[{target_idx}] field '{field}' must be a "
                    "non-scalar tensor"
                )
    if _point_cloud_is_empty(data):
        return "empty point cloud"

    feature_count = _safe_length(getattr(data, "features", None))
    if max_batch_size is not None and feature_count > max_batch_size:
        return f"batch exceeds max_batch_size ({feature_count} > {max_batch_size})"
    return None


def _output_tensor_failure(field, tensor, expected_device):
    if expected_device is not None and tensor.device != torch.device(
        expected_device
    ):
        return (
            f"forward output field '{field}' must be on device "
            f"{torch.device(expected_device)} (got {tensor.device})"
        )
    try:
        finite = bool(torch.isfinite(tensor).all().item())
    except Exception as error:
        return (
            f"forward output field '{field}' finite check failed: "
            f"{type(error).__name__}: {error}"
        )
    if not finite:
        return f"forward output field '{field}' contains non-finite values"
    return None


def _prediction_layer_failure(
    output,
    expected_batch_size,
    expected_device,
    field_prefix="",
    expect_pred_changes=None,
):
    if not isinstance(output, Mapping):
        field = field_prefix.removesuffix(".")
        return f"forward output field '{field}' must be a mapping"

    for field in ("pred_logits", "pred_masks"):
        if field not in output:
            return (
                "forward output missing required field "
                f"'{field_prefix}{field}'"
            )

    pred_logits_field = f"{field_prefix}pred_logits"
    pred_logits = output["pred_logits"]
    if not isinstance(pred_logits, torch.Tensor) or pred_logits.numel() == 0:
        return (
            f"forward output field '{pred_logits_field}' must be a "
            "non-empty tensor"
        )
    if pred_logits.ndim < 3 or pred_logits.shape[0] != expected_batch_size:
        return (
            f"forward output field '{pred_logits_field}' must have one "
            "batched entry per target"
        )
    tensor_failure = _output_tensor_failure(
        pred_logits_field,
        pred_logits,
        expected_device,
    )
    if tensor_failure is not None:
        return tensor_failure

    pred_masks_field = f"{field_prefix}pred_masks"
    pred_masks = output["pred_masks"]
    if isinstance(pred_masks, torch.Tensor):
        valid_masks = (
            pred_masks.numel() > 0
            and pred_masks.ndim >= 3
            and pred_masks.shape[0] == expected_batch_size
        )
    elif isinstance(pred_masks, (list, tuple)):
        valid_masks = len(pred_masks) == expected_batch_size and all(
            isinstance(mask, torch.Tensor)
            and mask.numel() > 0
            and mask.ndim >= 2
            for mask in pred_masks
        )
    else:
        valid_masks = False
    if not valid_masks:
        return (
            f"forward output field '{pred_masks_field}' must contain "
            "non-empty tensors"
        )
    mask_tensors = (
        [(pred_masks_field, pred_masks)]
        if isinstance(pred_masks, torch.Tensor)
        else [
            (f"{pred_masks_field}[{mask_idx}]", mask)
            for mask_idx, mask in enumerate(pred_masks)
        ]
    )
    for field, mask in mask_tensors:
        tensor_failure = _output_tensor_failure(
            field,
            mask,
            expected_device,
        )
        if tensor_failure is not None:
            return tensor_failure

    if expect_pred_changes is None:
        return None

    pred_changes_field = f"{field_prefix}pred_changes"
    if "pred_changes" not in output:
        return (
            "forward output missing required field "
            f"'{pred_changes_field}'"
        )
    pred_changes = output["pred_changes"]
    if not expect_pred_changes:
        if pred_changes is not None:
            return (
                f"forward output field '{pred_changes_field}' must be None "
                "when change objective is disabled"
            )
        return None
    if not isinstance(pred_changes, torch.Tensor) or pred_changes.numel() == 0:
        return (
            f"forward output field '{pred_changes_field}' must be a "
            "non-empty tensor"
        )
    if pred_changes.ndim < 3 or pred_changes.shape[0] != expected_batch_size:
        return (
            f"forward output field '{pred_changes_field}' must have one "
            "batched entry per target"
        )
    return _output_tensor_failure(
        pred_changes_field,
        pred_changes,
        expected_device,
    )


def _expected_aux_output_count(module):
    model = getattr(module, "model", None)
    try:
        count = int(model.num_levels) * int(model.num_decoders)
    except (AttributeError, TypeError, ValueError):
        return None
    return count if count > 0 else None


def _criterion_output_requirements(module):
    criterion = getattr(module, "criterion", None)
    losses = getattr(criterion, "losses", None)
    expect_pred_changes = (
        "changes" in losses if losses is not None else None
    )
    require_segment_features = bool(
        getattr(criterion, "use_contrastive_loss", False)
    )
    return expect_pred_changes, require_segment_features


def _forward_output_failure(
    output,
    expected_batch_size,
    expected_device=None,
    expected_aux_outputs=None,
    expect_pred_changes=None,
    require_segment_features=False,
):
    if not isinstance(output, Mapping):
        return "forward output must be a mapping"
    prediction_failure = _prediction_layer_failure(
        output,
        expected_batch_size,
        expected_device,
        expect_pred_changes=expect_pred_changes,
    )
    if prediction_failure is not None:
        return prediction_failure

    aux_outputs = output.get("aux_outputs")
    if aux_outputs is None:
        if expected_aux_outputs is not None:
            return "forward output missing required field 'aux_outputs'"
    else:
        if not isinstance(aux_outputs, (list, tuple)):
            return "forward output field 'aux_outputs' must be a sequence"
        if not aux_outputs and expected_aux_outputs is not None:
            return (
                "forward output field 'aux_outputs' must be a non-empty "
                "sequence"
            )
        if (
            expected_aux_outputs is not None
            and len(aux_outputs) != expected_aux_outputs
        ):
            return (
                "forward output field 'aux_outputs' must contain "
                f"{expected_aux_outputs} entries (got {len(aux_outputs)})"
            )
        for aux_idx, aux_output in enumerate(aux_outputs):
            prediction_failure = _prediction_layer_failure(
                aux_output,
                expected_batch_size,
                expected_device,
                field_prefix=f"aux_outputs[{aux_idx}].",
                expect_pred_changes=expect_pred_changes,
            )
            if prediction_failure is not None:
                return prediction_failure

    if not require_segment_features:
        return None
    if "segment_features" not in output:
        return "forward output missing required field 'segment_features'"
    segment_features = output["segment_features"]
    if not isinstance(segment_features, (list, tuple)) or not segment_features:
        return (
            "forward output field 'segment_features' must be a non-empty "
            "sequence"
        )
    if len(segment_features) != 1:
        return (
            "forward output field 'segment_features' must contain 1 layer "
            f"(got {len(segment_features)})"
        )
    for layer_idx, layer_features in enumerate(segment_features):
        field = f"segment_features[{layer_idx}]"
        if (
            not isinstance(layer_features, (list, tuple))
            or len(layer_features) != expected_batch_size
        ):
            return (
                f"forward output field '{field}' must contain one tensor "
                "per target"
            )
        for batch_item_idx, features in enumerate(layer_features):
            tensor_field = f"{field}[{batch_item_idx}]"
            if not isinstance(features, torch.Tensor) or features.numel() == 0:
                return (
                    f"forward output field '{tensor_field}' must be a "
                    "non-empty tensor"
                )
            tensor_failure = _output_tensor_failure(
                tensor_field,
                features,
                expected_device,
            )
            if tensor_failure is not None:
                return tensor_failure
    return None


def _batch_collective_device(module, data):
    module_device = getattr(module, "device", None)
    if module_device is not None:
        return module_device
    for component in ("features", "coordinates"):
        value = getattr(data, component, None)
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _fallback_collective_device(module):
    try:
        module_device = getattr(module, "device", None)
        if module_device is not None:
            return module_device
    except Exception:
        pass

    try:
        parameter = next(module.parameters())
        return parameter.device
    except Exception:
        pass

    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if distributed:
        try:
            if str(torch.distributed.get_backend()).lower() == "nccl":
                return torch.device("cuda", torch.cuda.current_device())
        except Exception:
            pass
    return torch.device("cpu")


def _p2_batch_input(module, batch, stage):
    data = None
    target = None
    file_names = "<unavailable>"
    collective_device = _fallback_collective_device(module)
    failure = None
    cause = None
    try:
        data, target, file_names = batch
        collective_device = _batch_collective_device(module, data)
        max_batch_size = (
            module.config.general.max_batch_size
            if stage == "train"
            else None
        )
        failure = _batch_preflight_failure(
            data,
            target,
            max_batch_size=max_batch_size,
            require_labels=stage != "test",
        )
    except Exception as error:
        failure = _phase_exception_reason("input", error)
        cause = error
    return (
        data,
        target,
        file_names,
        collective_device,
        failure,
        cause,
    )


def _safe_repr(value):
    try:
        return repr(value)
    except Exception as error:
        return f"<unrepresentable {type(error).__name__}>"


def _format_batch_contract_failures(failures):
    details = "; ".join(
        f"rank={failure['rank']}, stage={failure['stage']}, "
        f"batch_idx={failure['batch_idx']}, "
        f"file_names={failure['file_names']}, reason={failure['reason']}"
        for failure in failures
    )
    return f"Batch contract violation: {details}"


def _batch_contract_consensus(
    stage,
    batch_idx,
    file_names,
    reason=None,
    cause=None,
    device=None,
):
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    rank = torch.distributed.get_rank() if distributed else 0
    local_failure = (
        {
            "rank": rank,
            "stage": stage,
            "batch_idx": batch_idx,
            "file_names": _safe_repr(file_names),
            "reason": reason,
        }
        if reason is not None
        else None
    )

    if distributed:
        failure_flag = torch.tensor(
            int(local_failure is not None),
            dtype=torch.int32,
            device=device,
        )
        torch.distributed.all_reduce(
            failure_flag,
            op=torch.distributed.ReduceOp.MAX,
        )
        if failure_flag.item() == 0:
            return

        gathered_failures = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_failures, local_failure)
        failures = [failure for failure in gathered_failures if failure is not None]
    else:
        if local_failure is None:
            return
        failures = [local_failure]

    error = RuntimeError(_format_batch_contract_failures(failures))
    if cause is None:
        raise error
    raise error from cause


def _phase_exception_reason(phase, error):
    if (
        phase == "forward"
        and isinstance(error, RuntimeError)
        and str(error) == _SINGLE_POINT_CROSS_ATTENTION_ERROR
    ):
        return _SINGLE_POINT_CROSS_ATTENTION_ERROR
    return f"{phase} {type(error).__name__}: {error}"


def _validate_objective_value(value, description):
    tensor = (
        value.detach()
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value)
    )
    if tensor.numel() == 0:
        raise ValueError(f"empty {description}")
    if tensor.ndim != 0:
        raise ValueError(f"non-scalar {description}")
    if not bool(torch.isfinite(tensor).item()):
        raise ValueError(f"non-finite {description}")


def _validate_objective_finite(losses, total_loss, loss_names):
    if not loss_names:
        raise ValueError("no objective loss terms")
    for loss_name in loss_names:
        _validate_objective_value(
            losses[loss_name],
            f"raw objective term '{loss_name}'",
        )
    _validate_objective_value(total_loss, "aggregate objective")


def _optimizer_gradient_failure(optimizer):
    for group_idx, param_group in enumerate(optimizer.param_groups):
        for parameter_idx, parameter in enumerate(param_group["params"]):
            gradient = parameter.grad
            if gradient is None:
                continue
            gradient_values = (
                gradient.coalesce().values()
                if gradient.is_sparse
                else gradient
            )
            if not torch.isfinite(gradient_values).all():
                return (
                    "non-finite gradient at optimizer "
                    f"param_group={group_idx}, parameter={parameter_idx}"
                )
    return None


def aggregate_objective_loss(losses, weight_dict, validate_finite=True):
    """Build the optimized objective while omitting per-layer diagnostics."""
    objective_items = [
        (loss_name, loss)
        for loss_name, loss in losses.items()
        if not (loss_name.startswith("loss_") and "_contrastive_layer" in loss_name)
    ]
    if not objective_items:
        raise ValueError("no objective loss terms")
    total_loss = sum(
        loss * weight_dict.get(loss_name, 1.0)
        for loss_name, loss in objective_items
    )
    if validate_finite:
        _validate_objective_finite(
            losses,
            total_loss,
            [loss_name for loss_name, _ in objective_items],
        )
    return total_loss


def _configured_objective_loss(module, losses):
    p2_fail_closed_runtime = _p2_general_flag(
        module.config,
        "p2_fail_closed_runtime",
    )
    if _p2_general_flag(module.config, "p2_weighted_objective"):
        return aggregate_objective_loss(
            losses,
            module.criterion.weight_dict,
            validate_finite=p2_fail_closed_runtime,
        )

    total_loss = sum(losses.values())
    if p2_fail_closed_runtime:
        _validate_objective_finite(losses, total_loss, list(losses))
    return total_loss



class InstanceSegmentation(pl.LightningModule):
    _TRAIN_SAMPLER_CHECKPOINT_KEY = "p2_train_sampler_generator"
    _TRAIN_SAMPLER_CHECKPOINT_SCHEMA_VERSION = 1
    _TRAIN_SAMPLER_RESUME_SCOPE = "completed_epoch_boundary_only"
    _OPTIMIZER_PARAMETER_CONTRACT_KEY = "p2_optimizer_parameter_contract"
    _OPTIMIZER_PARAMETER_CONTRACT_SCHEMA_VERSION = 1

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._pending_train_sampler_generator_state = None
        self.automatic_optimization = True
        self._initialize_model()
        self._setup_training()
        self.save_hyperparameters(config)


    def _initialize_model(self):
        """Initialize model components and settings"""
        torch.set_float32_matmul_precision(self.config.general.matmul_precision)
        self.decoder_id = self.config.general.decoder_id
        self.mask_type = "segment_mask" if self.config.model.train_on_segments else "masks"
        self.eval_on_segments = self.config.general.eval_on_segments
        self.model = hydra.utils.instantiate(self.config.model)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Model compilation for performance (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.config.general.get('compile_model', False):
            print("🔧 Compiling model with torch.compile for performance optimization")
            self.model = torch.compile(self.model)
        
        # Freeze backbone parameters if specified - do this before DDP wrapping
        if self.config.general.freeze is not None:
            self._freeze_backbone_parameters()
        
        # Only watch model if logger is wandb and we're on rank 0
        # This prevents distributed training issues with W&B
        if (hasattr(self, 'logger') and 
            self.logger is not None and 
            hasattr(self.logger, '__class__') and
            self.logger.__class__.__name__ == "WandbLogger" and
            hasattr(self.trainer, 'global_rank') and 
            self.trainer.global_rank == 0):
            self.logger.watch(self.model, log="all", log_freq=100)


    def _setup_training(self):
        """Setup training components"""
        self.ignore_label = self.config.data.ignore_label
        self.criterion = self._setup_matcher_and_loss(self.config)
        self.instance_metric = hydra.utils.instantiate(self.config.instance_metric)
        self.aux_metric = (
            hydra.utils.instantiate(self.config.aux_metric)
            if hasattr(self.config, "aux_metric") and self.config.aux_metric is not None
            else None
        )

        
        # Suppress the Lightning sync_dist warning since we use TorchMetrics
        warnings.filterwarnings(
            "ignore", 
            message=".*sync_dist=True.*when logging on epoch level in distributed setting.*",
            category=UserWarning,
            module="pytorch_lightning"
        )
        
        if self.config.general.postprocessing_export:
        # Initialize tracking containers
            self.preds = {}

    def _setup_matcher_and_loss(self, config):
        """Setup matcher and loss components"""
        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
        }
        
        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            # weight ignored masked to zero 
            weight = 0.0 if i in config.general.ignore_mask_idx else 1.0
            aux_weight_dict.update({k + f"_{i}": v * weight for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        
        criterion = hydra.utils.instantiate(
            config.loss,
            matcher=matcher,
            weight_dict=weight_dict,
        )
        p2_fail_closed_runtime = _p2_general_flag(
            config,
            "p2_fail_closed_runtime",
        )
        criterion.p2_fail_closed_runtime = p2_fail_closed_runtime
        if hasattr(criterion, "contrastive_loss"):
            criterion.contrastive_loss.p2_fail_closed_runtime = (
                p2_fail_closed_runtime
            )
        return criterion

    def forward(self, x, point2segment=None, raw_coordinates=None, is_eval=False, targets=None):

        # Set device attribute for models that expect it (minkowski)
        x.device = self.device

        # Optional: attach GT targets for models that want to export segment-level GT info
        if targets is not None and getattr(self.config.general, "save_segment_info", False):
            x.gt_targets = targets
        
        return self.model(x, point2segment, raw_coordinates=raw_coordinates, is_eval=is_eval)

    

    def training_step(self, batch, batch_idx):
        """Training step implementation"""
        p2_fail_closed_runtime = _p2_general_flag(
            self.config,
            "p2_fail_closed_runtime",
        )
        if p2_fail_closed_runtime:
            (
                data,
                target,
                file_names,
                collective_device,
                preflight_failure,
                preflight_cause,
            ) = _p2_batch_input(
                self,
                batch,
                stage="train",
            )
            _batch_contract_consensus(
                stage="train",
                batch_idx=batch_idx,
                file_names=file_names,
                reason=preflight_failure,
                cause=preflight_cause,
                device=collective_device,
            )
            self._p2_optimizer_context = {
                "batch_idx": batch_idx,
                "file_names": file_names,
                "device": collective_device,
            }
        else:
            data, target, file_names = batch
            if data.features.shape[0] > self.config.general.max_batch_size:
                print("data exceeds threshold")
                raise RuntimeError("BATCH TOO BIG")
            if target == []:
                return None

        # Forward pass
        if p2_fail_closed_runtime:
            forward_failure = None
            forward_cause = None
            try:
                output = self.forward(
                    data,
                    point2segment=[
                        target[i]["point2segment"] for i in range(len(target))
                    ],
                    raw_coordinates=self._process_raw_coordinates(data),
                    targets=target,
                )
                (
                    expect_pred_changes,
                    require_segment_features,
                ) = _criterion_output_requirements(self)
                forward_failure = _forward_output_failure(
                    output,
                    expected_batch_size=len(target),
                    expected_device=collective_device,
                    expected_aux_outputs=_expected_aux_output_count(self),
                    expect_pred_changes=expect_pred_changes,
                    require_segment_features=require_segment_features,
                )
            except Exception as error:
                forward_failure = _phase_exception_reason("forward", error)
                forward_cause = error

            _batch_contract_consensus(
                stage="train",
                batch_idx=batch_idx,
                file_names=file_names,
                reason=forward_failure,
                cause=forward_cause,
                device=collective_device,
            )
        else:
            try:
                output = self.forward(
                    data,
                    point2segment=[
                        target[i]["point2segment"] for i in range(len(target))
                    ],
                    raw_coordinates=self._process_raw_coordinates(data),
                    targets=target,
                )
            except RuntimeError as run_err:
                if run_err.args[0] == _SINGLE_POINT_CROSS_ATTENTION_ERROR:
                    return None
                raise run_err

        # Compute losses
        if p2_fail_closed_runtime:
            objective_failure = None
            objective_cause = None
            try:
                losses = self.criterion(output, target, mask_type=self.mask_type)
            except Exception as error:
                objective_failure = _phase_exception_reason("criterion", error)
                objective_cause = error

            if objective_failure is None:
                try:
                    total_loss = _configured_objective_loss(self, losses)
                    log_values = {
                        **{f"train_{k}": v for k, v in losses.items()},
                        "train_loss": total_loss,
                        **self._get_mean_loss(losses, "train"),
                    }
                    log_batch_size = data.batch_size
                except Exception as error:
                    objective_failure = _phase_exception_reason("objective", error)
                    objective_cause = error

            _batch_contract_consensus(
                stage="train",
                batch_idx=batch_idx,
                file_names=file_names,
                reason=objective_failure,
                cause=objective_cause,
                device=collective_device,
            )
        else:
            try:
                losses = self.criterion(output, target, mask_type=self.mask_type)
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                raise val_err
            total_loss = _configured_objective_loss(self, losses)
            log_values = {
                **{f"train_{k}": v for k, v in losses.items()},
                "train_loss": total_loss,
                **self._get_mean_loss(losses, "train"),
            }
            log_batch_size = data.batch_size

        self.log_dict(
            log_values,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=log_batch_size,
        )

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val", batch_idx=batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test", batch_idx=batch_idx)
    
    
    def _get_mean_loss(self, losses: dict, prefix: str) -> dict:
        """Calculate mean of final and auxiliary losses grouped by type."""
        mean_losses = {}
        for loss_type in ["ce", "mask", "dice"]:
            # Filter and calculate mean for each loss type
            relevant_losses = [v for k, v in losses.items() if loss_type in k]
            if relevant_losses:
                mean_losses[f"{prefix}_mean_loss_{loss_type}"] = torch.stack(relevant_losses).mean()
        return mean_losses
    

    def export(self, pred_masks, scores, pred_classes, file_names, decoder_id):
        if hasattr(self.trainer, 'global_rank') and self.trainer.global_rank != 0:
            return

        root_path = f"eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}/decoder_{decoder_id}"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = file_names
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > self.config.general.export_threshold:
                    # reduce the export size a bit. I guess no performance difference
                    np.savetxt(
                        f"{pred_mask_path}/{file_name}_{real_id}.txt",
                        mask,
                        fmt="%d",
                    )
                    fout.write(
                        f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                    )

    def postprocessing_export(self, base_path):
        if hasattr(self.trainer, 'global_rank') and self.trainer.global_rank != 0:
            return
        with open(str( base_path / 'predictions.pkl'), 'wb') as f:
            pickle.dump(self.preds, f)
    
    def on_validation_epoch_end(self):

        ap_results = self.instance_metric.compute()
        if self.aux_metric is not None:
            ap_results.update(self.aux_metric.compute())
            self.aux_metric.reset()
        
        # Postprocessing export only on single GPU evaluation
        if self.config.general.postprocessing_export and self.trainer.global_rank == 0:
            # Use save_dir instead of eval_output for unified output directory
            base_path = Path(self.config.general.save_dir)
            base_path.mkdir(parents=True, exist_ok=True)
            print(f"Exported to {base_path}")
            self.postprocessing_export(base_path)
            
            # Write CSV file with AP results
            from benchmark.export_ap_csv import write_ap_csv
            write_ap_csv(
                ap_results,
                header=[head.label for head in self.instance_metric.heads],
                label_list=self.instance_metric.CLASS_LABELS,
                log_prefix=self.instance_metric.log_prefix,
                csv_path=base_path / "ap_results.csv",
            )

            # Clean up predictions self.preds only ever exists if exporting for postprocessing
            if hasattr(self, 'preds'):
                del self.preds
                gc.collect()
                self.preds = dict()

        self.log_dict({**ap_results}, sync_dist=False) # sync handled by torchmetric compute, warning suppressed
        
        self.instance_metric.reset()
      

    def _eval_step(self, batch, stage, batch_idx=None):
        """Unified evaluation step for validation and testing"""
        p2_fail_closed_runtime = _p2_general_flag(
            self.config,
            "p2_fail_closed_runtime",
        )
        if p2_fail_closed_runtime:
            (
                data,
                target,
                file_names,
                collective_device,
                preflight_failure,
                preflight_cause,
            ) = _p2_batch_input(self, batch, stage=stage)
            eval_data = {}
            if preflight_failure is None:
                for attribute in (
                    "inverse_maps",
                    "target_full",
                    "original_colors",
                    "idx",
                    "original_normals",
                    "original_coordinates",
                ):
                    try:
                        eval_data[attribute] = getattr(data, attribute)
                    except AttributeError as error:
                        preflight_failure = (
                            f"missing eval data attribute '{attribute}'"
                        )
                        preflight_cause = error
                        break
                    except Exception as error:
                        preflight_failure = _phase_exception_reason(
                            "evaluation metadata",
                            error,
                        )
                        preflight_cause = error
                        break
            _batch_contract_consensus(
                stage=stage,
                batch_idx=batch_idx,
                file_names=file_names,
                reason=preflight_failure,
                cause=preflight_cause,
                device=collective_device,
            )
            inverse_maps = eval_data["inverse_maps"]
            target_full = eval_data["target_full"]
            original_colors = eval_data["original_colors"]
            data_idx = eval_data["idx"]
            original_normals = eval_data["original_normals"]
            original_coordinates = eval_data["original_coordinates"]
        else:
            data, target, file_names = batch
            # save values from data (rewritten)
            inverse_maps = data.inverse_maps
            target_full = data.target_full
            original_colors = data.original_colors
            data_idx = data.idx
            original_normals = data.original_normals
            original_coordinates = data.original_coordinates

            if len(data.coordinates) == 0:
                return 0.0

        # Disable gradient computation during evaluation
        with torch.no_grad():
            if p2_fail_closed_runtime:
                forward_failure = None
                forward_cause = None
                try:
                    raw_coordinates = self._process_raw_coordinates(data)
                    output = self.forward(
                        data,
                        point2segment=[
                            target[i]["point2segment"]
                            for i in range(len(target))
                        ],
                        raw_coordinates=raw_coordinates,
                        is_eval=True,
                        targets=target,
                    )
                    (
                        expect_pred_changes,
                        require_segment_features,
                    ) = _criterion_output_requirements(self)
                    forward_failure = _forward_output_failure(
                        output,
                        expected_batch_size=len(target),
                        expected_device=collective_device,
                        expected_aux_outputs=_expected_aux_output_count(self),
                        expect_pred_changes=expect_pred_changes,
                        require_segment_features=require_segment_features,
                    )
                except Exception as error:
                    forward_failure = _phase_exception_reason("forward", error)
                    forward_cause = error

                _batch_contract_consensus(
                    stage=stage,
                    batch_idx=batch_idx,
                    file_names=file_names,
                    reason=forward_failure,
                    cause=forward_cause,
                    device=collective_device,
                )
            else:
                raw_coordinates = self._process_raw_coordinates(data)
                try:
                    output = self.forward(
                        data,
                        point2segment=[
                            target[i]["point2segment"]
                            for i in range(len(target))
                        ],
                        raw_coordinates=raw_coordinates,
                        is_eval=True,
                        targets=target,
                    )
                except RuntimeError as run_err:
                    if run_err.args[0] == _SINGLE_POINT_CROSS_ATTENTION_ERROR:
                        return None
                    raise run_err

            # Process predictions for metrics
            if p2_fail_closed_runtime:
                evaluation_failure = None
                evaluation_cause = None
                try:
                    with torch.amp.autocast("cuda", enabled=False):
                        predictions = self._process_predictions(
                            output=output,
                            target_low_res=target,
                            target_full_res=target_full,
                            inverse_maps=inverse_maps,
                            file_names=file_names,
                            full_res_coords=original_coordinates,
                            original_colors=original_colors,
                            original_normals=original_normals,
                            raw_coords=(
                                raw_coordinates
                                if self.config.general.use_dbscan
                                else None
                            ),
                            idx=data_idx,
                        )
                        self.instance_metric.update(predictions, target_full)
                        if self.aux_metric is not None:
                            self.aux_metric.update(predictions, target_full)
                except Exception as error:
                    evaluation_failure = _phase_exception_reason(
                        "evaluation",
                        error,
                    )
                    evaluation_cause = error

                _batch_contract_consensus(
                    stage=stage,
                    batch_idx=batch_idx,
                    file_names=file_names,
                    reason=evaluation_failure,
                    cause=evaluation_cause,
                    device=collective_device,
                )
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    predictions = self._process_predictions(
                        output=output,
                        target_low_res=target,
                        target_full_res=target_full,
                        inverse_maps=inverse_maps,
                        file_names=file_names,
                        full_res_coords=original_coordinates,
                        original_colors=original_colors,
                        original_normals=original_normals,
                        raw_coords=(
                            raw_coordinates
                            if self.config.general.use_dbscan
                            else None
                        ),
                        idx=data_idx,
                    )
                    self.instance_metric.update(predictions, target_full)
                    if self.aux_metric is not None:
                        self.aux_metric.update(predictions, target_full)

            # Clear intermediate tensors to free memory
            del predictions

            # Calculate losses if not in test mode
            if stage == "test":
                return 0.0
            if p2_fail_closed_runtime:
                objective_failure = None
                objective_cause = None
                try:
                    losses = self.criterion(
                        output,
                        target,
                        mask_type=self.mask_type,
                    )
                except Exception as error:
                    objective_failure = _phase_exception_reason(
                        "criterion",
                        error,
                    )
                    objective_cause = error

                if objective_failure is None:
                    try:
                        total_loss = _configured_objective_loss(self, losses)
                        log_values = {
                            **{f"{stage}_{k}": v for k, v in losses.items()},
                            **self._get_mean_loss(losses, stage),
                        }
                        if _p2_general_flag(
                            self.config,
                            "p2_weighted_objective",
                        ):
                            log_values[f"{stage}_loss"] = total_loss
                        log_batch_size = data.batch_size
                    except Exception as error:
                        objective_failure = _phase_exception_reason(
                            "objective",
                            error,
                        )
                        objective_cause = error

                _batch_contract_consensus(
                    stage=stage,
                    batch_idx=batch_idx,
                    file_names=file_names,
                    reason=objective_failure,
                    cause=objective_cause,
                    device=collective_device,
                )
            else:
                losses = self.criterion(
                    output,
                    target,
                    mask_type=self.mask_type,
                )
                total_loss = _configured_objective_loss(self, losses)
                log_values = {
                    **{f"{stage}_{k}": v for k, v in losses.items()},
                    **self._get_mean_loss(losses, stage),
                }
                if _p2_general_flag(
                    self.config,
                    "p2_weighted_objective",
                ):
                    log_values[f"{stage}_loss"] = total_loss
                log_batch_size = data.batch_size
            self.log_dict(
                log_values,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=log_batch_size,
            )
            
            return total_loss
    
    def _process_raw_coordinates(self, data):
        """Process raw coordinates"""
        raw_coordinates = None
        dim = self.config.model.D
        
        # if sonata data structure raw coordinates are stored in the last D columns of coords
        if hasattr(data, "coord") and data.coord is not None:
            return data.coord[:, -dim:]  # no change to features 
        
        elif self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -dim:]  
            data.features = data.features[:, :-dim]     
        return raw_coordinates
        
    def _calculate_eval_losses(self, output, target):
        """Calculate evaluation losses with error handling"""
        if self.config.trainer.deterministic:
            torch.use_deterministic_algorithms(False)
            
        try:
            losses = self.criterion(output, target, mask_type=self.mask_type)
            # Filter losses based on weight dictionary
            losses = {k: v * self.criterion.weight_dict[k] 
                     for k, v in losses.items() 
                     if k in self.criterion.weight_dict}
        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            return None
        finally:
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)
                
        return losses
    
    def _get_full_res_mask(self, mask, inverse_map, point2segment_full, is_heatmap=False):
        """Convert mask to full resolution"""
        mask = mask.detach().cpu()[inverse_map.detach().cpu()]  # full res

        if self.eval_on_segments and not is_heatmap:
            mask = scatter_mean(mask, point2segment_full.detach().cpu(), dim=0)  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[point2segment_full.detach().cpu()]  # full res points

        return mask
    
    def _process_predictions(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
    ):
        """Process instance evaluation for a batch"""
        # Get predictions from model output
        prediction = self._get_predictions(output)
        needs_features = self.config.general.save_visualizations or self.config.general.save_features
        backbone_features = output["backbone_features"].F.detach().cpu().numpy() if needs_features else None
        pca_feat = pca_features(backbone_features) if self.config.general.save_visualizations else None
        
        # Optional: per-segment features produced by some models (e.g. `models/rescene.py`)
        # Format: list[layer] -> list[batch] -> Tensor[num_segments, feat_dim]
        segment_features_out = output.get("segment_features", None)

        # Process each item in batch
        offset_coords_idx = 0
        batch_preds = []
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            # Get masks for current batch item
            masks = self._get_batch_masks(prediction, bid, target_low_res)

            # Process masks using DBSCAN if configured
            if self.config.general.use_dbscan:
                new_preds = self._process_dbscan(
                    masks, prediction, bid, raw_coords, offset_coords_idx
                )
                offset_coords_idx += masks.shape[0]
                scores, low_res_masks, classes, heatmap = self._get_mask_and_scores(
                        torch.stack(new_preds["pred_logits"]).detach().cpu(),
                        torch.stack(new_preds["pred_masks"]).T,
                        len(new_preds["pred_logits"]),
                        self.model.num_classes - 1,
                )
            else: # process normally if not using dbscan postprocessing 
                scores, low_res_masks, classes, heatmap = self._get_mask_and_scores(
                        prediction[self.decoder_id]["pred_logits"][bid].detach().cpu(),
                        masks,
                        prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                        self.model.num_classes - 1,
                    )


            # Get full resolution masks
            masks = self._get_full_res_mask(low_res_masks, inverse_maps[bid], 
                                            target_full_res[bid]["point2segment"]).numpy()
            heatmap = self._get_full_res_mask(heatmap, inverse_maps[bid], 
                                            target_full_res[bid]["point2segment"],
                                            is_heatmap=True).numpy()


            # Filter and sort predictions by scores and overlap if configured 
            classes, masks, scores, heatmap = self._filter_and_sort_predictions(
                masks, scores, classes, heatmap
            )

            # remap labels for 200 datasets
            if "200" in self.validation_dataset.dataset_name:
                classes[classes == 0] = -1
                if self.config.data.test_mode != "test":
                    target_full_res[bid]["labels"][target_full_res[bid]["labels"] == 0] = -1

            label_offset = self.validation_dataset.label_offset
            classes = self.validation_dataset._remap_model_output(classes.detach().cpu() + label_offset)

            if (self.config.data.test_mode != "test" and len(target_full_res) != 0):
                target_full_res[bid]["labels"] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["labels"].detach().cpu() + label_offset
                )
                
            
            masks = torch.from_numpy(masks)
            scores = torch.from_numpy(scores)
            
            # compute bounding boxes
            bboxs = self._compute_bounding_boxes(masks, torch.from_numpy(full_res_coords[bid]))

            # format  predictions
            pred_data = {
                "pred_masks": masks,
                "pred_scores": scores,
                "pred_classes": classes,
                "pred_boxes": bboxs,
            }
            batch_preds.append(pred_data)
            
            #update target to include bounding boxes 
            target_full_res[bid]["boxes"] = self._compute_bounding_boxes(target_full_res[bid]["masks"].T, torch.from_numpy(full_res_coords[bid]))
            target_full_res[bid]["file_name"] = file_names[bid]
            
            # save visualization per sample in batch 
            if self.config.general.save_visualizations:
                # determine full resolution pca features for visualization
                sample_features = self._get_full_res_mask(
                    torch.from_numpy(pca_feat),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True).numpy()
                
                self._handle_visualizations(file_names[bid], target_full_res[bid], full_res_coords[bid], 
                                        masks, classes, original_colors[bid], original_normals[bid],
                                        sample_features, idx[bid])
            
            # handle exports
            if self.config.general.export:
                file_name = file_names[bid]
                self.export(
                    pred_data["pred_masks"],
                    pred_data["pred_scores"],
                    pred_data["pred_classes"],
                    file_name,
                    self.decoder_id,
                    )

            if self.config.general.save_features:
                # determine avg feature per mask if required 
                features = self._get_avg_feature_per_mask(backbone_features,
                                                        low_res_masks)
                pred_data["features"] = features 
                
                # Optionally store per-segment info (features + optional GT aggregations)
                if getattr(self.config.general, "save_segment_info", False) and segment_features_out is not None:
                    last_layer = segment_features_out[-1] if len(segment_features_out) > 0 else None
                    if isinstance(last_layer, (list, tuple)) and len(last_layer) > bid:
                        seg = last_layer[bid]
                        pred_data["segment_features"] = (
                            seg.detach().cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
                        )

                    # If the model produced segment-level metadata, persist it (no recomputation here).
                    v = output.get("segment_gt_classes", None)
                    if isinstance(v, (list, tuple)) and len(v) > bid and v[bid] is not None:
                        pred_data["segment_gt_classes"] = (
                            v[bid].detach().cpu().numpy() if torch.is_tensor(v[bid]) else np.asarray(v[bid])
                        )

                    v = output.get("segment_gt_instance_ids", None)
                    if isinstance(v, (list, tuple)) and len(v) > bid and v[bid] is not None:
                        pred_data["segment_gt_instance_ids"] = (
                            v[bid].detach().cpu().numpy() if torch.is_tensor(v[bid]) else np.asarray(v[bid])
                        )

                    v = output.get("segment_temporal_stages", None)
                    if isinstance(v, (list, tuple)) and len(v) > bid and v[bid] is not None:
                        pred_data["segment_temporal_stages"] = (
                            v[bid].detach().cpu().numpy() if torch.is_tensor(v[bid]) else np.asarray(v[bid])
                        )
                
            if self.config.general.postprocessing_export:
                self.preds[file_names[bid]] = pred_data
        
        return batch_preds
            

    def _get_predictions(self, output):
        """Extract and process predictions from model output"""
        prediction = output["aux_outputs"].copy()
        prediction.append({
            "pred_logits": output["pred_logits"],
            "pred_masks": output["pred_masks"],
        })
        
        # Process logits
        prediction[self.decoder_id]["pred_logits"] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1
        )[..., :-1]
        
        return prediction
    
    def _get_batch_masks(self, prediction, bid, target_low_res):
        """Get masks for current batch item"""

        if self.model.train_on_segments:
            return prediction[self.decoder_id]["pred_masks"][bid].detach().cpu()[
                target_low_res[bid]["point2segment"].detach().cpu()
            ]
        return prediction[self.decoder_id]["pred_masks"][bid].detach().cpu()
    
    def _process_dbscan(self, masks, prediction, bid, raw_coords, offset_coords_idx):
        """Process masks using DBSCAN clustering"""
        new_preds = {
            "pred_masks": [],
            "pred_logits": [],
        }
        
        curr_coords_idx = masks.shape[0]
        curr_coords = raw_coords[offset_coords_idx:curr_coords_idx + offset_coords_idx]
        
        for curr_query in range(masks.shape[1]):
            curr_masks = masks[:, curr_query] > 0
            
            if curr_coords[curr_masks].shape[0] > 0:
                clusters = DBSCAN(
                    eps=self.config.general.dbscan_eps,
                    min_samples=self.config.general.dbscan_min_points,
                    n_jobs=-1
                ).fit(curr_coords[curr_masks]).labels_
                
                new_mask = torch.zeros(curr_masks.shape, dtype=int)
                new_mask[curr_masks] = torch.from_numpy(clusters) + 1
                
                for cluster_id in np.unique(clusters):
                    if cluster_id != -1:
                        new_preds["pred_masks"].append(
                            masks[:, curr_query] * (new_mask == cluster_id + 1)
                        )
                        new_preds["pred_logits"].append(
                            prediction[self.decoder_id]["pred_logits"][bid, curr_query]
                        )
                        
        return new_preds
    
    def _get_mask_and_scores(
        self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None
    ):
        if device is None:
            device = self.device
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        # Get top k predictions
        if self.config.general.topk_per_image != -1:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                self.config.general.topk_per_image, sorted=True
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap
    
    
    def _filter_and_sort_predictions(self, masks, scores, classes, heatmap):
        """
        Filter and sort instance predictions based on scores and overlap.
        
        Args:
            masks (torch.Tensor): Instance masks [N_points, N_instances]
            scores (torch.Tensor): Confidence scores [N_instances]
            classes (torch.Tensor): Class predictions [N_instances]
            heatmap (torch.Tensor): Prediction heatmaps [N_points, N_instances]
            
        Returns:
            tuple: (filtered_classes, filtered_masks, filtered_scores, filtered_heatmap)
        """
        
        # Sort predictions by confidence score
        sort_scores, sort_indices = scores.sort(descending=True)
        sort_scores_values = sort_scores.detach().cpu().numpy()
        sort_scores_index = sort_indices.detach().cpu().numpy()
        
        # Sort all arrays according to scores
        sort_classes = classes[sort_scores_index]
        sorted_masks = masks[:, sort_scores_index]
        sorted_heatmap = heatmap[:, sort_scores_index]
        
        if not self.config.general.filter_out_instances:
            return sort_classes, sorted_masks, sort_scores_values, sorted_heatmap
            
        # Calculate pairwise IoU matrix
        pairwise_overlap = sorted_masks.T @ sorted_masks  # [N_instances, N_instances]
        normalization = pairwise_overlap.max(axis=0)
        norm_overlaps = pairwise_overlap / normalization
        
        # Filter instances based on score threshold and overlap
        #TODO: inclusive instance ID logic is not correct for 3RScan individual rescans
        keep_instances = set()
        for instance_id in range(norm_overlaps.shape[0]):
            instance_score = sort_scores_values[instance_id]
            instance_mask = sorted_masks[:, instance_id]
            
            # Skip if score is below threshold or mask is empty
            if instance_score < self.config.general.scores_threshold:
                continue
            if instance_mask.sum() == 0.0:
                continue
                
            # Find overlapping instances
            overlap_ids = set(np.nonzero(
                norm_overlaps[instance_id, :] > self.config.general.iou_threshold
            )[0])
            
            # Keep instance if no overlaps or it's the first instance in overlapping set
            if not overlap_ids or instance_id == min(overlap_ids):
                keep_instances.add(instance_id)
        
        # Sort and apply filtering
        keep_instances = sorted(list(keep_instances))
        return (
            sort_classes[keep_instances],
            sorted_masks[:, keep_instances],
            sort_scores_values[keep_instances],
            sorted_heatmap[:, keep_instances]
        )
    
    
    def _compute_bounding_boxes(self, masks, coords):
        """Process prediction boxes for each instance"""
        #TODO: make this robust to temporal instances
        bbox_data = torch.empty((masks.shape[1], coords.shape[1]*2), dtype=torch.float32)
        for id in range(masks.shape[1]):
            obj_coords = coords.to(self.device)[masks[:, id].bool(), :]
            if obj_coords.shape[0] == 0:
                bbox_data[id] = torch.zeros(coords.shape[1]*2, dtype=torch.float32, device=self.device)
            else:
                obj_center = obj_coords.mean(dim=0)
                obj_axis_length = obj_coords.max(dim=0)[0] - obj_coords.min(dim=0)[0]
                bbox_data[id] = torch.cat((obj_center, obj_axis_length))

        return bbox_data

    def _handle_visualizations(self, file_name, target_full_res, full_res_coords, 
                         pred_masks, pred_classes, original_colors, original_normals, 
                         backbone_features, bid_idx):
        """Handle visualization saving"""       

        if hasattr(self.trainer, 'global_rank') and self.trainer.global_rank != 0:
            return     
        if "cond_inner" in self.test_dataset.data[bid_idx]:
            inner_mask = self.test_dataset.data[bid_idx]["cond_inner"]
            target_full_res["masks"] = target_full_res["masks"][:, inner_mask]
            
            # Preserve temporal_stages if present
            if "temporal_stages" in target_full_res and target_full_res["temporal_stages"] is not None:
                if isinstance(target_full_res["temporal_stages"], torch.Tensor):
                    target_full_res["temporal_stages"] = target_full_res["temporal_stages"][inner_mask]
                else:
                    target_full_res["temporal_stages"] = target_full_res["temporal_stages"][inner_mask]

            full_res_coords =full_res_coords[inner_mask]
            original_colors = original_colors[inner_mask]
            original_normals = original_normals[inner_mask]
            backbone_features = backbone_features[inner_mask]

        # Check if model is Sonata (RGB values need to be multiplied by 255)
        is_sonata = False
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            if hasattr(backbone, 'model_lib'):
                is_sonata = backbone.model_lib == 'sonata' or (hasattr(backbone.model_lib, '__name__') and backbone.model_lib.__name__ == 'sonata')
        
        # Use automatic visualization routing (handles temporal detection internally)
        save_visualizations_auto(
            target_full_res,
            full_res_coords,
            pred_masks,
            pred_classes,
            original_colors,
            original_normals,
            self.validation_dataset.map2color,
            f"{self.config['general']['save_dir']}/visualizations",
            file_name=file_name,
            point_size=self.config.general.visualization_point_size,
            backbone_features=backbone_features,
            is_sonata=is_sonata,
        )
    
    def _get_avg_feature_per_mask(self, features, masks):
        """Get average feature per mask"""
        
        normalized_masks = masks / (masks.sum(axis=0, keepdims=True) + 1e-10)
        features_per_mask = np.einsum('ij,ik->jk', normalized_masks, features)

        return features_per_mask

    def on_test_epoch_end(self):
        if not self.config.general.export:
            self.on_validation_epoch_end()

    def configure_optimizers(self):
        parameters = self.parameters()
        if _p2_general_flag(self.config, "p2_fail_closed_runtime"):
            parameters = (
                parameter for parameter in parameters if parameter.requires_grad
            )
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=parameters
        )
        
        # Configure scheduler with proper Lightning handling
        scheduler_config = self.config.scheduler.scheduler.copy()
        if "total_steps" in scheduler_config.keys() and scheduler_config["total_steps"] == -1:
            # Lightning will automatically set total_steps
            scheduler_config["total_steps"] = self.trainer.estimated_stepping_batches
        
        lr_scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)
        
        # Return in proper Lightning format
        scheduler_dict = {
            "scheduler": lr_scheduler,
            **self.config.scheduler.pytorch_lightning_params
        }
        
        return [optimizer], [scheduler_dict]

    def _train_sampler_generator(self):
        train_dataset = getattr(self, "train_dataset", None)
        sampler = getattr(train_dataset, "sampler", None)
        return getattr(sampler, "generator", None)

    def _restore_train_sampler_generator_state(self):
        state = getattr(self, "_pending_train_sampler_generator_state", None)
        if state is None or not hasattr(self, "train_dataset"):
            return

        generator = self._train_sampler_generator()
        enabled = (
            _p2_general_flag(self.config, "p2_fail_closed_runtime")
            or generator is not None
        )
        if not enabled:
            self._pending_train_sampler_generator_state = None
            return
        if generator is None:
            raise RuntimeError(
                "cannot restore train sampler generator state: sampler has no "
                "explicit generator"
            )

        generator.set_state(state)
        self._pending_train_sampler_generator_state = None

    def _optimizer_parameter_contract(self, checkpoint):
        trainer = getattr(self, "_trainer", None)
        optimizers = getattr(trainer, "optimizers", None)
        optimizer_states = checkpoint.get("optimizer_states")
        state_dict = checkpoint.get("state_dict")
        if (
            not isinstance(optimizers, list)
            or len(optimizers) != 1
            or not isinstance(optimizer_states, list)
            or len(optimizer_states) != 1
            or not isinstance(state_dict, Mapping)
        ):
            raise RuntimeError(
                "cannot checkpoint formal P2 optimizer parameter contract"
            )

        optimizer = optimizers[0]
        optimizer_state = optimizer_states[0]
        saved_groups = optimizer_state.get("param_groups")
        live_groups = getattr(optimizer, "param_groups", None)
        if (
            not isinstance(saved_groups, list)
            or not isinstance(live_groups, list)
            or len(saved_groups) != len(live_groups)
        ):
            raise RuntimeError(
                "cannot checkpoint formal P2 optimizer parameter contract"
            )

        model_state = {}
        for name, value in state_dict.items():
            if (
                not isinstance(name, str)
                or not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
            ):
                raise RuntimeError(
                    "cannot checkpoint formal P2 optimizer parameter contract"
                )
            model_state[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        model_state_entries = [
            [name, metadata["shape"], metadata["dtype"]]
            for name, metadata in sorted(model_state.items())
        ]
        model_state_payload = json.dumps(
            model_state_entries,
            ensure_ascii=True,
            separators=(",", ":"),
        )

        names_by_identity = {
            id(parameter): name for name, parameter in self.named_parameters()
        }
        trainable_named_parameters = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        parameters = {}
        contract_parameter_groups = []
        ordered_trainable_parameters = []
        for saved_group, live_group in zip(saved_groups, live_groups):
            saved_parameter_ids = saved_group.get("params")
            live_parameters = live_group.get("params")
            if (
                not isinstance(saved_parameter_ids, list)
                or not isinstance(live_parameters, list)
                or len(saved_parameter_ids) != len(live_parameters)
            ):
                raise RuntimeError(
                    "cannot checkpoint formal P2 optimizer parameter contract"
                )
            contract_parameter_groups.append(list(saved_parameter_ids))
            for parameter_id, parameter in zip(
                saved_parameter_ids,
                live_parameters,
            ):
                name = names_by_identity.get(id(parameter))
                saved_parameter = state_dict.get(name) if name is not None else None
                if (
                    not isinstance(parameter_id, int)
                    or isinstance(parameter_id, bool)
                    or parameter_id in parameters
                    or not parameter.requires_grad
                    or not isinstance(name, str)
                    or not isinstance(saved_parameter, torch.Tensor)
                    or saved_parameter.shape != parameter.shape
                    or saved_parameter.dtype != parameter.dtype
                ):
                    raise RuntimeError(
                        "cannot checkpoint formal P2 optimizer parameter contract"
                    )
                parameters[parameter_id] = {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
                ordered_trainable_parameters.append(
                    [name, list(parameter.shape), str(parameter.dtype)]
                )
        if not parameters:
            raise RuntimeError(
                "cannot checkpoint formal P2 optimizer parameter contract"
            )
        expected_trainable_parameters = [
            [name, list(parameter.shape), str(parameter.dtype)]
            for name, parameter in trainable_named_parameters
        ]
        if ordered_trainable_parameters != expected_trainable_parameters:
            raise RuntimeError(
                "cannot checkpoint formal P2 optimizer parameter contract: "
                "optimizer order does not match trainable parameter order"
            )
        trainable_parameter_payload = json.dumps(
            ordered_trainable_parameters,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return {
            "schema_version": self._OPTIMIZER_PARAMETER_CONTRACT_SCHEMA_VERSION,
            "state_dict": model_state,
            "state_dict_schema_sha256": hashlib.sha256(
                model_state_payload.encode("ascii")
            ).hexdigest(),
            "param_groups": contract_parameter_groups,
            "parameters": parameters,
            "trainable_parameters": ordered_trainable_parameters,
            "trainable_parameter_schema_sha256": hashlib.sha256(
                trainable_parameter_payload.encode("ascii")
            ).hexdigest(),
        }

    def on_save_checkpoint(self, checkpoint):
        generator = self._train_sampler_generator()
        p2_fail_closed_runtime = _p2_general_flag(
            self.config,
            "p2_fail_closed_runtime",
        )
        if generator is None:
            if p2_fail_closed_runtime:
                raise RuntimeError(
                    "cannot checkpoint train sampler: sampler has no explicit "
                    "generator"
                )
            return

        checkpoint[self._TRAIN_SAMPLER_CHECKPOINT_KEY] = {
            "schema_version": self._TRAIN_SAMPLER_CHECKPOINT_SCHEMA_VERSION,
            "resume_scope": self._TRAIN_SAMPLER_RESUME_SCOPE,
            "mid_epoch_resume_supported": False,
            "dataloader_prefetch_state_checkpointed": False,
            "generator_state": generator.get_state().detach().cpu().clone(),
        }
        if p2_fail_closed_runtime:
            checkpoint[self._OPTIMIZER_PARAMETER_CONTRACT_KEY] = (
                self._optimizer_parameter_contract(checkpoint)
            )

    def on_load_checkpoint(self, checkpoint):
        self._pending_train_sampler_generator_state = None
        payload = checkpoint.get(self._TRAIN_SAMPLER_CHECKPOINT_KEY)
        if payload is None:
            return
        if not isinstance(payload, dict):
            raise ValueError("invalid train sampler generator checkpoint payload")
        if payload.get("schema_version") != self._TRAIN_SAMPLER_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported train sampler generator checkpoint schema")
        if payload.get("resume_scope") != self._TRAIN_SAMPLER_RESUME_SCOPE:
            raise ValueError("unsupported train sampler generator resume scope")

        state = payload.get("generator_state")
        if not isinstance(state, torch.Tensor):
            raise ValueError("train sampler generator checkpoint state must be a tensor")
        self._pending_train_sampler_generator_state = state.detach().cpu().clone()
        self._restore_train_sampler_generator_state()

    def setup(self, stage=None):
        """Setup is called on every process and after prepare_data"""
        if stage == 'fit' or stage is None:
            self.train_dataset = hydra.utils.instantiate(
                self.config.data.train_dataset
            )
            if (
                _p2_general_flag(self.config, "p2_fail_closed_runtime")
                and self._train_sampler_generator() is None
            ):
                raise RuntimeError(
                    "cannot set up P2 train sampler: sampler has no explicit "
                    "generator"
                )
            self._restore_train_sampler_generator_state()
            self.validation_dataset = hydra.utils.instantiate(
                self.config.data.validation_dataset
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = hydra.utils.instantiate(
                self.config.data.test_dataset
            )
            self.validation_dataset = hydra.utils.instantiate(
                self.config.data.validation_dataset
            )
            self.labels_info = self.validation_dataset.label_info
            

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        
        # If dataset has a sampler, pass it as override (Hydra accepts kwargs that override config)
        kwargs = {}
        if hasattr(self.train_dataset, 'sampler') and self.train_dataset.sampler is not None:
            kwargs['sampler'] = self.train_dataset.sampler
            kwargs['shuffle'] = False  # Can't use shuffle with sampler
        
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
            **kwargs
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )

    def _freeze_backbone_parameters(self):
        """Freeze backbone parameters based on freeze mode - DDP compatible
        
        Handles both Minkowski and Sonata backbones:
        - Minkowski: backbone.conv0, backbone.bn0, backbone.block1, etc.
        - Sonata: backbone.model.enc.*, backbone.model.embedding.*, etc.
        """
        freeze_mode = self.config.general.freeze
        
        if freeze_mode == "none":
            return
            
        frozen_count = 0
        total_backbone_params = 0
        
        for name, param in self.model.named_parameters():
            if name.startswith("backbone"):
                total_backbone_params += 1
                
                
            if freeze_mode == "backbone_encoder":
                # Minkowski encoder blocks (conv0, bn0, block1-4, etc.)
                if (name.startswith("backbone") and 
                    any(x in name for x in ['backbone.conv0', 'backbone.bn0', 
                                            'backbone.conv1p1', 'backbone.bn1','backbone.block1', 
                                            'backbone.conv2p2', 'backbone.bn2', 'backbone.block2', 
                                            'backbone.conv3p4', 'backbone.bn3', 'backbone.block3', 
                                            'backbone.conv4p8', 'backbone.bn4', 'backbone.block4'])):
                    param.requires_grad_(False)
                    frozen_count += 1
                # Sonata encoder and embedding parameters
                elif (name.startswith("backbone.model.enc") or 
                      name.startswith("backbone.model.embedding")):
                    param.requires_grad_(False)
                    frozen_count += 1
            elif freeze_mode == "backbone":
                # Freeze all backbone parameters (both Minkowski and Sonata)
                if name.startswith("backbone"):
                    param.requires_grad_(False)
                    frozen_count += 1
        
        # Set frozen modules to eval mode for optimal performance
        self._set_frozen_modules_eval(freeze_mode)
        
        print(f"Frozen {frozen_count}/{total_backbone_params} backbone parameters (mode: {freeze_mode})")

    def _set_frozen_modules_eval(self, freeze_mode):
        """Set frozen backbone modules to eval mode
        
        Handles both Minkowski and Sonata backbones:
        - Minkowski: backbone.conv0, backbone.bn0, backbone.block1, etc.
        - Sonata: backbone.model.enc.*, backbone.model.embedding.*, etc.
        """
        for name, module in self.model.named_modules():
            # only set to eval more if the entire backbone is frozen 
            if freeze_mode == "backbone" and name.startswith("backbone"):
                # For Minkowski, don't freeze final layer
                if not name.startswith("backbone.final"):
                    module.eval()
                # For Sonata, freeze all model parameters
                elif name.startswith("backbone.model"):
                    module.eval()


    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Quick check for unused parameters at the end of each training step"""
        if self.config.general.get('quick_param_check', False):
            unused = sum(1 for p in self.model.parameters() if not p.requires_grad)
            if unused > 0:
                print(f"⚠️  {unused} unused parameters detected")

    def on_before_optimizer_step(self, optimizer):
        if _p2_general_flag(self.config, "p2_fail_closed_runtime"):
            collective_device = _fallback_collective_device(self)
            batch_idx = None
            file_names = "<unavailable>"
            gradient_failure = None
            gradient_cause = None
            try:
                context = getattr(self, "_p2_optimizer_context", {})
                if isinstance(context, Mapping):
                    batch_idx = context.get("batch_idx")
                    file_names = context.get("file_names", file_names)
                    collective_device = context.get(
                        "device",
                        collective_device,
                    )
                gradient_failure = _optimizer_gradient_failure(optimizer)
            except Exception as error:
                gradient_failure = _phase_exception_reason("gradient", error)
                gradient_cause = error

            _batch_contract_consensus(
                stage="train",
                batch_idx=batch_idx,
                file_names=file_names,
                reason=gradient_failure,
                cause=gradient_cause,
                device=collective_device,
            )

        norms = pl.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)
