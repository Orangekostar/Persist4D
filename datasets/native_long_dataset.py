"""Metadata holdouts and stateless, paired native training draws."""

import copy
import hashlib
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def stable_seed(*values):
    payload = ":".join(map(str, values)).encode("utf8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


@contextmanager
def isolated_rng(seed, *, cuda_devices=()):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(cuda_devices)):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        for device in cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def horizon_weights(update, total_updates=29700):
    p = update / total_updates
    if p < .1:
        return {2: 1., 3: 0., 4: 0., 5: 0.}
    if p < .25:
        a = (p - .1) / .15
        return {2: 1 - .75 * a, 3: .25 * a, 4: .25 * a, 5: .25 * a}
    return {h: .25 for h in (2, 3, 4, 5)}


def assign_development_roles(records, target=8):
    groups = {h: [] for h in (2, 3, 4, 5)}
    for record in records:
        item = copy.deepcopy(record)
        item["role"] = "TRAIN"
        groups[min(5, item["Tmax"])].append(item)
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha256(
            ("native-long-7001:" + r["reference_id"]).encode()).hexdigest())
    counts = {"CAL": 0, "SEL": 0}
    retained = {h: 1 for h in groups}
    retained[5] = 2 if len(groups[5]) >= 6 else 1

    def take(h, role):
        train = [r for r in groups[h] if r["role"] == "TRAIN"]
        if len(train) <= retained[h] or counts[role] >= target:
            return False
        train[0]["role"] = role
        counts[role] += 1
        return True

    for _ in range(2 if len(groups[5]) >= 6 else 1):
        for role in ("CAL", "SEL"):
            take(5, role)
    next_role = 0
    while any(counts[role] < target for role in counts):
        changed = False
        for horizon in (4, 3, 2, 5):
            role = ("CAL", "SEL")[next_role]
            if counts[role] >= target:
                role = ("SEL", "CAL")[next_role]
            if take(horizon, role):
                changed = True
                next_role = 1 - next_role
        if not changed:
            break
    return sorted((r for g in groups.values() for r in g), key=lambda r: r["reference_id"])


def canonical_orders(reference, seed=7001):
    size = len(reference["scan_ids"])
    return [np.random.default_rng(stable_seed(seed, reference["reference_id"], i))
            .permutation(size).tolist() for i in range(3)]


def evaluation_inputs(references, role):
    inputs, seen, singles = [], set(), {}
    for reference in references:
        if reference["role"] != role:
            continue
        for order in canonical_orders(reference):
            for h in range(2, min(5, reference["Tmax"]) + 1):
                positions = order[:h]
                scans = tuple(reference["scan_ids"][i] for i in positions)
                if scans in seen:
                    continue
                seen.add(scans)
                inputs.append({"input_id": f"{role}:" + hashlib.sha256(
                    ":".join(scans).encode()).hexdigest()[:24],
                    "reference_id": reference["reference_id"], "horizon": h,
                    "domain": "rio", "scan_ids": list(scans),
                    "scan_indices": [reference["scan_indices"][i] for i in positions]})
                for i in positions:
                    singles[reference["scan_ids"][i]] = (reference, i)
    for scan, (reference, i) in sorted(singles.items()):
        inputs.append({"input_id": f"{role}:T1:{scan}", "domain": "rio", "horizon": 1,
                       "reference_id": reference["reference_id"], "scan_ids": [scan],
                       "scan_indices": [reference["scan_indices"][i]]})
    return inputs


class NativeDrawPlan:
    def __init__(self, references, *, scannet_count, seed=45, total_updates=29700):
        self.references = sorted([r for r in references if r["role"] == "TRAIN"],
                                 key=lambda r: r["reference_id"])
        if not self.references or scannet_count < 1:
            raise ValueError("both official training domains must be populated")
        self.scannet_count = scannet_count
        self.seed, self.total_updates = seed, total_updates

    def draw(self, draw_id, *, arm):
        if arm not in {"E0", "E1", "E2", "E3"} or not 0 <= draw_id < self.total_updates * 32:
            raise ValueError("illegal arm or global draw ID")
        rng = np.random.default_rng(stable_seed(self.seed, "draw", draw_id))
        augmentation_seed = stable_seed(self.seed, "augmentation", draw_id)
        common = {"draw_id": draw_id, "augmentation_seed": augmentation_seed}
        if rng.random() >= 5 / 9:
            index = int(rng.integers(self.scannet_count))
            return {**common, "domain": "scannet", "horizon": 1,
                    "reference_id": f"scannet:{index}", "scan_ids": [f"scannet:{index}"],
                    "scan_indices": [index]}
        ref = self.references[int(rng.integers(len(self.references)))]
        order_index = int(rng.integers(3))
        order = canonical_orders(ref, self.seed)[order_index]
        start = int(rng.integers(len(order)))
        order = order[start:] + order[:start]
        weights = horizon_weights(draw_id // 32, self.total_updates)
        horizons = list(range(2, min(5, ref["Tmax"]) + 1))
        probabilities = np.array([weights[h] for h in horizons])
        probabilities /= probabilities.sum()
        h = int(rng.choice(horizons, p=probabilities))
        if arm == "E0":
            h = 2
        order = order[:h]
        return {**common, "domain": "rio", "reference_id": ref["reference_id"],
                "horizon": h, "order": order_index, "start": start,
                "scan_ids": [ref["scan_ids"][i] for i in order],
                "scan_indices": [ref["scan_indices"][i] for i in order]}


class GlobalDrawSampler(Sampler):
    def __init__(self, *, next_draw, total_updates, rank, world_size, per_rank_batch):
        if world_size * per_rank_batch not in (1, 2, 4) or 32 % (world_size * per_rank_batch):
            raise ValueError("physical batch must divide global batch32")
        if next_draw % 32 or next_draw > total_updates * 32:
            raise ValueError("resume must start at an optimizer boundary")
        self.next_draw, self.end = next_draw, total_updates * 32
        self.rank, self.world_size, self.per_rank_batch = rank, world_size, per_rank_batch

    def __iter__(self):
        width = self.world_size * self.per_rank_batch
        for start in range(self.next_draw, self.end, width):
            for slot in range(self.per_rank_batch):
                yield start + self.rank * self.per_rank_batch + slot

    def __len__(self):
        return (self.end - self.next_draw) // self.world_size


class NativeLongDataset(Dataset):
    def __init__(self, datasets, plan, *, arm):
        self.datasets, self.plan, self.arm = datasets, plan, arm

    def __len__(self):
        return self.plan.total_updates * 32

    def __getitem__(self, draw_id):
        record = self.plan.draw(int(draw_id), arm=self.arm)
        dataset = self.datasets[record["domain"]]
        indices = tuple(record["scan_indices"])
        with isolated_rng(record["augmentation_seed"]):
            sample = list(dataset.load_scan_indices(indices[0], indices, change_file=None))
        if sample[0].shape[1] != 4 or sample[6].shape[1] != 4:
            raise ValueError("all native inputs must use xyz+t")
        sample[3] = f"draw:{draw_id}:{record['domain']}:{record['reference_id']}"
        sample[7] = int(draw_id)
        return tuple(sample)


class NativeLongCollator:
    def __init__(self, config, *, training):
        import hydra

        self.base = hydra.utils.instantiate(
            config.data.train_collation if training else config.data.validation_collation,
            preserve_empty_targets=True)
        self.seed = int(config.general.seed) if training else 9001

    def __call__(self, samples):
        transform = self.base.collate_fn.transform
        seeds = iter(stable_seed(self.seed, "voxel", sample[3], int(stage))
                     for sample in samples for stage in np.unique(sample[0][:, 3]))

        def isolated_transform(point):
            with isolated_rng(next(seeds)):
                return transform(point)

        self.base.collate_fn.transform = isolated_transform
        try:
            return self.base(samples)
        finally:
            self.base.collate_fn.transform = transform


def build_native_datasets(config, data_root, *, augmentation):
    import hydra
    from omegaconf import OmegaConf

    root = Path(data_root)
    result = {}
    common = OmegaConf.to_container(config.data.train_dataset, resolve=True)
    for child in common["datasets"]:
        domain = child["dataset_name"]
        kwargs = {k: v for k, v in common.items()
                  if k not in {"_target_", "weights", "datasets", "epoch_sample_multiple", "sampler_seed"}}
        kwargs.update({k: v for k, v in child.items() if k != "target"})
        folder = root / "processed" / domain
        kwargs.update(data_dir=str(folder), label_db_filepath=str(folder / "label_database.yaml"),
                      color_mean_std=str(folder / "color_mean_std.yaml"), temporal_window=1,
                      mode="train", apply_training_augmentation=augmentation,
                      known_empty_scan_policy="allow_actual", exclude_unsupervised_sequences=False)
        result[domain] = hydra.utils.instantiate({"_target_": child["target"], **kwargs})
        for item in result[domain].data:
            for key in ("filepath", "instance_gt_filepath"):
                path = Path(item[key].replace("../../", ""))
                item[key] = str(root / path.relative_to("data")) if not path.is_absolute() else str(path)
        if domain == "rio":
            import yaml

            sequences = yaml.safe_load((folder / "sequence_database_sliding_2.yaml").read_text())
            by_scene = {}
            for entry in sequences.values():
                if entry.get("type") == "train":
                    scene = entry["scene"]
                    ambiguities = entry.get("ambiguities")
                    if scene in by_scene and by_scene[scene] != ambiguities:
                        raise ValueError("ambiguity metadata differs within reference")
                    by_scene[scene] = ambiguities
            result[domain].ambiguities = [by_scene.get(row["scene"]) for row in result[domain].data]
    return result
