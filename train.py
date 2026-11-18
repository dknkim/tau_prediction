import argparse
import os
from pathlib import Path

import torch
from monai.networks.nets import DenseNet264
from torch import nn
from torch.utils.data import DataLoader

from tau.dataset.dataloader import Loader
from tau.dataset.datasetcontainer import DatasetContainer
from tau.trainer.trainer import Trainer
from utils_sample import train_transforms_pre, train_transforms_pre_severe, valid_transforms_pre


COHORTS = ("adni", "oasis", "a4", "habs")
SCANS = ("MPRAGE", "FLAIR")
BRAAK_TARGETS = ("Braak1", "Braak2", "Braak3", "Braak4", "Braak5", "Braak6", "Braak1to4", "Braak5_6")


def correlation_loss(pred, target):
    """Correlation loss to maintain relationships between outputs"""
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    vx = pred - torch.mean(pred, dim=0)
    vy = target - torch.mean(target, dim=0)
    corr = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)) + 1e-8)
    return 1 - torch.mean(corr)


def residual_composite_consistency_loss(pred, target):
    """
    Match the residual relationship between composite outputs and
    the mean of their constituent Braak stage outputs.

    Compare each predicted residual with the corresponding target residual.

    Output convention:
        pred[:, 0] = Braak1
        pred[:, 1] = Braak2
        pred[:, 2] = Braak3
        pred[:, 3] = Braak4
        pred[:, 4] = Braak5
        pred[:, 5] = Braak6
        pred[:, 6] = Braak1to4
        pred[:, 7] = Braak5_6
    """

    # ---- Braak1to4 residual ----
    pred_mean_1to4 = torch.mean(pred[:, :4], dim=1)
    target_mean_1to4 = torch.mean(target[:, :4], dim=1)

    pred_res_1to4 = pred[:, 6] - pred_mean_1to4
    target_res_1to4 = target[:, 6] - target_mean_1to4

    loss_1to4 = torch.mean((pred_res_1to4 - target_res_1to4) ** 2)

    # ---- Braak5_6 residual ----
    pred_mean_5_6 = torch.mean(pred[:, 4:6], dim=1)
    target_mean_5_6 = torch.mean(target[:, 4:6], dim=1)

    pred_res_5_6 = pred[:, 7] - pred_mean_5_6
    target_res_5_6 = target[:, 7] - target_mean_5_6

    loss_5_6 = torch.mean((pred_res_5_6 - target_res_5_6) ** 2)

    return loss_1to4 + loss_5_6


class MSECorrResidualCompositeLoss(nn.Module):
    """
    Combine regression, correlation, and composite residual losses.
        MSE + corr_weight * correlation_loss
            + residual_weight * residual_composite_consistency_loss
    """
    def __init__(self, corr_weight=0.2, residual_weight=0.05):
        super().__init__()
        self.corr_weight = corr_weight
        self.residual_weight = residual_weight
        self.reg_loss = nn.MSELoss()

    def forward(self, pred, target):
        reg = self.reg_loss(pred, target)
        corr = correlation_loss(pred, target)
        residual_comp = residual_composite_consistency_loss(pred, target)

        total_loss = (
            reg
            + self.corr_weight * corr
            + self.residual_weight * residual_comp
        )
        return total_loss


class SeverityAwareLoader:
    """Use stronger augmentation for cases with Braak1to4 above 1.34."""

    def __init__(self, base_loader):
        self.base_loader = base_loader
        self.severity_flags = []
        for entry_index, _, _ in base_loader.indicies:
            value = getattr(base_loader.container[entry_index], "Braak1to4", None)
            self.severity_flags.append(value is not None and value > 1.34)

    def __len__(self):
        return len(self.base_loader)

    def __getitem__(self, index):
        original = self.base_loader.transforms
        if self.severity_flags[index]:
            self.base_loader.transforms = train_transforms_pre_severe
        try:
            return self.base_loader[index]
        finally:
            self.base_loader.transforms = original


def load_split(cohort, fold, split):
    path = Path("dataset") / cohort / f"fold_{fold}" / f"{split}.json"
    dataset = DatasetContainer.from_json(path)
    for entry in dataset:
        for instance in entry:
            if "_bet" in instance.image_path:
                instance.sequence_type = "MASK"
    return dataset


def make_loader(container, transforms):
    return Loader(
        container=container,
        scans=SCANS,
        tracer="FTP",
        transforms=transforms,
        classification=False,
        histogram_normalize=False,
        brain_mask_centering=False,
        top_percentile=None,
        encode_dataset=False,
        clip=False,
        use_status=False,
    )


def rank_correlation_loss(pred, target):
    """Mean squared error between the predicted and target ranks."""
    pred_rank = torch.argsort(torch.argsort(pred, dim=1), dim=1).float()
    target_rank = torch.argsort(torch.argsort(target, dim=1), dim=1).float()
    return torch.nn.functional.mse_loss(pred_rank, target_rank)


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet264 for longitudinal tau regression.")
    parser.add_argument("-fold", type=int, required=True)
    parser.add_argument("-model", choices=["dense"], required=True)
    parser.add_argument("-gpu", type=int, required=True)
    parser.add_argument(
        "-loss_type", choices=["mse_corr_residual_comp"], default="mse_corr_residual_comp"
    )
    args = parser.parse_args()

    save_dir = (
        Path(os.environ.get("TAU_OUTPUT_DIR", "runs"))
        / "_".join(SCANS)
        / args.model
        / f"test9_fold_{args.fold}_{args.loss_type}"
    )
    config = {
        "name": f"fold_{args.fold}_{args.model}_Sample_{args.loss_type}",
        "epochs": 240,
        "iterative": False,
        "inputs_pr_iteration": 250,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "save_dir": str(save_dir),
        "save_period": 50,
        "weight_decay": 1e-4,
        "scans": list(SCANS),
        "data": list(COHORTS),
        "model": args.model,
        "loss_type": args.loss_type,
    }

    train = DatasetContainer()
    validation = {}
    for cohort in COHORTS:
        try:
            train += load_split(cohort, args.fold, "train")
            validation_dataset = load_split(cohort, args.fold, "valid")
        except FileNotFoundError as error:
            parser.error(f"Missing manifest {error.filename}. See the dataset templates in README.md.")
        validation[cohort.upper()] = make_loader(validation_dataset, valid_transforms_pre)

    train_dataset = SeverityAwareLoader(make_loader(train, train_transforms_pre))
    config["train_cases"] = len(train_dataset)
    config["valid_cases"] = sum(map(len, validation.values()))
    print(
        f"Training cases: {config['train_cases']} "
        f"({sum(train_dataset.severity_flags)} with stronger augmentation)"
    )
    print("Validation cases:", {cohort: len(loader) for cohort, loader in validation.items()})

    loader_options = {"num_workers": 16, "batch_size": config["batch_size"]}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    valid_loader = {
        cohort: DataLoader(loader, **loader_options)
        for cohort, loader in validation.items()
    }

    model = DenseNet264(
        spatial_dims=3,
        in_channels=len(SCANS),
        out_channels=8,
        init_features=64,
        growth_rate=32,
        pretrained=False,
        progress=True,
    )

    loss = MSECorrResidualCompositeLoss(corr_weight=0.2, residual_weight=0.05)
    print(f"{sum(p.numel() for p in model.parameters()):,} total parameters.")
    print(f"Using loss function: {args.loss_type}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    warmup_steps = 50
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1e-2 + step / warmup_steps * (1 - 1e-2)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=warmup_steps - config["epochs"]
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    metrics = {"MSE": nn.MSELoss(), "MAE": nn.L1Loss(), "RankCorr": rank_correlation_loss}
    for index, target_name in enumerate(BRAAK_TARGETS):
        metrics[f"MSE_{target_name}"] = (
            lambda pred, target, idx=index: nn.functional.mse_loss(pred[:, idx], target[:, idx])
        )
        metrics[f"MAE_{target_name}"] = (
            lambda pred, target, idx=index: nn.functional.l1_loss(pred[:, idx], target[:, idx])
        )

    trainer = Trainer(
        model=model,
        loss_function=loss,
        metric_ftns=metrics,
        optimizer=optimizer,
        config=config,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        lr_scheduler=lr_scheduler,
        seed=42,
        device=f"cuda:{args.gpu}",
        mixed_precision=True,
        tags=["3D", "regression", "8-output", args.model, "tau", "DK", args.loss_type],
        project="tau_MregALL56",
        classification=False,
        crop_foreground=False,
        entity=os.environ.get("WANDB_ENTITY"),
    )
    trainer.train()


if __name__ == "__main__":
    main()
