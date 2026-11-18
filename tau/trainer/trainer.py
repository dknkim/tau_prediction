import os
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
import wandb


class Trainer:
    """Train and validate a regression model on one device."""

    def __init__(
        self,
        model,
        loss_function,
        metric_ftns,
        optimizer,
        lr_scheduler,
        config,
        project,
        data_loader,
        valid_data_loader,
        device,
        seed=None,
        tags=None,
        mixed_precision=False,
        entity=None,
        classification=False,
        crop_foreground=False,
    ):
        if classification or crop_foreground:
            raise ValueError("Trainer supports regression on preprocessed volumes.")
        if seed is not None:
            torch.manual_seed(seed)

        self.config = config
        self.run = wandb.init(
            config=config,
            entity=entity,
            project=project,
            tags=tags,
            save_code=True,
            reinit=True,
            name=config["name"],
            mode=os.environ.get("WANDB_MODE", "disabled"),
        )
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_function = loss_function.to(self.device)
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.mixed_precision = mixed_precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
        self.checkpoint_dir = Path(config["save_dir"]) / date.today().isoformat()

    def train(self):
        best_loss = float("inf")
        for epoch in range(1, self.config["epochs"] + 1):
            started = time.time()
            training = self._train_epoch(epoch)
            validation = self._valid_epoch()
            print(f"Epoch {epoch} with validation completed in {time.time() - started:.1f}s.")
            self.run.log({
                **training,
                **validation,
                "epoch": epoch,
                "learning_rate": self.lr_scheduler.get_last_lr()[0],
            })
            if epoch % self.config["save_period"] == 0 or epoch == self.config["epochs"]:
                self.save_checkpoint(epoch)
            if validation["val_loss"] < best_loss:
                best_loss = validation["val_loss"]
                self.save_checkpoint(epoch, best=True)

    def _train_epoch(self, epoch):
        self.model.train()
        losses = []
        batches = len(self.data_loader)
        log_step = max(1, batches // 4)
        for index, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                loss = self.loss_function(self.model(data), target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(loss.item())
            if index % log_step == 0:
                print(f"Train Epoch: {epoch} [{index}/{batches}] Loss: {losses[-1]:.6f}")

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return {"loss": np.mean(losses)}

    @torch.no_grad()
    def _valid_epoch(self):
        self.model.eval()
        metrics = defaultdict(list)
        for cohort, loader in self.valid_data_loader.items():
            for data, target in loader:
                target = target.to(self.device)
                prediction = self.model(data.to(self.device))
                loss = self.loss_function(prediction, target).item()
                metrics[f"{cohort}_val_loss"].append(loss)
                metrics["val_loss"].append(loss)
                prediction, target = prediction.cpu(), target.cpu()
                for name, metric in self.metric_ftns.items():
                    metrics[name].append(metric(prediction, target).item())
        return {name: np.mean(values) for name, values in metrics.items()}

    def save_checkpoint(self, epoch, best=False):
        state = {
            "arch": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "loss_func": str(self.loss_function),
        }
        if best:
            path = self.checkpoint_dir / "best_validation" / "checkpoint-best.pth"
        else:
            path = self.checkpoint_dir / f"epoch_{epoch}" / f"checkpoint-epoch{epoch}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        print(f"Saving checkpoint: {path} ...")
