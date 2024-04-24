import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from mthd.presto import Presto


class PrestoModule(LightningModule):

    def __init__(self,
                 encoder_embedding_size: int = 128,
                 channel_embed_ratio: int = 0.25,
                 month_embed_ratio: int = 0.25,
                 encoder_depth: int = 2,
                 mlp_ratio: int = 4,
                 encoder_num_heads: int = 8,
                 decoder_embedding_size: int = 128,
                 decoder_depth: int = 2,
                 decoder_num_heads: int = 8,
                 num_landcover_classes: int = 10,
                 band_groups: list = [("IR", [0, 1, 2, 3, 4, 5, 6]),
                                      ("VIS", [7, 8]), ("WV", [9, 10])],
                 loss="mse",
                 loss_lc="ce",
                 lc_loss_weight: float = 2.0,
                 optimizer: str = 'adam',
                 lr: float = 1e-3,
                 scheduler: str = 'cosine'):

        super(PrestoModule, self).__init__()
        self.model = Presto.construct(num_landcover_classes,
                                      band_groups,
                                      encoder_embedding_size=128,
                                      channel_embed_ratio=0.25,
                                      month_embed_ratio=0.25,
                                      encoder_depth=2,
                                      mlp_ratio=4,
                                      encoder_num_heads=8,
                                      decoder_embedding_size=128,
                                      decoder_depth=2,
                                      decoder_num_heads=8,
                                      max_sequence_length=672)

        assert loss in ["mse", "mae"], "Loss must be either 'mse' or 'mae'"
        assert loss_lc in ["ce"], "Landcover loss must be 'ce' "
        self.loss = torch.nn.MSELoss() if loss == "mse" else torch.nn.L1Loss()
        self.loss_lc = torch.nn.CrossEntropyLoss()
        self.loss_class = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.lc_loss_weight = lc_loss_weight

    def forward(self, x, mask, landcover, latlons, months):
        y_pred, lc_pred = self.model(x,
                                     mask=mask,
                                     landcover=landcover,
                                     latlons=latlons,
                                     months=months)
        return y_pred, lc_pred

    def compute_loss(self,
                     y_pred,
                     y_true,
                     lc_pred,
                     lc_true,
                     mask,
                     lc_mask,
                     class_gt,
                     class_pred,
                     padding_mask_eo=None,
                     padding_mask_lc=None):

        if padding_mask_eo is not None:
            y_pred = y_pred[padding_mask_eo]
            y_true = y_true[padding_mask_eo]

        if padding_mask_lc is not None:
            lc_pred = lc_pred[padding_mask_lc]
            lc_true = lc_true[padding_mask_lc]

        eo_loss = self.loss(y_pred, y_true)
        lc_loss = self.loss_lc(lc_pred[lc_mask], lc_true[lc_mask])
        num_eo_masked, num_lc_masked = len(y_pred[mask]), len(lc_pred[lc_mask])
        with torch.no_grad():
            ratio = num_lc_masked / max(num_eo_masked, 1)
            # weight shouldn't be > 1
            weight = min(1, self.lc_loss_weight * ratio)
        class_loss = self.loss_class(class_pred, class_gt.float())
        total_loss = eo_loss + weight * lc_loss + class_loss
        return total_loss, eo_loss, lc_loss, class_loss

    def forward_compute_loss(self, batch):
        mask_eo, mask_lc, x_eo, y_eo, x_lc, y_lc, latlons, months, class_gt = batch
        # mask_eo, mask_lc, x_eo, y_eo, x_lc, y_lc, latlons, months, class_gt, padding_mask_eo, padding_mask_lc = batch
        y_pred, lc_pred, class_pred = self.model(x_eo, x_lc, latlons, mask_eo,
                                                 months)
        loss, eo_loss, lc_loss, class_loss = self.compute_loss(
            y_pred, y_eo, lc_pred, y_lc, mask_eo, mask_lc, class_gt,
            class_pred)
        return loss, eo_loss, lc_loss, class_loss

    def training_step(self, batch, batch_idx):
        loss, eo_loss, lc_loss, class_loss = self.forward_compute_loss(batch)
        self.log("train_loss", loss)
        self.log("train_eo_loss", eo_loss)
        self.log("train_lc_loss", lc_loss)
        self.log("train_class_loss", class_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, eo_loss, lc_loss, class_loss = self.forward_compute_loss(batch)
        self.log("val_loss", loss)
        self.log("val_eo_loss", eo_loss)
        self.log("val_lc_loss", lc_loss)
        self.log("val_class_loss", class_loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, eo_loss, lc_loss, class_loss = self.forward_compute_loss(batch)
        self.log("test_loss", loss)
        self.log("test_eo_loss", eo_loss)
        self.log("test_lc_loss", lc_loss)
        self.log("test_class_loss", class_loss)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        momentum=0.9,
                                        weight_decay=1e-4)
        if self.scheduler == 'lrplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   patience=10)
            return [optimizer], [{
                "scheduler": scheduler,
                "monitor": "val_loss"
            }]

        elif self.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=10)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

        elif self.scheduler == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=10)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(
            epoch=self.current_epoch)  # timm's scheduler need the epoch value
