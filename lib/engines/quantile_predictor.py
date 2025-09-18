from typing import Type, Mapping, Optional

from einops import rearrange
from torchmetrics import Metric

from lib.engines.base_predictor import BasePredictor
from lib.metrics.torch_metrics.pinball_loss import MaskedMultiPinballLoss
from tsl.utils.python_utils import ensure_list

class QuantilePredictor(BasePredictor):
    def __init__(
        self,
        model_class: Type,
        model_kwargs: Mapping,
        optim_class: Type,
        optim_kwargs: Mapping,
        quantiles: list,
        scale_target: bool = False,
        metrics: Optional[Mapping[str, Metric]] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
        sensor_names: Optional[list] = None,
    ):
        self.quantiles = ensure_list(quantiles)
        self.n_quantiles = len(self.quantiles)
        self.n_targets = model_kwargs['output_size'] // self.n_quantiles
        self.sensor_names = sensor_names

        loss_fn = MaskedMultiPinballLoss(qs=self.quantiles)

        super(QuantilePredictor, self).__init__(
            model_class=model_class,
            model_kwargs=model_kwargs,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            loss_fn=loss_fn,
            scale_target=scale_target,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
        )

    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, Metric):
            raise ValueError('Each metric must be an instance of Metric')
        metric = metric.clone()
        metric.reset()
        return metric

    def shared_step(self, batch, preprocess=False):
        y = y_loss = batch.y
        mask = batch.get('mask_target')

        # Compute predictions
        y_hat_loss = self.predict_batch(
            batch, preprocess=preprocess, postprocess=not self.scale_target
        )

        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        return y_hat, y, loss, mask

    def training_step(self, batch, batch_idx):
        y_hat, y, loss, mask = self.shared_step(batch)

        self.train_metrics.update(y_hat, y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss, mask = self.shared_step(batch)
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', loss, batch_size=batch.batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y, test_loss, mask = self.shared_step(batch)
        self.test_metrics.update(y_hat, y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)

        # Log per-sensor and general normalized metrics
        if self.sensor_names:
            # TSL format: [quantiles, batch, horizon, nodes, channels]
            # Convert to [batch, nodes, quantiles] for analysis
            
            if y_hat.dim() == 5:
                # [quantiles, batch, horizon, nodes, channels] -> [batch, nodes, quantiles]
                y_hat_reshaped = y_hat.permute(1, 3, 0, 2, 4)  # [batch, nodes, quantiles, horizon, channels]
                y_hat_reshaped = y_hat_reshaped.squeeze(dim=(3, 4))  # [batch, nodes, quantiles]
            else:
                # Fallback for other shapes
                y_hat_reshaped = y_hat.permute(1, 3, 0, 2).squeeze(dim=3)  # [batch, nodes, quantiles]
            
            # Calculate metrics for different confidence levels
            alphas = [0.01, 0.05, 0.1, 0.2]  # 99%, 95%, 90%, 80% confidence intervals
            
            for alpha in alphas:
                target_qs = self.quantiles
                idx_low = min(range(len(target_qs)), key=lambda i: abs(target_qs[i] - alpha/2))
                idx_high = min(range(len(target_qs)), key=lambda i: abs(target_qs[i] - (1-alpha/2)))
                
                # Calculate PI widths in ORIGINAL space
                pi_widths_original = y_hat_reshaped[:, :, idx_high] - y_hat_reshaped[:, :, idx_low]  # [batch, nodes]
                avg_pi_widths_original = pi_widths_original.mean(dim=0)  # Average across batch: [nodes]
                
                # Calculate general original space metrics
                general_pi_width_original = pi_widths_original.mean()  # Average across all sensors and batch
                
                # Calculate PI widths in NORMALIZED space
                if self.scale_target and 'transform' in batch and 'y' in batch.transform:
                    # Get normalized predictions and targets
                    y_hat_normalized = batch.transform['y'].transform(y_hat_reshaped)  # [1, batch, nodes, quantiles]
                    y_normalized = batch.transform['y'].transform(y)  # [batch, horizon, nodes, channels]
                    
                    # Handle the extra dimension from transform
                    while y_hat_normalized.dim() > 3 and y_hat_normalized.shape[0] == 1:
                        y_hat_normalized = y_hat_normalized.squeeze(dim=0)
                    
                    # Ensure we have the right shape: [batch, nodes, quantiles]
                    if y_hat_normalized.shape != y_hat_reshaped.shape:
                        if y_hat_normalized.dim() == 4:
                            y_hat_normalized = y_hat_normalized.permute(1, 2, 3, 0).squeeze(dim=-1)
                    
                    # Calculate normalized PI widths
                    pi_widths_normalized = y_hat_normalized[:, :, idx_high] - y_hat_normalized[:, :, idx_low]  # [batch, nodes]
                    avg_pi_widths_normalized = pi_widths_normalized.mean(dim=0)  # [nodes]
                    
                    # Calculate general normalized space metrics
                    general_pi_width_normalized = pi_widths_normalized.mean()  # Average across all sensors and batch
                    
                    # Calculate normalized Winkler scores
                    from lib.metrics.torch_metrics.winkler import winkler_score
                    winkler_scores_normalized = winkler_score(
                        [y_hat_normalized[:, :, idx_low], y_hat_normalized[:, :, idx_high]], 
                        y_normalized.squeeze(dim=(1, 3)),  # [batch, nodes]
                        alpha=alpha
                    )  # [batch, nodes]
                    avg_winkler_normalized = winkler_scores_normalized.mean(dim=0)  # [nodes]
                    
                    # Calculate general normalized Winkler score
                    general_winkler_normalized = winkler_scores_normalized.mean()  # Average across all sensors and batch
                    
                    # Log general normalized metrics
                    confidence_pct = int((1-alpha) * 100)
                    self.log(f'test_pi_width_at_{confidence_pct}_normalized', float(general_pi_width_normalized))
                    self.log(f'test_winkler_at_{confidence_pct}_normalized', float(general_winkler_normalized))
                    
                    # Log per-sensor normalized metrics
                    for i, sensor in enumerate(self.sensor_names):
                        self.log(f'test_pi_width_{sensor}_normalized', float(avg_pi_widths_normalized[i]))
                        self.log(f'test_winkler_{sensor}_normalized', float(avg_winkler_normalized[i]))
                
                # Log per-sensor original space metrics
                for i, sensor in enumerate(self.sensor_names):
                    self.log(f'test_pi_width_{sensor}', float(avg_pi_widths_original[i]))

        return test_loss

    def attack_detection_step(self, batch, batch_idx):
        """Evaluate model on attack data and compute detection metrics"""
        y_hat, y, loss, mask = self.shared_step(batch)
        
        # Compute coverage deviation metrics
        if self.sensor_names:
            # Calculate per-sensor coverage rates
            for alpha in [0.1, 0.2]:  # 90%, 80% confidence
                target_qs = self.quantiles
                idx_low = min(range(len(target_qs)), key=lambda i: abs(target_qs[i] - alpha/2))
                idx_high = min(range(len(target_qs)), key=lambda i: abs(target_qs[i] - (1-alpha/2)))
                
                # Check if predictions fall within intervals
                lower_bound = y_hat[idx_low]
                upper_bound = y_hat[idx_high]
                coverage = ((y >= lower_bound) & (y <= upper_bound)).float()
                
                # Log coverage deviation (should be ~0.9 for 90% confidence)
                expected_coverage = 1 - alpha
                coverage_deviation = coverage.mean() - expected_coverage
                
                confidence_pct = int((1-alpha) * 100)
                self.log(f'attack_coverage_deviation_{confidence_pct}', float(coverage_deviation))
                
                # Log per-sensor coverage deviations
                for i, sensor in enumerate(self.sensor_names):
                    sensor_coverage = coverage[:, i].mean()
                    sensor_deviation = sensor_coverage - expected_coverage
                    self.log(f'attack_coverage_deviation_{sensor}_{confidence_pct}', float(sensor_deviation))
        
        return loss

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.

        :param batch: the batch
        :return: batch_data, batch_preprocessing
        """
        inputs, targets = batch.input, batch.target
        mask = batch.get('mask_target')
        transform = batch.get('transform')
        return inputs, targets, mask, transform

    def compute_metrics(self, batch, preprocess=False, postprocess=True):
        """"""
        raise NotImplementedError("compute_metrics not implemented for QuantilePredictor")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out = super(QuantilePredictor, self).predict_step(batch, batch_idx, dataloader_idx)
        # reshape quantile to allow for stacking predictions
        out['y_hat'] = rearrange(out['y_hat'], 'q ... -> ... q')
        return out
