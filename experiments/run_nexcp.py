import os
import numpy as np
import pandas as pd
import torch
import yaml
from typing import Dict, List, Tuple

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

# Add wandb import
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from lib.datasets.swat import SWaTDataset

# Optional imports for datasets that may not be available
try:
    from lib.datasets.gpvar import GPVARDataset
except ImportError:
    GPVARDataset = None

try:
    from lib.datasets.air_quality import AirQuality
except ImportError:
    AirQuality = None
from lib.datasets.tep import TEPDataset
from lib.datasets.wadi import WADIDataset
from lib.metrics.torch_metrics.coverage import MaskedCoverage, MaskedDeltaCoverage, MaskedPIWidth
from lib.metrics.torch_metrics.winkler import MaskedWinklerScore
from lib.utils.data_utils import parse_and_filter_indices, find_close
from tsl import logger

from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule, BatchMap, BatchMapItem
from tsl.data.datamodule.splitters import FixedIndicesSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA
from tsl.experiment import Experiment

from lib.metrics.torch_metrics.wrappers import MaskedMetricWrapper


def get_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    
    if name == 'la':
        dataset = MetrLA()
    elif name == 'air':
        if AirQuality is None:
            raise ImportError("AirQuality dataset not available. Please ensure lib.datasets.air_quality is properly installed.")
        dataset = AirQuality()
    elif name == 'gpvar':
        if GPVARDataset is None:
            raise ImportError("GPVARDataset not available. Please ensure lib.datasets.gpvar is properly installed.")
        dataset = GPVARDataset(**dataset_cfg["hparams"], p_max=0)
    elif name == 'swat':
        sample_rate = dataset_cfg.get('sample_rate', 10)
        data_dir = dataset_cfg.get('data_dir', None)
        dataset = SWaTDataset(root=data_dir, sample_rate=sample_rate)
        print(f"SWaT dataset: {dataset.n_nodes} sensors, {len(dataset.actuator_names)} actuators (used as covariates)")
    elif name == 'tep':
        sample_rate = dataset_cfg.get('sample_rate', 10)
        data_dir = dataset_cfg.get('data_dir', None)
        dataset = TEPDataset(root=data_dir, sample_rate=sample_rate)
        print(f"TEP dataset: {dataset.n_nodes} sensors, {len(dataset.actuator_names)} actuators (used as covariates)")
    elif name == 'wadi':
        sample_rate = dataset_cfg.get('sample_rate', 10)
        data_dir = dataset_cfg.get('data_dir', None)
        dataset = WADIDataset(root=data_dir, sample_rate=sample_rate)
        print(f"WADI dataset: {dataset.n_nodes} sensors, {len(dataset.actuator_names)} actuators (used as covariates)")
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


class NexCPPredictor:
    """
    Next Conformal Prediction (NexCP) baseline with exponential weights.
    
    Uses a simpler approach: sliding window with exponential weighting.
    """
    
    def __init__(self, alphas: List[float], window_size: int = 1000, decay_rate: float = 0.99, sensor_names: List[str] = None):
        self.alphas = sorted(alphas)
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.sensor_names = sensor_names
        self.residual_history = []
        
    def update(self, residuals: np.ndarray):
        """
        Update the residual history with new residuals.
        
        Args:
            residuals: [N_sensors] array of residuals
        """
        self.residual_history.append(residuals.copy())
        
        # Keep only the most recent window_size residuals
        if len(self.residual_history) > self.window_size:
            self.residual_history = self.residual_history[-self.window_size:]
    
    def predict_intervals(self, predictions: np.ndarray) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate prediction intervals using NexCP with exponential weights.
        
        Args:
            predictions: [N_sensors] array of point predictions
            
        Returns:
            Dict mapping alpha -> (lower_bound, upper_bound)
        """
        if len(self.residual_history) == 0:
            raise ValueError("No residuals in history. Call update() first.")
        
        # Get recent residuals
        recent_residuals = np.array(self.residual_history)  # [T, N_sensors]
        abs_residuals = np.abs(recent_residuals)
        
        intervals = {}
        
        for alpha in self.alphas:
            # Compute exponential weights
            n = len(abs_residuals)
            weights = np.array([self.decay_rate ** (n - 1 - i) for i in range(n)])
            weights = weights / np.sum(weights)  # Normalize
            
            # Compute weighted quantile threshold
            thresholds = []
            
            for sensor_idx in range(abs_residuals.shape[1]):
                sensor_residuals = abs_residuals[:, sensor_idx]  # [T]
                
                # Sort residuals and weights together
                sorted_indices = np.argsort(sensor_residuals)
                sorted_residuals = sensor_residuals[sorted_indices]
                sorted_weights = weights[sorted_indices]
                
                # Compute weighted quantile
                cumsum_weights = np.cumsum(sorted_weights)
                target_weight = 1 - alpha
                
                # Find the quantile index
                if target_weight <= cumsum_weights[0]:
                    threshold = sorted_residuals[0]
                elif target_weight >= cumsum_weights[-1]:
                    threshold = sorted_residuals[-1]
                else:
                    # Linear interpolation
                    idx = np.searchsorted(cumsum_weights, target_weight)
                    if idx == 0:
                        threshold = sorted_residuals[0]
                    else:
                        # Interpolate between idx-1 and idx
                        w1 = cumsum_weights[idx-1]
                        w2 = cumsum_weights[idx]
                        r1 = sorted_residuals[idx-1]
                        r2 = sorted_residuals[idx]
                        
                        # Linear interpolation
                        if w2 > w1:
                            ratio = (target_weight - w1) / (w2 - w1)
                            threshold = r1 + ratio * (r2 - r1)
                        else:
                            threshold = r1
                
                thresholds.append(threshold)
            
            thresholds = np.array(thresholds)  # [N_sensors]
            
            # Create symmetric intervals around predictions
            lower_bound = predictions - thresholds
            upper_bound = predictions + thresholds
            
            intervals[alpha] = (lower_bound, upper_bound)
        
        return intervals


def run_experiment(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################

    local_dir = cfg.src_dir

    # Load residuals from base model
    residuals_input: pd.DataFrame = pd.read_hdf(os.path.join(local_dir, "residuals.h5"), key='input')
    residuals_target: pd.DataFrame = pd.read_hdf(os.path.join(local_dir, "residuals.h5"), key='target')
    
    with open(os.path.join(local_dir, "config.yaml"), 'r') as fp:
        src_config = yaml.load(fp, Loader=yaml.FullLoader)

    assert cfg.dataset.name == src_config["dataset"]["name"]
    dataset = get_dataset(src_config["dataset"])

    if cfg.dataset.name in {'gpvar'}:
        ds_index = pd.Index(dataset.index)
    else:
        ds_index = dataset.index

    try:
        mask_target = pd.read_hdf(os.path.join(local_dir, "residuals.h5"), key='target_mask')
        mask_target = mask_target.reindex(index=dataset.index)
    except KeyError:
        mask_target = None

    indices = np.load(os.path.join(local_dir, "indices.npz"))

    # Create dataset for evaluation
    covariates = dict(
        residuals_input=(residuals_input.reindex(index=ds_index), 't n f'),
        residuals_target=(residuals_target.reindex(index=ds_index), 't n f'),
    )

    # Use true targets for evaluation, not residuals
    target_map = BatchMap()
    target_map['y'] = BatchMapItem('target', synch_mode='horizon', pattern='t n f', preprocess=True)

    input_map = BatchMap()
    if mask_target is not None:
        input_map['mask_target'] = BatchMapItem(['mask_target'],
                                                synch_mode='horizon',
                                                pattern='t n f')
        covariates.update(mask_target=mask_target.astype('bool'))

    inputs_ = ['residuals_input']
    input_map['x'] = BatchMapItem(inputs_,
                                  synch_mode='window',
                                  pattern='t n f')

    torch_dataset = SpatioTemporalDataset(index=ds_index,
                                          target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          window=src_config["window"],
                                          stride=src_config["stride"],
                                          target_map=target_map,
                                          input_map=input_map,
                                          delay=src_config.get("delay", 0),
                                          horizon=1)

    calib_indices, test_indices = parse_and_filter_indices(torch_dataset, indices)

    # Split calibration into train/val for NexCP
    val_len = int(cfg.val_len * len(calib_indices))
    calib_indices, val_indices = calib_indices[:-val_len - torch_dataset.samples_offset], calib_indices[-val_len:]

    calib_splitter = FixedIndicesSplitter(
        train_idxs=calib_indices,
        val_idxs=val_indices,
        test_idxs=test_indices,
    )

    scale_axis = (0,) if src_config.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis),
        'residuals_target': StandardScaler(axis=scale_axis),
        'residuals_input': StandardScaler(axis=scale_axis),
    }

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=calib_splitter,
        batch_size=src_config["batch_size"],
        workers=src_config["workers"]
    )
    dm.setup()

    ########################################
    # NexCP Implementation                 #
    ########################################

    print("Implementing Next Conformal Prediction (NexCP) with exponential weights")
    
    # Initialize NexCP predictor
    nexcp_predictor = NexCPPredictor(
        alphas=cfg.alphas,
        window_size=cfg.get('window_size', 1000),
        decay_rate=cfg.get('decay_rate', 0.99),
        sensor_names=dataset.sensor_names if hasattr(dataset, 'sensor_names') else None
    )
    
    # Warm up with calibration data (use residuals for calibration)
    calib_data = dm.valset
    print(f"Warming up NexCP with {len(calib_data)} calibration samples")
    
    # Build calibration residuals DIRECTLY from the saved DataFrame
    residuals_all = residuals_target.reindex(index=ds_index).values  # [T_total, N_sensors]
    val_mask = np.zeros(len(ds_index), dtype=bool)
    val_mask[dm.valset.indices] = True
    calib_residuals = residuals_all[val_mask]                        # [N_calib, N_sensors]
    
    print(f"Calibration residuals shape: {calib_residuals.shape}")
    
    # Update NexCP with calibration residuals
    for i in range(len(calib_residuals)):
        nexcp_predictor.update(calib_residuals[i])
    
    print(f"NexCP warmed up with {len(nexcp_predictor.residual_history)} residuals")
    
    ########################################
    # Evaluation                           #
    ########################################
    
    print("Evaluating NexCP on test set")
    
    # Build y (true targets) on test set
    test_targets_scaled = []
    for i in range(len(dm.testset)):
        s = dm.testset[i]
        test_targets_scaled.append(s['y'].reshape(-1, s['y'].shape[-2]))
    y_scaled = np.concatenate(test_targets_scaled, axis=0)  # [N_test, N_sensors]

    # Collect residuals for the same test timesteps (scaled)
    residuals_all = residuals_target.reindex(index=ds_index).values
    test_mask = np.zeros(len(ds_index), dtype=bool)
    test_mask[dm.testset.indices] = True
    residuals_scaled = residuals_all[test_mask]  # [N_test, N_sensors]

    # Reconstruct base predictions in scaled space
    yhat_scaled = y_scaled - residuals_scaled
    
    print(f"True targets shape: {y_scaled.shape}")
    print(f"Residuals shape: {residuals_scaled.shape}")
    print(f"Base predictions shape: {yhat_scaled.shape}")
    
    print("Fitting scalers on training data for proper normalization")
    train_data = dm.trainset
    train_targets = []
    for i in range(len(train_data)):
        sample = train_data[i]
        targets = sample['y'].numpy()
        train_targets.append(targets)
    train_targets = np.concatenate(train_targets, axis=0)
    train_targets = train_targets.reshape(-1, train_targets.shape[-2])
    print(f"Training targets shape: {train_targets.shape}")

    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    train_scaler = SklearnStandardScaler()
    train_scaler.fit(train_targets)
    print(f"Fitted scaler - scale shape: {train_scaler.scale_.shape}")
    print(f"Scale range: [{train_scaler.scale_.min():.6f}, {train_scaler.scale_.max():.6f}]")
    print(f"Mean range: [{train_scaler.mean_.min():.6f}, {train_scaler.mean_.max():.6f}]")

    results = {}
    
    for alpha in cfg.alphas:
        q_scaled = nexcp_predictor.quantile_thresholds[alpha] if hasattr(nexcp_predictor, 'quantile_thresholds') else None
        
        if q_scaled is None:
            # Compute thresholds from current residual history
            recent_residuals = np.array(nexcp_predictor.residual_history)
            abs_residuals = np.abs(recent_residuals)
            n = len(abs_residuals)
            weights = np.array([nexcp_predictor.decay_rate ** (n - 1 - i) for i in range(n)])
            weights = weights / np.sum(weights)
            
            thresholds = []
            for sensor_idx in range(abs_residuals.shape[1]):
                sensor_residuals = abs_residuals[:, sensor_idx]
                sorted_indices = np.argsort(sensor_residuals)
                sorted_residuals = sensor_residuals[sorted_indices]
                sorted_weights = weights[sorted_indices]
                
                cumsum_weights = np.cumsum(sorted_weights)
                target_weight = 1 - alpha
                
                if target_weight <= cumsum_weights[0]:
                    threshold = sorted_residuals[0]
                elif target_weight >= cumsum_weights[-1]:
                    threshold = sorted_residuals[-1]
                else:
                    idx = np.searchsorted(cumsum_weights, target_weight)
                    if idx == 0:
                        threshold = sorted_residuals[0]
                    else:
                        w1 = cumsum_weights[idx-1]
                        w2 = cumsum_weights[idx]
                        r1 = sorted_residuals[idx-1]
                        r2 = sorted_residuals[idx]
                        
                        if w2 > w1:
                            ratio = (target_weight - w1) / (w2 - w1)
                            threshold = r1 + ratio * (r2 - r1)
                        else:
                            threshold = r1
                
                thresholds.append(threshold)
            
            q_scaled = np.array(thresholds)

        lower_scaled = yhat_scaled - q_scaled
        upper_scaled = yhat_scaled + q_scaled

        yhat_original = train_scaler.inverse_transform(yhat_scaled)
        y_original = train_scaler.inverse_transform(y_scaled)

        halfwidth_orig = q_scaled * train_scaler.scale_
        lower_orig = yhat_original - halfwidth_orig
        upper_orig = yhat_original + halfwidth_orig

        targets_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
        pi_bounds_scaled = torch.stack([
            torch.tensor(lower_scaled).unsqueeze(1).unsqueeze(-1).float(),
            torch.tensor(upper_scaled).unsqueeze(1).unsqueeze(-1).float()
        ], dim=0)

        targets_orig_tensor = torch.tensor(y_original, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
        pi_bounds_original = torch.stack([
            torch.tensor(lower_orig).unsqueeze(1).unsqueeze(-1).float(),
            torch.tensor(upper_orig).unsqueeze(1).unsqueeze(-1).float()
        ], dim=0)

        coverage_metric = MaskedCoverage()
        delta_cov_metric = MaskedDeltaCoverage(alpha=alpha)
        pi_width_metric = MaskedPIWidth()
        winkler_metric = MaskedWinklerScore(alpha=alpha)

        coverage = coverage_metric(pi_bounds_scaled, targets_tensor)
        delta_cov = delta_cov_metric(pi_bounds_scaled, targets_tensor)
        pi_width = pi_width_metric(pi_bounds_scaled, targets_tensor)
        winkler = winkler_metric(pi_bounds_scaled, targets_tensor)

        pi_width_original = pi_width_metric(pi_bounds_original, targets_orig_tensor)
        winkler_original = winkler_metric(pi_bounds_original, targets_orig_tensor)

        pct = int((1 - alpha) * 100)
        results[f'test_coverage_at_{pct}'] = float(coverage)
        results[f'test_delta_cov_at_{pct}'] = float(delta_cov)
        results[f'test_pi_width_at_{pct}_scaled'] = float(pi_width)
        results[f'test_winkler_at_{pct}_scaled'] = float(winkler)
        results[f'test_pi_width_at_{pct}_original'] = float(pi_width_original)
        results[f'test_winkler_at_{pct}_original'] = float(winkler_original)
        
        if hasattr(dataset, 'sensor_names') and dataset.sensor_names:
            for i, sensor_name in enumerate(dataset.sensor_names):
                sensor_pi_width = np.mean(upper_orig[:, i] - lower_orig[:, i])
                results[f'test_pi_width_{sensor_name}_original'] = float(sensor_pi_width)
                
                sensor_winkler = np.mean(
                    (upper_orig[:, i] - lower_orig[:, i]) + 
                    (2/alpha) * np.maximum(0, lower_orig[:, i] - y_original[:, i]) +
                    (2/alpha) * np.maximum(0, y_original[:, i] - upper_orig[:, i])
                )
                results[f'test_winkler_{sensor_name}_original'] = float(sensor_winkler)
        
        if alpha == 0.1:  # 90% CI results only
            print(f"{pct}% Confidence Interval:")
            print(f"  Coverage: {coverage:.4f}")
            print(f"  Delta Coverage: {delta_cov:.4f}")
            print(f"  PI Width (Scaled): {pi_width:.4f}")
            print(f"  Winkler Score (Scaled): {winkler:.4f}")
            print(f"  PI Width (Original): {pi_width_original:.4f}")
            print(f"  Winkler Score (Original): {winkler_original:.4f}")
            print(f"  Per-sensor metrics computed for {len(dataset.sensor_names) if hasattr(dataset, 'sensor_names') else 0} sensors")

    ########################################
    # Logging                              #
    ########################################

    # Initialize loggers
    loggers = []
    
    # Always use TensorBoard
    tb_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)
    loggers.append(tb_logger)
    
    # Add wandb logger if enabled
    if cfg.get('use_wandb', False) and WANDB_AVAILABLE:
        wandb_logger = WandbLogger(
            project=cfg.wandb.get('project', 'corel-ics'),
            entity=cfg.wandb.get('entity', None),
            name=f"{cfg.dataset.name}_nexcp_baseline",
            tags=cfg.wandb.get('tags', ['nexcp', 'baseline', cfg.dataset.name]),
            notes=cfg.wandb.get('notes', 'NexCP baseline evaluation'),
            save_dir=cfg.run.dir,
            config=cfg
        )
        loggers.append(wandb_logger)
        
        # Log results to wandb
        wandb_logger.experiment.log(results)
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_path = os.path.join(cfg.run.dir, 'nexcp_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    print("NexCP evaluation completed successfully!")
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run NexCP baseline evaluation')
    parser.add_argument('--src_dir', type=str, required=True,
                        help='Path to the source directory containing residuals.h5 and config.yaml')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2],
                        help='List of alpha values for prediction intervals')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='Exponential decay rate for NexCP weights')
    parser.add_argument('--window_size', type=int, default=1000,
                        help='Window size for NexCP (number of recent residuals to use)')
    parser.add_argument('--val_len', type=float, default=0.2,
                        help='Fraction of calibration data to use for validation')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Whether to use wandb logging')
    parser.add_argument('--wandb_project', type=str, default='corel-baselines',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity name')
    parser.add_argument('--run_dir', type=str, default='logs/baseline/nexcp',
                        help='Directory to save results')
    parser.add_argument('--run_name', type=str, default='nexcp_evaluation',
                        help='Name for this run')
    
    args = parser.parse_args()
    
    # Load source config to get dataset name
    with open(os.path.join(args.src_dir, "config.yaml"), 'r') as fp:
        src_config = yaml.load(fp, Loader=yaml.FullLoader)
    
    dataset_name = src_config["dataset"]["name"]
    print(f"Detected dataset: {dataset_name}")
    
    # Create config from arguments
    cfg = DictConfig({
        'src_dir': args.src_dir,
        'alphas': args.alphas,
        'decay_rate': args.decay_rate,
        'window_size': args.window_size,
        'val_len': args.val_len,
        'use_wandb': args.use_wandb,
        'wandb': {
            'project': args.wandb_project,
            'entity': args.wandb_entity
        },
        'run': {
            'dir': args.run_dir,
            'name': args.run_name
        },
        'dataset': {
            'name': dataset_name
        }
    })
    
    # Create run directory
    os.makedirs(cfg.run.dir, exist_ok=True)
    
    res = run_experiment(cfg)
    logger.info(res)