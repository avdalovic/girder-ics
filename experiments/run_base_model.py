import os
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Add wandb import
try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from lib.datasets.swat import SWaTDataset
from lib.engines.base_predictor import BasePredictor
from lib.utils.data_utils import create_residuals_frame
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import torch_metrics

from lib import config
from tsl.nn.models import TransformerModel
from tsl.utils.casting import torch_to_numpy

from lib.nn.base import RNNModel, STGNNModel  # , MLPModel
from tsl.data import BatchMap, BatchMapItem
from lib.datasets.tep import TEPDataset
from lib.datasets.wadi import WADIDataset

# Optional imports for datasets that may not be available
try:
    from lib.datasets.gpvar import GPVARDataset
except ImportError:
    GPVARDataset = None

try:
    from lib.datasets.air_quality import AirQuality
except ImportError:
    AirQuality = None


def get_model_class(model_str):
    # Basic models  #####################################################
    if model_str == 'rnn':
        model = RNNModel
    elif model_str == 'transformer':
        model = TransformerModel
    elif model_str == 'stgnn':
        model = STGNNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_cfg):
    name = dataset_cfg.name
    
    if name == 'la':
        dataset = MetrLA()
    elif name == 'air':
        if AirQuality is None:
            raise ImportError("AirQuality dataset not available. Please ensure lib.datasets.air_quality is properly installed.")
        dataset = AirQuality()
    elif name == 'gpvar':
        if GPVARDataset is None:
            raise ImportError("GPVARDataset not available. Please ensure lib.datasets.gpvar is properly installed.")
        dataset = GPVARDataset(**dataset_cfg.hparams, p_max=0)
    elif name == 'swat':
        # Get dataset parameters
        sample_rate = dataset_cfg.get('sample_rate', 10)
        data_dir = dataset_cfg.get('data_dir', None)
        dataset = SWaTDataset(root=data_dir, sample_rate=sample_rate)
        print(f"SWaT dataset: {dataset.n_nodes} sensors, {len(dataset.actuator_names)} actuators (used as covariates)")
    elif name == 'tep':
        # Get dataset parameters
        sample_rate = dataset_cfg.get('sample_rate', 10)
        data_dir = dataset_cfg.get('data_dir', None)
        dataset = TEPDataset(root=data_dir, sample_rate=sample_rate)
        print(f"TEP dataset: {dataset.n_nodes} sensors, {len(dataset.actuator_names)} actuators (used as covariates)")
    elif name == 'wadi':  # Add this case
        sample_rate = dataset_cfg.get('sample_rate', 10)
        data_dir = dataset_cfg.get('data_dir', None)
        dataset = WADIDataset(root=data_dir, sample_rate=sample_rate)
        print(f"WADI dataset: {dataset.n_nodes} sensors, {len(dataset.actuator_names)} actuators (used as covariates)")
    else:
        raise ValueError(f"Dataset {name} not available.")
    
    return dataset


def run_experiment(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset)

    covariates = dict()
    if cfg.get('add_exogenous'):
        assert cfg.dataset.name not in {'gpvar'}
        # Check if dataset has datetime methods before calling them
        if hasattr(dataset, 'datetime_encoded') and hasattr(dataset, 'datetime_onehot'):
            # encode time of the day and use it as exogenous variable
            day_sin_cos = dataset.datetime_encoded('day').values
            weekdays = dataset.datetime_onehot('weekday').values
            covariates['u'] = np.concatenate([day_sin_cos, weekdays], axis=-1)
        else:
            pass  # Skip time encoding silently

    if cfg.dataset.name in {'gpvar', 'toy', 'mso'}:
        ds_index = pd.Index(dataset.index)
        index_type = 'scalar'
    else:
        ds_index = dataset.index
        index_type = 'datetime'



    torch_dataset = SpatioTemporalDataset(index=ds_index,
                                          target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride,
                                          delay=cfg.get('delay', 0))

    if cfg.apply_scaler is False:
        transform = {}
    else:
        scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
        transform = {
            'target': StandardScaler(axis=scale_axis)
        }

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    ########################################
    # training                             #
    ########################################

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=dm.train_slice)
    dm.torch_dataset.set_connectivity(adj)

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = 0
    if 'u' in torch_dataset:
        d_exog += torch_dataset.input_map.u.shape[-1]

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        window=torch_dataset.window,
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   'mre': torch_metrics.MaskedMRE()}

    if cfg.dataset.name in ['la', 'bay']:
        multistep_metrics = {
            'mape': torch_metrics.MaskedMAPE(),
            'mae@15': torch_metrics.MaskedMAE(at=2),
            'mae@30': torch_metrics.MaskedMAE(at=5),
            'mae@60': torch_metrics.MaskedMAE(at=11),
        }
        log_metrics.update(multistep_metrics)

    # setup predictor
    predictor = BasePredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scale_target=cfg.scale_target,
    )

    ########################################
    # logging options                      #
    ########################################

    run_args = exp.get_config_dict()
    run_args['model']['trainable_parameters'] = predictor.trainable_parameters

    # Initialize loggers
    loggers = []
    
    # Always use TensorBoard
    tb_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)
    loggers.append(tb_logger)
    
    # Add wandb logger if enabled
    if cfg.get('use_wandb', False) and WANDB_AVAILABLE:
        # Dataset info for wandb logging
        dataset_info = {
            'dataset_name': cfg.dataset.name,
            'n_nodes': dataset.n_nodes,
            'n_channels': dataset.n_channels if hasattr(dataset, 'n_channels') else 1,
            'data_shape': str(dataset.dataframe().shape),
            'sample_rate': cfg.dataset.get('sample_rate', 'unknown'),
            'window_size': cfg.window,
            'horizon': cfg.horizon,
            'stride': cfg.stride,
            'delay': cfg.get('delay', 0),
            'scale_axis': cfg.get('scale_axis', 'graph'),
            'connectivity_method': cfg.dataset.connectivity.get('method', 'unknown'),
            'connectivity_threshold': cfg.dataset.connectivity.get('threshold', 'unknown'),
        }
        
        # Add SWAT-specific info
        if cfg.dataset.name == 'swat' and hasattr(dataset, 'sensor_names'):
            dataset_info.update({
                'n_sensors': len(dataset.sensor_names),
                'n_actuators': len(dataset.actuator_names) if hasattr(dataset, 'actuator_names') else 0,
                'sensor_list': dataset.sensor_names[:10],  # First 10 sensors
                'actuator_list': dataset.actuator_names[:10] if hasattr(dataset, 'actuator_names') else [],
            })
        
        wandb_logger = WandbLogger(
            project=cfg.wandb.get('project', 'corel-ics'),
            entity=cfg.wandb.get('entity', None),
            name=cfg.wandb.get('name', f"{cfg.dataset.name}_{cfg.model.name}_base"),
            tags=cfg.wandb.get('tags', ['base_model', cfg.dataset.name, cfg.model.name]),
            notes=cfg.wandb.get('notes', 'Base model training for conformal prediction pipeline'),
            save_dir=cfg.run.dir,
            config={**run_args, **dataset_info}  # Log all config + dataset info
        )
        loggers.append(wandb_logger)
        
        # Log additional dataset statistics to wandb
        if hasattr(dataset, 'get_sensor_statistics'):
            try:
                sensor_stats = dataset.get_sensor_statistics()
                wandb_logger.experiment.log({
                    "dataset/sensor_statistics": wandb.Table(
                        columns=["sensor", "mean", "std", "min", "max", "missing_pct"],
                        data=[[k, v['mean'], v['std'], v['min'], v['max'], v['missing_pct']] 
                              for k, v in list(sensor_stats.items())[:20]]  # First 20 sensors
                    )
                })
            except Exception as e:
                pass
        
        # Log connectivity visualization if available
        try:
            adj = dataset.get_connectivity(**cfg.dataset.connectivity)
            if isinstance(adj, tuple):
                edge_index, _ = adj
                if edge_index.shape[1] > 0:
                    wandb_logger.experiment.log({
                        "dataset/n_edges": edge_index.shape[1],
                        "dataset/connectivity_density": edge_index.shape[1] / (dataset.n_nodes * dataset.n_nodes)
                    })
        except Exception as e:
            pass
    
    exp_logger = loggers[0] if len(loggers) == 1 else loggers

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    val_batches = .25

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      limit_val_batches=val_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    predictor.freeze()

    ########################################
    # compute residuals                    #
    ########################################

    output = trainer.predict(predictor, dataloaders=[dm.val_dataloader(),
                                                     dm.test_dataloader()])  # has size [[len_val], [len_test]]
    output = predictor.collate_prediction_outputs(output)  # has size [len_val + len_test]
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'], output.get('mask', None))

    residuals = (y_true - y_hat).squeeze(-1)
    calib_indices = dm.valset.indices
    test_indices = dm.testset.indices

    # input covariates
    val_index = dm.torch_dataset.data_timestamps(calib_indices)['horizon']
    test_index = dm.torch_dataset.data_timestamps(test_indices)['horizon']

    # Remove residuals at the beginning and at the end of the time series that have less than window and horizon time steps respectively
    # The second dimension in the dataframe is nodes x horizon
    # Input: [samples, nodes, horizon]
    # Output: [filtered_samples, nodes x horizon]
    lagged_residuals = create_residuals_frame(residuals,
                                              np.concatenate([val_index, test_index], axis=0),
                                              channels_index=dataset._columns_multiindex(),
                                              horizon=cfg.horizon,
                                              idx_type=index_type)

    # concatenate indices and take the one corresponding to the last time step
    target_index = np.concatenate([val_index, test_index], axis=0)[:, 0]

    # combinations of nodex x horizon
    col_idx = [(c[0], f'{c[1]}_{i}') for c in dataset._columns_multiindex() for i in range(cfg.horizon)]

    # create a dataframe with the residuals arranged in shape [samples, nodes x horizon]
    target_df = pd.DataFrame(data=rearrange(residuals, "t h n ... -> t (n ... h)"),
                             index=target_index,
                             columns=pd.MultiIndex.from_tuples(col_idx))
    if mask is not None:
        mask_df = pd.DataFrame(data=rearrange(mask, "t h n ... -> t (n ... h)"),
                               index=target_index,
                               columns=pd.MultiIndex.from_tuples(col_idx))
    else:
        mask_df = None

    # filter calib and test indices
    valid_input_indices = torch_dataset.index.get_indexer(lagged_residuals.index)
    valid_target_indices = torch_dataset.index.get_indexer(target_index)

    ########################################
    # save residuals for CP                #
    ########################################
    if cfg.save_outputs:
        lagged_residuals.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='input')
        target_df.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='target')
        if mask_df is not None:
            mask_df.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='target_mask')
        np.savez(os.path.join(cfg.run.dir,'indices.npz'),
                 calib_indices=calib_indices,
                 test_indices=test_indices,
                 valid_input_indices=valid_input_indices,
                 valid_target_indices=valid_target_indices)

    return 'done'


if __name__ == '__main__':
    exp = Experiment(run_fn=run_experiment, config_path='../config/training',
                     config_name='default')
    res = exp.run()
    logger.info(res)