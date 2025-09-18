import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tsl.datasets.prototypes import TabularDataset
from tsl.data.datamodule.splitters import TemporalSplitter
import sys
sys.path.append('../..')
from utils.attack_utils import is_actuator, TEP_COLUMN_NAMES

class TEPDataset(TabularDataset):
    """
    TEP Dataset - METR-LA Format Compatible

    This dataset creates TEP data that exactly matches METR-LA's structure:
    - DatetimeIndex with uniform 10s sampling
    - Columns = 41 sensor names (targets only)  
    - Values = float (n_channels = 1)
    - Actuators provided separately as exogenous features

    Key transformations:
    1. Load TEP CSV files
    2. Downsample from 3s to 10s intervals
    3. Separate sensors (targets) from actuators (exogenous)
    4. Create TSL-compatible DataFrame format
    """

    def __init__(self, 
                 root=None,
                 csv_file='TEP_train.csv',
                 sample_rate=10,
                 split='train',
                 name='TEP_Clean',
                 **kwargs):
        
        # Store parameters  
        self.csv_file = csv_file
        self.sample_rate = sample_rate
        self.split = split
        
        # Set default root if not provided
        if root is None:
            root = os.path.join(os.path.dirname(__file__), '../../data/TEP')
        self.root = root
        
        # Load and process data to METR-LA format
        self._load_and_process_data()
        
        # Initialize TabularDataset with clean data (METR-LA style)
        super().__init__(
            target=self._target_df,
            mask=self._mask_df,
            covariates={},
            name=name,
            **kwargs
        )
        
        # Store actuator data AFTER super().__init__() to avoid conversion
        if hasattr(self, 'actuator_names') and self.actuator_names:
            self._actuator_data = self._actuator_df
        
    def _load_and_process_data(self):
        """Load CSV and transform to METR-LA format"""
        
        csv_path = os.path.join(self.root, self.csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Step 1: Load raw CSV data with proper data type handling
        try:
            df_raw = pd.read_csv(csv_path, low_memory=False)
            
            # Check for mixed types and convert to numeric
            for col in df_raw.columns:
                if df_raw[col].dtype == 'object':
                    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
            # Remove any rows with NaN values after conversion
            df_raw = df_raw.dropna()
                
        except Exception as e:
            # Fallback: load with explicit dtype specification
            df_raw = pd.read_csv(csv_path, dtype=str, low_memory=False)
            
            # Convert all columns to numeric, coercing errors to NaN
            for col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
            # Remove rows with NaN values
            df_raw = df_raw.dropna()
        
        # Step 2: Create timestamps (1-second intervals for TEP)
        start_time = datetime(2020, 1, 1)
        timestamps = [start_time + timedelta(seconds=1*i) for i in range(len(df_raw))]
        df_raw.index = pd.DatetimeIndex(timestamps)
        
        # Step 3: Downsample to target interval (e.g., 10s)
        try:
            df_resampled = df_raw.resample(f'{self.sample_rate}s').mean()
            df_resampled.dropna(inplace=True)
        except Exception as e:
            # Alternative: manual downsampling
            step = self.sample_rate
            df_resampled = df_raw.iloc[::step].copy()
        
        # Step 4: Classify sensors vs actuators using attack_utils
        self.sensor_names = []
        self.actuator_names = []
        
        for col in df_resampled.columns:
            if is_actuator('TEP', col):
                self.actuator_names.append(col)
            else:
                self.sensor_names.append(col)
        
        # Step 5: Create target DataFrame (sensors only) - METR-LA format
        self._target_df = df_resampled[self.sensor_names].copy()
        self._target_df = self._target_df.astype(np.float32)
        
        # Step 6: Create mask (all valid for clean data)
        self._mask_df = pd.DataFrame(
            data=np.ones_like(self._target_df.values, dtype=bool),
            index=self._target_df.index,
            columns=self._target_df.columns
        )
        
        # Step 7: Create actuator covariates (exogenous features)
        self._covariates = {}
        if self.actuator_names:
            self._actuator_df = df_resampled[self.actuator_names].copy()
            self._actuator_df = self._actuator_df.astype(np.float32)
            self._covariates['actuators'] = self._actuator_df
        
        # Store for backward compatibility
        self.data = self._target_df.values
        self.mask = self._mask_df.values.astype(np.uint8)
        self._df_index = self._target_df.index
        self.sensor_cols = self.sensor_names
        self.actuator_cols = self.actuator_names
        
    def load_raw(self):
        """Load raw data - required by TSL TabularDataset interface"""
        eval_mask = self.mask.copy()
        return self.data, self.mask, eval_mask
    
    def load(self, impute_nans=None):
        """Load processed data - required by TSL TabularDataset interface"""
        if impute_nans is None:
            impute_nans = self.impute_nans
            
        data, mask, eval_mask = self.load_raw()
        
        if impute_nans and np.any(np.isnan(data)):
            data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
            
        return data, mask, eval_mask
    
    @property
    def n_nodes(self):
        """Number of sensors (nodes in the graph)"""
        return len(self.sensor_names)
    
    @property
    def n_channels(self):
        """Number of channels per node (1 for univariate time series)"""
        return 1
    
    def dataframe(self):
        """Return target data as DataFrame - METR-LA compatible"""
        return self._target_df
    
    def get_connectivity(self, 
                        method='correlation',
                        threshold=0.3,
                        include_self=False,
                        layout='edge_index',
                        **kwargs):
        """Generate connectivity/adjacency matrix for sensors"""
        n_nodes = self.n_nodes
        
        if method == 'correlation':
            corr_matrix = self._target_df.corr().abs().values
            np.fill_diagonal(corr_matrix, 0 if not include_self else 1)
            
            edge_list = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if (i != j or include_self) and not np.isnan(corr_matrix[i, j]) and corr_matrix[i, j] > threshold:
                        edge_list.append([i, j])
                        
        elif method == 'full':
            edge_list = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j or include_self:
                        edge_list.append([i, j])
        else:
            raise ValueError(f"Unknown connectivity method: {method}")
        
        if len(edge_list) == 0:
            for i in range(min(n_nodes-1, 5)):
                edge_list.extend([[i, i+1], [i+1, i]])
        
        if layout == 'edge_index':
            edge_index = np.array(edge_list).T if edge_list else np.array([[], []]).astype(int)
            return edge_index, None
        else:
            raise ValueError(f"Unsupported layout: {layout}")
    
    def get_splitter(self, 
                     method='temporal',
                     val_len=0.2,
                     test_len=0.2,
                     **kwargs):
        """Get data splitter for train/validation/test splits"""
        if method == 'temporal':
            return TemporalSplitter(val_len=val_len, test_len=test_len)
        else:
            raise ValueError(f"Unknown splitting method: {method}")
    
    def _columns_multiindex(self):
        """Create multiindex for columns - used by COREL for residuals processing"""
        tuples = [(sensor, 0) for sensor in self.sensor_names]
        return pd.MultiIndex.from_tuples(tuples, names=['nodes', 'channels'])
    
    def get_covariates(self):
        """Get actuator data as covariates"""
        if hasattr(self, '_actuator_df'):
            return {'actuators': self._actuator_df}
        return {}
    
    def get_benign_data_for_scaling(self):
        """Get benign-only data for proper scaler fitting"""
        return self.dataframe()
    
    def get_sensor_statistics(self):
        """Get per-sensor statistics for analysis"""
        df = self.dataframe()
        stats = {}
        
        for sensor in self.sensor_names:
            sensor_data = df[sensor]
            stats[sensor] = {
                'mean': float(sensor_data.mean()),
                'std': float(sensor_data.std()),
                'min': float(sensor_data.min()),
                'max': float(sensor_data.max()),
                'missing_pct': float(sensor_data.isna().mean() * 100)
            }
        
        return stats


def create_tep_dataset(data_dir=None, sample_rate=10, **kwargs):
    """Create a TEPDataset instance with sensible defaults"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../../data/TEP')
    
    return TEPDataset(
        root=data_dir,
        sample_rate=sample_rate,
        **kwargs
    )
