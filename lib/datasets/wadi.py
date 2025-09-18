import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tsl.datasets.prototypes import TabularDataset
from tsl.data.datamodule.splitters import TemporalSplitter
import sys
sys.path.append('../..')
from utils.attack_utils import is_actuator

class WADIDataset(TabularDataset):
    """
    WADI Dataset - METR-LA Format Compatible

    This dataset creates WADI data that exactly matches METR-LA's structure:
    - DatetimeIndex with uniform 10s sampling
    - Columns = sensor names (targets only)  
    - Values = float (n_channels = 1)
    - Actuators provided separately as exogenous features

    Key transformations:
    1. Load WADI CSV files
    2. Downsample to 10s intervals
    3. Separate sensors (targets) from actuators (exogenous)
    4. Create TSL-compatible DataFrame format
    """

    def __init__(self, 
                 root=None,
                 csv_file='WADI_train.csv',
                 sample_rate=10,  # Downsample interval in seconds
                 split='train',
                 name='WADI_Clean',
                 **kwargs):
        
        # Store parameters  
        self.csv_file = csv_file
        self.sample_rate = sample_rate
        self.split = split
        
        # Set default root if not provided
        if root is None:
            root = os.path.join(os.path.dirname(__file__), '../../data/WADI')
        self.root = root
        
        # Load and process data to METR-LA format
        self._load_and_process_data()
        
        # Initialize TabularDataset with clean data (METR-LA style)
        super().__init__(
            target=self._target_df,      # [T, N] DataFrame like METR-LA
            mask=self._mask_df,          # [T, N] boolean mask  
            covariates={},               # Start with empty covariates
            name=name,
            **kwargs
        )
        
        # Store actuator data AFTER super().__init__() to avoid conversion
        if hasattr(self, 'actuator_names') and self.actuator_names:
            self._actuator_data = self._actuator_df
        
        print(f"WADI dataset: {self.n_nodes} sensors, {len(self.actuator_names)} actuators (used as covariates)")
        
    def _load_and_process_data(self):
        """Load CSV and transform to METR-LA format"""
        
        csv_path = os.path.join(self.root, self.csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Step 1: Load raw CSV data
        df_raw = pd.read_csv(csv_path, low_memory=False)
        
        # Step 2: Parse timestamps and set as index 
        # Combine Date and Time columns to create proper timestamps
        if 'Date' in df_raw.columns and 'Time' in df_raw.columns:
            # Create datetime string from Date and Time
            df_raw['DateTime'] = df_raw['Date'].astype(str) + ' ' + df_raw['Time'].astype(str)
            # Parse datetime with flexible format
            df_raw['DateTime'] = pd.to_datetime(df_raw['DateTime'], errors='coerce')
            
            # Remove rows with NaT timestamps
            nat_count = df_raw['DateTime'].isna().sum()
            if nat_count > 0:
                df_raw = df_raw.dropna(subset=['DateTime'])
            
            df_raw.set_index('DateTime', inplace=True)
            df_raw.sort_index(inplace=True)
        else:
            # Fallback: create timestamps assuming 1-second intervals
            start_time = datetime(2020, 1, 1)
            timestamps = [start_time + timedelta(seconds=1*i) for i in range(len(df_raw))]
            df_raw.index = pd.DatetimeIndex(timestamps)
        
        # Step 3: Remove non-sensor columns (Row, Date, Time, Attack) and plant operation columns
        columns_to_remove = ['Row', 'Date', 'Time', 'Attack', 'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW']
        for col in columns_to_remove:
            if col in df_raw.columns:
                df_raw.drop(col, axis=1, inplace=True)
        
        # Step 4: Remove completely empty columns first (before numeric conversion)
        empty_cols = []
        for col in df_raw.columns:
            # Check if column is all NaN or all empty strings
            if df_raw[col].isna().all() or (df_raw[col] == '').all():
                empty_cols.append(col)
        
        if empty_cols:
            df_raw = df_raw.drop(columns=empty_cols)
        
        # Step 5: Convert remaining columns to numeric
        for col in df_raw.columns:
            if df_raw[col].dtype == 'object':
                # Replace empty strings with NaN before conversion
                df_raw[col] = df_raw[col].replace('', np.nan)
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        # Remove any rows with NaN values after conversion
        df_raw = df_raw.dropna()
        
        # Step 6: Downsample to target interval (e.g., 10s)
        try:
            df_resampled = df_raw.resample(f'{self.sample_rate}s').mean()
            df_resampled.dropna(inplace=True)
        except Exception as e:
            step = self.sample_rate
            df_resampled = df_raw.iloc[::step].copy()
        
        # Step 7: Classify sensors vs actuators using attack_utils
        self.sensor_names = []
        self.actuator_names = []
        
        for col in df_resampled.columns:
            if is_actuator('WADI', col):
                self.actuator_names.append(col)
            else:
                self.sensor_names.append(col)
        
        # Step 8: Create target DataFrame (sensors only) - METR-LA format
        self._target_df = df_resampled[self.sensor_names].copy()
        self._target_df = self._target_df.astype(np.float32)
        
        # Step 9: Create mask (all valid for clean data)
        self._mask_df = pd.DataFrame(
            data=np.ones_like(self._target_df.values, dtype=bool),
            index=self._target_df.index,
            columns=self._target_df.columns
        )
        
        # Step 10: Create actuator covariates (exogenous features)
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


def create_wadi_dataset(data_dir=None, sample_rate=10, **kwargs):
    """Create a WADIDataset instance with sensible defaults"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../../data/WADI')
    
    return WADIDataset(
        root=data_dir,
        sample_rate=sample_rate,
        **kwargs
    )
