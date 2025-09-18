import pandas as pd
import numpy as np
import os
from datetime import datetime
from tsl.datasets.prototypes import TabularDataset
from tsl.data.datamodule.splitters import TemporalSplitter
import sys
sys.path.append('../..')
from utils.attack_utils import is_actuator

class SWaTDataset(TabularDataset):
    """
    Clean SWaT Dataset - METR-LA Format Compatible

    This dataset creates SWAT data that exactly matches METR-LA's structure:
    - DatetimeIndex with uniform 10s sampling
    - Columns = 25 sensor names (targets only)  
    - Values = float (n_channels = 1)
    - Actuators provided separately as exogenous features

    Key transformations:
    1. Remove first 6 hours (21,600 points) for testbed stabilization
    2. Downsample from 1s to 10s intervals
    3. Separate sensors (targets) from actuators (exogenous)
    4. Create TSL-compatible DataFrame format
    """

    def __init__(self, 
             root=None,
             csv_file='SWATv0_train.csv',
             sample_rate=10,  # Downsample interval in seconds
             stabilization_hours=6,  # Remove first N hours
             split='train',
             name='SWaT_Clean',
             **kwargs):
        
        # Store parameters  
        self.csv_file = csv_file
        self.sample_rate = sample_rate  # Now means downsample interval
        self.stabilization_hours = stabilization_hours
        self.split = split
        
        # Set default root if not provided
        if root is None:
            root = os.path.join(os.path.dirname(__file__), '../../data/SWAT')
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
            self._actuator_data = self._target_df  # Store the actuator DataFrame directly
        
    def _load_and_process_data(self):
        """Load CSV and transform to METR-LA format"""
        
        csv_path = os.path.join(self.root, self.csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Step 1: Load raw CSV data
        df_raw = pd.read_csv(csv_path)
        
        # Step 2: Parse timestamps and set as index
        # Clean timestamp strings - remove extra spaces and normalize format
        df_raw['Timestamp'] = df_raw['Timestamp'].str.strip()
        # Use flexible parsing with dayfirst=True for DD/MM/YYYY format
        df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], dayfirst=True, errors='coerce')
        
        # Remove rows with NaT timestamps
        nat_count = df_raw['Timestamp'].isna().sum()
        if nat_count > 0:
            df_raw = df_raw.dropna(subset=['Timestamp'])
        
        df_raw.set_index('Timestamp', inplace=True)
        df_raw.sort_index(inplace=True)
        
        # Step 3: Remove attack labels and non-sensor columns
        if 'Normal/Attack' in df_raw.columns:
            df_raw.drop('Normal/Attack', axis=1, inplace=True)
        
        # Step 4: Remove stabilization period (only for training data, not for test/attack data)
        if self.split == 'train':
            stabilization_seconds = self.stabilization_hours * 3600
            
            start_time = df_raw.index[0]
            cutoff_time = start_time + pd.Timedelta(seconds=stabilization_seconds)
            df_stable = df_raw[df_raw.index >= cutoff_time].copy()
        else:
            df_stable = df_raw.copy()
        
        # Step 5: Downsample to target interval (e.g., 10s)
        df_resampled = df_stable.resample(f'{self.sample_rate}s').mean()
        df_resampled.dropna(inplace=True)  # Remove any NaN rows from resampling
        
        # Step 6: Classify sensors vs actuators using attack_utils
        self.sensor_names = []
        self.actuator_names = []
        
        for col in df_resampled.columns:
            if is_actuator('SWAT', col):
                self.actuator_names.append(col)
            else:
                self.sensor_names.append(col)
        
        # Step 7: Create target DataFrame (sensors only) - METR-LA format
        self._target_df = df_resampled[self.sensor_names].copy()
        self._target_df = self._target_df.astype(np.float32)
        
        # Step 8: Create mask (all valid for clean data)
        self._mask_df = pd.DataFrame(
            data=np.ones_like(self._target_df.values, dtype=bool),
            index=self._target_df.index,
            columns=self._target_df.columns
        )
        
        # Step 9: Create actuator covariates (exogenous features)
        self._covariates = {}
        if self.actuator_names:
            actuator_df = df_resampled[self.actuator_names].copy()
            actuator_df = actuator_df.astype(np.float32)
            self._covariates['actuators'] = actuator_df
            self._actuator_data = actuator_df  # Fix: use _actuator_data with underscore
        
        # Store for backward compatibility
        self.data = self._target_df.values  # [T, N] array
        self.mask = self._mask_df.values.astype(np.uint8)
        self._df_index = self._target_df.index
        self.sensor_cols = self.sensor_names  # Backward compatibility
        self.actuator_cols = self.actuator_names  # Backward compatibility
        
    def load_raw(self):
        """
        Load raw data - required by TSL TabularDataset interface
        
        Returns:
            tuple: (data, mask, eval_mask)
                - data: np.ndarray of shape [T, N] with sensor readings
                - mask: np.ndarray of shape [T, N] indicating valid data points
                - eval_mask: np.ndarray of shape [T, N] for evaluation (same as mask for now)
        """
        eval_mask = self.mask.copy()  # Use same mask for evaluation
        return self.data, self.mask, eval_mask
    
    def load(self, impute_nans=None):
        """
        Load processed data - required by TSL TabularDataset interface
        
        Args:
            impute_nans (bool): Whether to impute NaN values
            
        Returns:
            tuple: (data, mask, eval_mask)
        """
        if impute_nans is None:
            impute_nans = self.impute_nans
            
        data, mask, eval_mask = self.load_raw()
        
        if impute_nans and np.any(np.isnan(data)):
            # Simple forward fill imputation
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
        """
        Return target data as DataFrame - METR-LA compatible
        
        Returns:
            pd.DataFrame: [T, N] with DatetimeIndex and sensor columns
                         Exactly like METR-LA format
        """
        return self._target_df
    
    def get_connectivity(self, 
                        method='correlation',
                        threshold=0.3,
                        include_self=False,
                        layout='edge_index',
                        **kwargs):
        """
        Generate connectivity/adjacency matrix for sensors
        
        Args:
            method (str): Method for computing connectivity ('correlation', 'distance', 'full')
            threshold (float): Threshold for edge creation
            include_self (bool): Whether to include self-loops
            layout (str): Output format ('edge_index', 'dense', 'sparse')
            
        Returns:
            tuple or np.ndarray: Connectivity information in requested format
        """
        n_nodes = self.n_nodes
        
        if method == 'correlation':
            # Use correlation between sensors
            corr_matrix = self._target_df.corr().abs().values
            np.fill_diagonal(corr_matrix, 0 if not include_self else 1)
            
            # Create edges based on threshold
            edge_list = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if (i != j or include_self) and not np.isnan(corr_matrix[i, j]) and corr_matrix[i, j] > threshold:
                        edge_list.append([i, j])
                        
        elif method == 'distance':
            # Random distance-based connectivity (placeholder)
            np.random.seed(42)
            distances = np.random.rand(n_nodes, n_nodes)
            distances = (distances + distances.T) / 2  # Make symmetric
            np.fill_diagonal(distances, 0 if not include_self else 1)
            
            edge_list = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if (i != j or include_self) and distances[i, j] < threshold:
                        edge_list.append([i, j])
                        
        elif method == 'full':
            # Fully connected graph
            edge_list = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j or include_self:
                        edge_list.append([i, j])
        else:
            raise ValueError(f"Unknown connectivity method: {method}")
        
        # Ensure we have at least some connectivity
        if len(edge_list) == 0:
            for i in range(min(n_nodes-1, 5)):  # Connect first few nodes in chain
                edge_list.extend([[i, i+1], [i+1, i]])
        
        # Return in requested format
        if layout == 'edge_index':
            edge_index = np.array(edge_list).T if edge_list else np.array([[], []]).astype(int)
            return edge_index, None  # (edge_index, edge_weights)
        elif layout == 'dense':
            adj_matrix = np.zeros((n_nodes, n_nodes))
            for i, j in edge_list:
                adj_matrix[i, j] = 1.0
            return adj_matrix
        else:
            raise ValueError(f"Unsupported layout: {layout}")
    
    def get_splitter(self, 
                     method='temporal',
                     val_len=0.2,
                     test_len=0.2,
                     **kwargs):
        """
        Get data splitter for train/validation/test splits
        
        Args:
            method (str): Splitting method ('temporal')
            val_len (float): Fraction of data for validation
            test_len (float): Fraction of data for test
            
        Returns:
            Splitter: TSL splitter object
        """
        if method == 'temporal':
            return TemporalSplitter(val_len=val_len, test_len=test_len)
        else:
            raise ValueError(f"Unknown splitting method: {method}")
    
    def _columns_multiindex(self):
        """
        Create multiindex for columns - used by COREL for residuals processing
        
        Returns:
            pd.MultiIndex: MultiIndex with (sensor_name, channel_index) tuples
        """
        # Create MultiIndex like METR-LA: (node_name, channel_index)
        tuples = [(sensor, 0) for sensor in self.sensor_names]  # Channel index 0 for univariate
        return pd.MultiIndex.from_tuples(tuples, names=['nodes', 'channels'])
    
    def get_covariates(self):
        """
        Get actuator data as covariates
        
        Returns:
            dict: Dictionary with actuator data for use as covariates
        """
        if hasattr(self, '_actuator_data'):
            return {
                'actuators': self._actuator_data  # DataFrame
            }
        return {}
    
    def get_benign_data_for_scaling(self):
        """
        Get benign-only data for proper scaler fitting
        
        For SWAT training data, all data is benign since we're using SWATv0_train.csv
        For attack data, this would filter out attack periods.
        
        Returns:
            pd.DataFrame: Benign sensor data for scaler fitting
        """
        return self.dataframe()  # All training data is benign
    
    def get_sensor_statistics(self):
        """
        Get per-sensor statistics for analysis
        
        Returns:
            dict: Statistics per sensor including mean, std, min, max
        """
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


# Convenience function for easy dataset creation
def create_swat_dataset(data_dir=None, sample_rate=10, **kwargs):
    """
    Create a SWaTDataset instance with sensible defaults
    
    Args:
        data_dir (str): Path to directory containing SWAT CSV files
        sample_rate (int): Sample every N seconds
        **kwargs: Additional arguments for SWaTDataset
        
    Returns:
        SWaTDataset: Configured dataset instance
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../../data/SWAT')
    
    return SWaTDataset(
        root=data_dir,
        sample_rate=sample_rate,
        **kwargs
    )


if __name__ == "__main__":
    # Test the clean dataset
    print("=== Testing Clean SWaT Dataset ===")
    
    dataset = create_swat_dataset(sample_rate=10)
    
    print(f"\n=== Dataset Info ===")
    print(f"Shape: {dataset.dataframe().shape}")
    print(f"Sensors: {dataset.n_nodes}")
    print(f"Channels per sensor: {dataset.n_channels}")
    
    print(f"\n=== DataFrame Format (like METR-LA) ===")
    df = dataset.dataframe()
    print(f"Index: {df.index[:3]}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample values:\n{df.iloc[:3, :3]}")
    
    print(f"\n=== Covariates ===")
    covariates = dataset.get_covariates()
    if 'actuators' in covariates:
        act_df = covariates['actuators']
        print(f"Actuator shape: {act_df.shape}")
        print(f"Actuator columns: {list(act_df.columns[:5])}...")
    
    # Test connectivity
    edge_index, _ = dataset.get_connectivity(method='correlation', threshold=0.3)
    print(f"\n=== Connectivity ===")
    print(f"Connectivity edges: {edge_index.shape[1]}")
    
    # Test splitter
    splitter = dataset.get_splitter()
    print(f"\n=== Splitter ===")
    print(f"Splitter created: {type(splitter)}")
    
    print(f"\n✅ Clean SWaT Dataset test passed!")
