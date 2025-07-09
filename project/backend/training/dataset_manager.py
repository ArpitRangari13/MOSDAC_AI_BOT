"""
Dataset Manager for MOSDAC AI Training Pipeline
Handles downloading, processing, and integration of external datasets
"""

import os
import requests
import pandas as pd
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import zipfile
import tarfile
from urllib.parse import urljoin
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class DatasetConfig:
    name: str
    url: str
    dataset_type: str  # 'qa', 'scientific', 'conversational', 'geospatial', 'satellite'
    format: str  # 'json', 'csv', 'xml', 'pdf', 'txt'
    description: str
    preprocessing_required: bool = True

class DatasetManager:
    def __init__(self, data_dir: str = "data/external"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
        # Define external datasets for training
        self.datasets = [
            # General QA Datasets
            DatasetConfig(
                name="natural_questions",
                url="https://ai.google.com/research/NaturalQuestions/dataset",
                dataset_type="qa",
                format="json",
                description="Google Natural Questions dataset for QA training"
            ),
            DatasetConfig(
                name="squad_v2",
                url="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
                dataset_type="qa",
                format="json",
                description="Stanford QA Dataset v2.0"
            ),
            DatasetConfig(
                name="hotpot_qa",
                url="https://hotpotqa.github.io/data/hotpot_train_v1.1.json",
                dataset_type="qa",
                format="json",
                description="Multi-hop QA dataset"
            ),
            
            # Scientific Datasets
            DatasetConfig(
                name="arxiv_metadata",
                url="https://www.kaggle.com/datasets/Cornell-University/arxiv",
                dataset_type="scientific",
                format="json",
                description="ArXiv research papers metadata"
            ),
            
            # Conversational Datasets
            DatasetConfig(
                name="multiwoz",
                url="https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2.zip",
                dataset_type="conversational",
                format="json",
                description="Multi-domain dialogue dataset"
            ),
            
            # Geospatial Datasets
            DatasetConfig(
                name="geonames",
                url="http://download.geonames.org/export/dump/allCountries.zip",
                dataset_type="geospatial",
                format="txt",
                description="GeoNames geographical database"
            ),
            
            # Satellite/Earth Observation Datasets
            DatasetConfig(
                name="nasa_cmr_metadata",
                url="https://cmr.earthdata.nasa.gov/search/collections.json?page_size=2000",
                dataset_type="satellite",
                format="json",
                description="NASA CMR satellite mission metadata"
            )
        ]
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for dataset operations"""
        logger = logging.getLogger('dataset_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def download_dataset(self, config: DatasetConfig, force_download: bool = False) -> Optional[Path]:
        """Download a dataset if not already present"""
        dataset_path = self.data_dir / config.name
        
        if dataset_path.exists() and not force_download:
            self.logger.info(f"Dataset {config.name} already exists, skipping download")
            return dataset_path
        
        try:
            self.logger.info(f"Downloading {config.name} from {config.url}")
            
            # Create dataset directory
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Download file
            response = requests.get(config.url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Determine filename
            filename = config.url.split('/')[-1]
            if not filename or '.' not in filename:
                filename = f"{config.name}.{config.format}"
            
            file_path = dataset_path / filename
            
            # Save file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract if compressed
            if filename.endswith('.zip'):
                self._extract_zip(file_path, dataset_path)
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                self._extract_tar(file_path, dataset_path)
            
            self.logger.info(f"Successfully downloaded {config.name}")
            return dataset_path
            
        except Exception as e:
            self.logger.error(f"Failed to download {config.name}: {str(e)}")
            return None
    
    def _extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        zip_path.unlink()  # Remove zip file after extraction
    
    def _extract_tar(self, tar_path: Path, extract_to: Path):
        """Extract TAR file"""
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
        tar_path.unlink()  # Remove tar file after extraction
    
    def download_all_datasets(self, max_workers: int = 3) -> Dict[str, Optional[Path]]:
        """Download all datasets concurrently"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {
                executor.submit(self.download_dataset, config): config 
                for config in self.datasets
            }
            
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results[config.name] = result
                except Exception as e:
                    self.logger.error(f"Error downloading {config.name}: {str(e)}")
                    results[config.name] = None
        
        return results
    
    def get_dataset_info(self) -> List[Dict[str, Any]]:
        """Get information about all configured datasets"""
        return [
            {
                'name': config.name,
                'type': config.dataset_type,
                'format': config.format,
                'description': config.description,
                'downloaded': (self.data_dir / config.name).exists()
            }
            for config in self.datasets
        ]