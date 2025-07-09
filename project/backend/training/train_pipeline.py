"""
Complete Training Pipeline for MOSDAC AI System
Orchestrates data download, preprocessing, and model training
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

from dataset_manager import DatasetManager
from data_preprocessor import DataPreprocessor, ProcessedSample
from model_trainer import MOSDACModelTrainer, TrainingConfig

class MOSDACTrainingPipeline:
    def __init__(self, base_dir: str = "training_workspace"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
        
        # Initialize components
        self.dataset_manager = DatasetManager(str(self.data_dir / "external"))
        self.preprocessor = DataPreprocessor()
        
        # Training configuration
        self.training_config = TrainingConfig(
            output_dir=str(self.models_dir),
            batch_size=8,  # Reduced for better compatibility
            num_epochs=3,
            learning_rate=2e-5
        )
        
        self.trainer = MOSDACModelTrainer(self.training_config)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the training pipeline"""
        logger = logging.getLogger('mosdac_training_pipeline')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        self.logger.info("Starting MOSDAC AI training pipeline...")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'datasets_downloaded': {},
            'preprocessing_results': {},
            'training_results': {},
            'evaluation_results': {},
            'status': 'running'
        }
        
        try:
            # Step 1: Download datasets
            self.logger.info("Step 1: Downloading external datasets...")
            download_results = self.dataset_manager.download_all_datasets()
            pipeline_results['datasets_downloaded'] = {
                name: str(path) if path else None 
                for name, path in download_results.items()
            }
            
            # Step 2: Preprocess data
            self.logger.info("Step 2: Preprocessing datasets...")
            all_processed_samples = []
            
            for dataset_name, dataset_path in download_results.items():
                if dataset_path is None:
                    continue
                
                try:
                    samples = await self._process_dataset(dataset_name, dataset_path)
                    all_processed_samples.extend(samples)
                    pipeline_results['preprocessing_results'][dataset_name] = len(samples)
                except Exception as e:
                    self.logger.error(f"Error processing {dataset_name}: {str(e)}")
                    pipeline_results['preprocessing_results'][dataset_name] = f"Error: {str(e)}"
            
            # Step 3: Create training datasets
            self.logger.info("Step 3: Creating training datasets...")
            training_data_path = self.data_dir / "mosdac_training_data.json"
            self.preprocessor.create_training_dataset(all_processed_samples, training_data_path)
            
            # Create domain-specific datasets
            domain_data_dir = self.data_dir / "domain_specific"
            self.preprocessor.create_domain_specific_datasets(all_processed_samples, domain_data_dir)
            
            # Step 4: Train models
            self.logger.info("Step 4: Training models...")
            training_results = self.trainer.train_all_models(training_data_path)
            pipeline_results['training_results'] = training_results
            
            # Step 5: Evaluate models (if test data available)
            self.logger.info("Step 5: Evaluating models...")
            # For now, we'll use a subset of training data for evaluation
            test_data_path = training_data_path  # In practice, use separate test set
            evaluation_results = self.trainer.evaluate_models(test_data_path, training_results)
            pipeline_results['evaluation_results'] = evaluation_results
            
            pipeline_results['status'] = 'completed'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            self.logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
        
        # Save pipeline results
        results_file = self.logs_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        return pipeline_results
    
    async def _process_dataset(self, dataset_name: str, dataset_path: Path) -> list[ProcessedSample]:
        """Process a specific dataset"""
        samples = []
        
        try:
            if dataset_name == "squad_v2":
                # Find the JSON file in the dataset directory
                json_files = list(dataset_path.glob("*.json"))
                if json_files:
                    samples = self.preprocessor.process_squad_dataset(json_files[0])
            
            elif dataset_name == "natural_questions":
                # Process Natural Questions format
                jsonl_files = list(dataset_path.glob("*.jsonl"))
                if jsonl_files:
                    samples = self.preprocessor.process_natural_questions(jsonl_files[0])
            
            elif dataset_name == "arxiv_metadata":
                # Process ArXiv metadata
                json_files = list(dataset_path.glob("*.json"))
                if json_files:
                    samples = self.preprocessor.process_arxiv_metadata(json_files[0])
            
            elif dataset_name == "nasa_cmr_metadata":
                # Process NASA CMR data
                json_files = list(dataset_path.glob("*.json"))
                if json_files:
                    samples = self.preprocessor.process_nasa_cmr_metadata(json_files[0])
            
            elif dataset_name == "geonames":
                # Process GeoNames data
                txt_files = list(dataset_path.glob("*.txt"))
                if txt_files:
                    samples = self.preprocessor.process_geonames_data(txt_files[0])
            
            self.logger.info(f"Processed {len(samples)} samples from {dataset_name}")
            
        except Exception as e:
            self.logger.error(f"Error processing {dataset_name}: {str(e)}")
        
        return samples
    
    def create_mosdac_specific_data(self) -> list[ProcessedSample]:
        """Create MOSDAC-specific training data"""
        mosdac_samples = []
        
        # MOSDAC-specific Q&A pairs
        mosdac_qa_pairs = [
            {
                "question": "How do I download rainfall data for Kerala?",
                "answer": "To download rainfall data for Kerala: 1) Register on MOSDAC portal, 2) Navigate to Data Products > Precipitation, 3) Select Kerala region and date range, 4) Choose data format (NetCDF/HDF5), 5) Add to cart and download.",
                "intent": "data_download",
                "domain": "mosdac_specific"
            },
            {
                "question": "What satellites provide ocean color data?",
                "answer": "OCEANSAT-3 and INSAT-3D provide ocean color data including chlorophyll concentration, sea surface temperature, and water quality parameters for Indian coastal waters.",
                "intent": "mission_info",
                "domain": "mosdac_specific"
            },
            {
                "question": "What is the spatial resolution of CARTOSAT-3 data?",
                "answer": "CARTOSAT-3 provides high-resolution Earth observation data with 0.25m panchromatic resolution and 1m multispectral resolution, suitable for urban planning and mapping applications.",
                "intent": "product_specification",
                "domain": "mosdac_specific"
            },
            {
                "question": "How can I access cyclone tracking data?",
                "answer": "Cyclone tracking data is available through MOSDAC's Weather & Alerts section. You can access real-time cyclone positions, intensity forecasts, and historical track data for the Indian Ocean region.",
                "intent": "data_search",
                "domain": "mosdac_specific"
            },
            {
                "question": "What data formats are supported for download?",
                "answer": "MOSDAC supports multiple data formats including NetCDF, HDF5, GeoTIFF, CSV, and KML. Choose the format based on your analysis software and requirements.",
                "intent": "technical_support",
                "domain": "mosdac_specific"
            }
        ]
        
        for qa_pair in mosdac_qa_pairs:
            entities = self.preprocessor._extract_entities(qa_pair["question"])
            
            sample = ProcessedSample(
                question=qa_pair["question"],
                answer=qa_pair["answer"],
                context=f"MOSDAC Portal Information",
                entities=entities,
                intent=qa_pair["intent"],
                domain=qa_pair["domain"],
                source="mosdac_manual"
            )
            mosdac_samples.append(sample)
        
        return mosdac_samples
    
    def generate_training_report(self, pipeline_results: Dict[str, Any]) -> str:
        """Generate a comprehensive training report"""
        report = f"""
# MOSDAC AI Training Pipeline Report

## Pipeline Execution Summary
- **Start Time**: {pipeline_results.get('start_time', 'N/A')}
- **End Time**: {pipeline_results.get('end_time', 'N/A')}
- **Status**: {pipeline_results.get('status', 'Unknown')}

## Dataset Download Results
"""
        
        for dataset, result in pipeline_results.get('datasets_downloaded', {}).items():
            status = "✅ Success" if result else "❌ Failed"
            report += f"- **{dataset}**: {status}\n"
        
        report += "\n## Preprocessing Results\n"
        
        total_samples = 0
        for dataset, count in pipeline_results.get('preprocessing_results', {}).items():
            if isinstance(count, int):
                total_samples += count
                report += f"- **{dataset}**: {count} samples\n"
            else:
                report += f"- **{dataset}**: {count}\n"
        
        report += f"\n**Total Training Samples**: {total_samples}\n"
        
        report += "\n## Model Training Results\n"
        
        for model_type, path in pipeline_results.get('training_results', {}).items():
            report += f"- **{model_type}**: Saved to {path}\n"
        
        report += "\n## Model Evaluation Results\n"
        
        for model_type, metrics in pipeline_results.get('evaluation_results', {}).items():
            report += f"- **{model_type}**:\n"
            for metric, value in metrics.items():
                report += f"  - {metric}: {value:.3f}\n"
        
        report += """
## Next Steps

1. **Deploy Models**: Deploy trained models to production environment
2. **Integration Testing**: Test models with MOSDAC portal integration
3. **Performance Monitoring**: Set up monitoring for model performance
4. **Continuous Learning**: Implement feedback loop for model improvement
5. **User Acceptance Testing**: Conduct UAT with domain experts

## Model Usage

The trained models can be loaded and used as follows:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load intent classifier
intent_tokenizer = AutoTokenizer.from_pretrained('models/intent_classifier')
intent_model = AutoModelForSequenceClassification.from_pretrained('models/intent_classifier')

# Load QA model
qa_tokenizer = AutoTokenizer.from_pretrained('models/qa_model')
qa_model = AutoModelForQuestionAnswering.from_pretrained('models/qa_model')
```
"""
        
        return report

# Main execution function
async def main():
    """Main function to run the training pipeline"""
    pipeline = MOSDACTrainingPipeline()
    
    # Add MOSDAC-specific data
    mosdac_samples = pipeline.create_mosdac_specific_data()
    
    # Save MOSDAC-specific data
    mosdac_data_path = pipeline.data_dir / "mosdac_specific_data.json"
    pipeline.preprocessor.create_training_dataset(mosdac_samples, mosdac_data_path)
    
    # Run complete pipeline
    results = await pipeline.run_complete_pipeline()
    
    # Generate and save report
    report = pipeline.generate_training_report(results)
    report_path = pipeline.logs_dir / "training_report.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Training pipeline completed. Report saved to: {report_path}")
    print(f"Results: {results['status']}")

if __name__ == "__main__":
    asyncio.run(main())