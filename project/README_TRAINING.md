# MOSDAC AI Training Pipeline

This comprehensive training pipeline integrates multiple external datasets to train a sophisticated AI assistant for the MOSDAC portal. The system combines general QA datasets, scientific literature, satellite metadata, and geospatial data to create a domain-aware conversational AI.

## 🎯 Training Objectives

1. **Multi-Domain Understanding**: Train on diverse datasets for robust performance
2. **Domain Adaptation**: Specialize for meteorological and oceanographic queries
3. **Spatial Awareness**: Handle location-specific queries effectively
4. **Scientific Accuracy**: Provide accurate information about satellite missions and data

## 📊 Integrated Datasets

### General QA Datasets
- **SQuAD v2.0**: Stanford Question Answering Dataset
- **Natural Questions**: Google's real-world QA dataset
- **HotpotQA**: Multi-hop reasoning questions

### Scientific Datasets
- **ArXiv Metadata**: Research papers in Earth sciences
- **NASA CMR**: Satellite mission metadata
- **PubMed Central**: Biomedical and Earth science articles

### Conversational Datasets
- **MultiWOZ**: Multi-domain dialogue dataset
- **DSTC**: Dialog system challenges

### Geospatial Datasets
- **GeoNames**: Global geographical database
- **OpenStreetMap**: Spatial data for location awareness

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   Preprocessing  │───▶│  Model Training │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ External APIs   │    │ Entity Extraction│    │ Intent Classifier│
│ Web Scraping    │    │ Intent Mapping   │    │ QA Model        │
│ File Processing │    │ Data Augmentation│    │ Entity Recognizer│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository>
cd mosdac-ai-training

# Run setup script
chmod +x backend/scripts/setup_training.sh
./backend/scripts/setup_training.sh

# Activate environment
source venv/bin/activate
```

### 2. Configuration
Update the configuration files:

**`.env`** - Add your API keys:
```env
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_KEY=your_hf_key
WANDB_API_KEY=your_wandb_key
```

**`config/training_config.yaml`** - Adjust training parameters:
```yaml
training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 3
```

### 3. Quick Training (Recommended for Testing)
```bash
python scripts/quick_start.py
```

### 4. Full Training Pipeline
```bash
python training/train_pipeline.py
```

## 📋 Training Pipeline Steps

### Step 1: Dataset Download
- Downloads external datasets from configured sources
- Handles various formats (JSON, CSV, XML, TXT)
- Implements retry logic and error handling

### Step 2: Data Preprocessing
- Standardizes data formats across sources
- Extracts entities and classifies intents
- Creates domain-specific training splits

### Step 3: Model Training
- **Intent Classifier**: Categorizes user queries
- **Question Answering**: Generates contextual responses
- **Entity Recognizer**: Identifies domain-specific entities

### Step 4: Model Evaluation
- Calculates accuracy, F1-score, BLEU, and ROUGE metrics
- Generates comprehensive evaluation reports
- Validates performance across domains

## 🎛️ Configuration Options

### Training Configuration
```yaml
model:
  base_model: "microsoft/DialoGPT-medium"
  max_length: 512

training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 500

datasets:
  external:
    - name: "squad_v2"
      weight: 1.0
    - name: "natural_questions"
      weight: 0.8
  
  mosdac_specific:
    weight: 2.0
```

### Dataset Weights
- **Higher weights**: More influence during training
- **MOSDAC-specific data**: Weighted 2.0 for domain specialization
- **Scientific datasets**: Weighted 1.2 for technical accuracy

## 📊 Model Performance

### Expected Metrics
- **Intent Classification Accuracy**: >90%
- **Entity Extraction F1-Score**: >85%
- **QA BLEU Score**: >75%
- **Response Relevance**: >80%

### Evaluation Commands
```bash
# Evaluate all models
python scripts/evaluate_models.py

# Generate detailed report
python training/generate_report.py

# Monitor training progress
tensorboard --logdir models/*/logs
```

## 🐳 Docker Deployment

### Development Environment
```bash
docker-compose up -d
```

### Production Deployment
```bash
# Build image
docker build -t mosdac-ai-training .

# Run training
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           mosdac-ai-training
```

## 📈 Monitoring and Logging

### Weights & Biases Integration
```python
import wandb

wandb.init(project="mosdac-ai-training")
wandb.config.update({
    "learning_rate": 2e-5,
    "batch_size": 8,
    "epochs": 3
})
```

### TensorBoard Monitoring
```bash
tensorboard --logdir models/*/logs --port 6006
```

### Log Files
- **Training logs**: `logs/training_*.log`
- **Pipeline results**: `logs/pipeline_results.json`
- **Evaluation reports**: `evaluation/evaluation_results.json`

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```yaml
# Reduce batch size in config
training:
  batch_size: 4  # Reduce from 8
```

**2. Dataset Download Failures**
```bash
# Check network connectivity
curl -I https://rajpurkar.github.io/SQuAD-explorer/

# Retry with force download
python -c "
from training.dataset_manager import DatasetManager
dm = DatasetManager()
dm.download_dataset(config, force_download=True)
"
```

**3. Model Loading Errors**
```bash
# Clear model cache
rm -rf models/cache/*

# Reinstall transformers
pip install --upgrade transformers
```

### Performance Optimization

**1. Multi-GPU Training**
```python
# Enable DataParallel
model = torch.nn.DataParallel(model)
```

**2. Mixed Precision Training**
```python
# Add to training arguments
training_args.fp16 = True
```

**3. Gradient Accumulation**
```python
# Increase effective batch size
training_args.gradient_accumulation_steps = 4
```

## 📚 Advanced Usage

### Custom Dataset Integration
```python
from training.data_preprocessor import DataPreprocessor, ProcessedSample

# Create custom samples
custom_samples = [
    ProcessedSample(
        question="Your question",
        answer="Your answer",
        intent="custom_intent",
        domain="custom_domain"
    )
]

# Add to training pipeline
preprocessor = DataPreprocessor()
preprocessor.create_training_dataset(custom_samples, "custom_data.json")
```

### Model Fine-tuning
```python
from training.model_trainer import MOSDACModelTrainer, TrainingConfig

# Custom training configuration
config = TrainingConfig(
    model_name="your-custom-model",
    batch_size=16,
    learning_rate=1e-5,
    num_epochs=5
)

trainer = MOSDACModelTrainer(config)
```

### Evaluation Metrics
```python
# Custom evaluation
from training.evaluation import evaluate_model

results = evaluate_model(
    model_path="models/qa_model",
    test_data="data/test_data.json",
    metrics=["bleu", "rouge", "meteor"]
)
```

## 🤝 Contributing

### Adding New Datasets
1. Update `dataset_manager.py` with new dataset configuration
2. Implement preprocessing in `data_preprocessor.py`
3. Add tests for new functionality
4. Update documentation

### Model Improvements
1. Experiment with different base models
2. Implement new evaluation metrics
3. Add domain-specific fine-tuning
4. Optimize hyperparameters

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google**: Natural Questions dataset
- **Stanford**: SQuAD dataset
- **NASA**: CMR metadata API
- **GeoNames**: Geographical database
- **Hugging Face**: Transformers library
- **ISRO**: MOSDAC portal and satellite data

## 📞 Support

For questions and support:
- **Email**: mosdac-ai-support@example.com
- **Issues**: GitHub Issues
- **Documentation**: [Training Wiki](wiki/training)
- **Community**: [Discord Channel](discord-link)

---

**Happy Training! 🚀**