#!/bin/bash

# MOSDAC AI Training Setup Script
# This script sets up the training environment and downloads required models

set -e

echo "🚀 Setting up MOSDAC AI Training Environment..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/external
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p evaluation

# Download pre-trained models for fine-tuning
echo "🤖 Downloading pre-trained models..."
python -c "
from transformers import AutoTokenizer, AutoModel
import os

models = [
    'microsoft/DialoGPT-medium',
    'bert-base-uncased',
    'distilbert-base-uncased-distilled-squad'
]

for model_name in models:
    print(f'Downloading {model_name}...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f'✅ {model_name} downloaded successfully')
    except Exception as e:
        print(f'❌ Failed to download {model_name}: {e}')
"

# Set up environment variables
echo "🔧 Setting up environment variables..."
cat > .env << EOF
# MOSDAC AI Training Configuration
PYTHONPATH=\${PYTHONPATH}:$(pwd)
TRANSFORMERS_CACHE=./models/cache
HF_HOME=./models/cache
WANDB_PROJECT=mosdac-ai-training
CUDA_VISIBLE_DEVICES=0

# API Keys (add your keys here)
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
WANDB_API_KEY=your_wandb_key_here

# Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379

# Training Configuration
BATCH_SIZE=8
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MAX_LENGTH=512
EOF

# Create training configuration file
echo "⚙️ Creating training configuration..."
cat > config/training_config.yaml << EOF
# MOSDAC AI Training Configuration

model:
  base_model: "microsoft/DialoGPT-medium"
  max_length: 512
  
training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 500
  weight_decay: 0.01
  
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  
datasets:
  external:
    - name: "squad_v2"
      weight: 1.0
    - name: "natural_questions"
      weight: 0.8
    - name: "arxiv_metadata"
      weight: 0.6
    - name: "nasa_cmr_metadata"
      weight: 1.2
  
  mosdac_specific:
    weight: 2.0
    
evaluation:
  metrics:
    - "accuracy"
    - "f1_score"
    - "bleu_score"
    - "rouge_score"
  
logging:
  level: "INFO"
  wandb_enabled: true
  tensorboard_enabled: true
EOF

# Create Docker configuration
echo "🐳 Creating Docker configuration..."
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/models/cache

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "training/train_pipeline.py"]
EOF

# Create docker-compose for development
cat > docker-compose.yml << EOF
version: '3.8'

services:
  mosdac-training:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
      - TRANSFORMERS_CACHE=/app/models/cache
    depends_on:
      - neo4j
      - redis
    
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_dbms_memory_heap_initial__size: 512m
      NEO4J_dbms_memory_heap_max__size: 2G
    volumes:
      - neo4j_data:/data
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  neo4j_data:
  redis_data:
EOF

# Create evaluation script
echo "📊 Creating evaluation script..."
cat > scripts/evaluate_models.py << 'EOF'
#!/usr/bin/env python3
"""
Model Evaluation Script for MOSDAC AI
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from training.model_trainer import MOSDACModelTrainer, TrainingConfig
from training.data_preprocessor import DataPreprocessor

def main():
    print("🔍 Evaluating MOSDAC AI Models...")
    
    # Load test data
    test_data_path = Path("data/test_data.json")
    if not test_data_path.exists():
        print("❌ Test data not found. Please run training pipeline first.")
        return
    
    # Initialize trainer
    config = TrainingConfig(output_dir="models")
    trainer = MOSDACModelTrainer(config)
    
    # Model paths
    model_paths = {
        'intent_classifier': 'models/intent_classifier',
        'qa_model': 'models/qa_model'
    }
    
    # Evaluate models
    results = trainer.evaluate_models(test_data_path, model_paths)
    
    # Print results
    print("\n📈 Evaluation Results:")
    print("=" * 50)
    
    for model_type, metrics in results.items():
        print(f"\n{model_type.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    # Save results
    results_path = Path("evaluation/evaluation_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/evaluate_models.py

# Create quick start script
echo "🏃 Creating quick start script..."
cat > scripts/quick_start.py << 'EOF'
#!/usr/bin/env python3
"""
Quick Start Script for MOSDAC AI Training
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_pipeline import MOSDACTrainingPipeline

async def main():
    print("🚀 Starting MOSDAC AI Quick Training...")
    
    # Initialize pipeline
    pipeline = MOSDACTrainingPipeline("quick_training")
    
    # Run pipeline with limited datasets for quick testing
    pipeline.dataset_manager.datasets = pipeline.dataset_manager.datasets[:2]  # Limit to 2 datasets
    
    # Reduce training epochs for quick testing
    pipeline.training_config.num_epochs = 1
    pipeline.training_config.batch_size = 4
    
    # Run pipeline
    results = await pipeline.run_complete_pipeline()
    
    print(f"\n✅ Quick training completed with status: {results['status']}")
    
    if results['status'] == 'completed':
        print("\n🎉 Models are ready for testing!")
        print("Run 'python scripts/evaluate_models.py' to evaluate performance.")
    else:
        print(f"\n❌ Training failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x scripts/quick_start.py

echo "✅ MOSDAC AI Training Environment Setup Complete!"
echo ""
echo "🎯 Next Steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Update API keys in .env file"
echo "3. Run quick training: python scripts/quick_start.py"
echo "4. Or run full pipeline: python training/train_pipeline.py"
echo ""
echo "📚 Documentation:"
echo "- Training configuration: config/training_config.yaml"
echo "- Environment variables: .env"
echo "- Docker deployment: docker-compose up"
echo ""
echo "🔧 Troubleshooting:"
echo "- Check logs in: logs/"
echo "- Monitor training: tensorboard --logdir models/*/logs"
echo "- Evaluate models: python scripts/evaluate_models.py"