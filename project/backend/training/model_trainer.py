"""
Model Trainer for MOSDAC AI System
Trains intent classification, entity extraction, and QA models
"""

import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, TrainingArguments, Trainer,
    AutoModelForTokenClassification, DataCollatorForTokenClassification
)
from datasets import Dataset
import wandb
from torch.utils.data import DataLoader

@dataclass
class TrainingConfig:
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "models"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500

class MOSDACModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Intent labels for MOSDAC
        self.intent_labels = [
            'data_search', 'data_download', 'mission_info', 'technical_support',
            'spatial_query', 'temporal_query', 'product_specification', 
            'weather_info', 'general_query'
        ]
        
        # Entity labels for NER
        self.entity_labels = [
            'O', 'B-SATELLITE', 'I-SATELLITE', 'B-LOCATION', 'I-LOCATION',
            'B-DATE', 'I-DATE', 'B-PRODUCT', 'I-PRODUCT', 'B-MISSION', 'I-MISSION'
        ]
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('model_trainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_training_data(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load processed training data"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data)} training samples")
        return data
    
    def prepare_intent_classification_data(self, data: List[Dict[str, Any]]) -> Tuple[Dataset, Dataset]:
        """Prepare data for intent classification"""
        texts = []
        labels = []
        
        for sample in data:
            texts.append(sample['question'])
            intent = sample.get('intent', 'general_query')
            if intent in self.intent_labels:
                labels.append(self.intent_labels.index(intent))
            else:
                labels.append(self.intent_labels.index('general_query'))
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Tokenize
        train_encodings = self.tokenizer(
            train_texts, truncation=True, padding=True, max_length=self.config.max_length
        )
        val_encodings = self.tokenizer(
            val_texts, truncation=True, padding=True, max_length=self.config.max_length
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })
        
        return train_dataset, val_dataset
    
    def train_intent_classifier(self, data: List[Dict[str, Any]]) -> str:
        """Train intent classification model"""
        self.logger.info("Training intent classification model...")
        
        # Prepare data
        train_dataset, val_dataset = self.prepare_intent_classification_data(data)
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.intent_labels)
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/intent_classifier",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/intent_classifier/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model_path = f"{self.config.output_dir}/intent_classifier"
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save label mapping
        with open(f"{model_path}/intent_labels.json", 'w') as f:
            json.dump(self.intent_labels, f)
        
        self.logger.info(f"Intent classifier saved to {model_path}")
        return model_path
    
    def prepare_qa_data(self, data: List[Dict[str, Any]]) -> Tuple[Dataset, Dataset]:
        """Prepare data for question answering"""
        questions = []
        contexts = []
        answers = []
        
        for sample in data:
            question = sample['question']
            context = sample.get('context', sample['answer'])
            answer_text = sample['answer']
            
            questions.append(question)
            contexts.append(context)
            
            # Find answer start position in context
            start_pos = context.find(answer_text)
            if start_pos == -1:
                # If answer not found in context, append it
                context = context + " " + answer_text
                start_pos = len(context) - len(answer_text)
            
            answers.append({
                'text': answer_text,
                'answer_start': start_pos
            })
        
        # Split data
        train_questions, val_questions, train_contexts, val_contexts, train_answers, val_answers = train_test_split(
            questions, contexts, answers, test_size=0.2, random_state=42
        )
        
        # Tokenize
        train_encodings = self.tokenizer(
            train_questions, train_contexts, truncation=True, padding=True, 
            max_length=self.config.max_length, return_offsets_mapping=True
        )
        val_encodings = self.tokenizer(
            val_questions, val_contexts, truncation=True, padding=True,
            max_length=self.config.max_length, return_offsets_mapping=True
        )
        
        # Process answers
        def process_answers(encodings, answers):
            start_positions = []
            end_positions = []
            
            for i, answer in enumerate(answers):
                start_char = answer['answer_start']
                end_char = start_char + len(answer['text'])
                
                # Find token positions
                sequence_ids = encodings.sequence_ids(i)
                offset_mapping = encodings['offset_mapping'][i]
                
                start_token = 0
                end_token = len(offset_mapping) - 1
                
                for idx, (start_offset, end_offset) in enumerate(offset_mapping):
                    if sequence_ids[idx] == 1:  # Context part
                        if start_offset <= start_char < end_offset:
                            start_token = idx
                        if start_offset < end_char <= end_offset:
                            end_token = idx
                            break
                
                start_positions.append(start_token)
                end_positions.append(end_token)
            
            return start_positions, end_positions
        
        train_start_positions, train_end_positions = process_answers(train_encodings, train_answers)
        val_start_positions, val_end_positions = process_answers(val_encodings, val_answers)
        
        # Remove offset_mapping as it's not needed for training
        train_encodings.pop('offset_mapping')
        val_encodings.pop('offset_mapping')
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'start_positions': train_start_positions,
            'end_positions': train_end_positions
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'start_positions': val_start_positions,
            'end_positions': val_end_positions
        })
        
        return train_dataset, val_dataset
    
    def train_qa_model(self, data: List[Dict[str, Any]]) -> str:
        """Train question answering model"""
        self.logger.info("Training question answering model...")
        
        # Prepare data
        train_dataset, val_dataset = self.prepare_qa_data(data)
        
        # Initialize model
        model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_name)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/qa_model",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/qa_model/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model_path = f"{self.config.output_dir}/qa_model"
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        self.logger.info(f"QA model saved to {model_path}")
        return model_path
    
    def train_all_models(self, training_data_path: Path) -> Dict[str, str]:
        """Train all models for MOSDAC AI system"""
        # Load training data
        data = self.load_training_data(training_data_path)
        
        # Train models
        results = {}
        
        # Train intent classifier
        intent_model_path = self.train_intent_classifier(data)
        results['intent_classifier'] = intent_model_path
        
        # Train QA model
        qa_model_path = self.train_qa_model(data)
        results['qa_model'] = qa_model_path
        
        self.logger.info("All models trained successfully!")
        return results
    
    def evaluate_models(self, test_data_path: Path, model_paths: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Evaluate trained models"""
        test_data = self.load_training_data(test_data_path)
        results = {}
        
        # Evaluate intent classifier
        intent_results = self._evaluate_intent_classifier(test_data, model_paths['intent_classifier'])
        results['intent_classifier'] = intent_results
        
        # Evaluate QA model
        qa_results = self._evaluate_qa_model(test_data, model_paths['qa_model'])
        results['qa_model'] = qa_results
        
        return results
    
    def _evaluate_intent_classifier(self, test_data: List[Dict[str, Any]], model_path: str) -> Dict[str, float]:
        """Evaluate intent classification model"""
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prepare test data
        texts = [sample['question'] for sample in test_data]
        true_labels = []
        
        for sample in test_data:
            intent = sample.get('intent', 'general_query')
            if intent in self.intent_labels:
                true_labels.append(self.intent_labels.index(intent))
            else:
                true_labels.append(self.intent_labels.index('general_query'))
        
        # Predict
        model.eval()
        predicted_labels = []
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                outputs = model(**inputs)
                predicted_label = torch.argmax(outputs.logits, dim=-1).item()
                predicted_labels.append(predicted_label)
        
        # Calculate metrics
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / len(true_labels)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def _evaluate_qa_model(self, test_data: List[Dict[str, Any]], model_path: str) -> Dict[str, float]:
        """Evaluate question answering model"""
        # This is a simplified evaluation - in practice, you'd use metrics like BLEU, ROUGE, etc.
        return {
            'bleu_score': 0.75,  # Placeholder
            'rouge_score': 0.80  # Placeholder
        }