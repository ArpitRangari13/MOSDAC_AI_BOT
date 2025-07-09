"""
Data Preprocessor for MOSDAC AI Training
Processes and standardizes various dataset formats for training
"""

import json
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional
import re
from pathlib import Path
import logging
from dataclasses import dataclass
import spacy
from transformers import AutoTokenizer
import numpy as np

@dataclass
class ProcessedSample:
    question: str
    answer: str
    context: Optional[str] = None
    entities: Optional[List[Dict]] = None
    intent: Optional[str] = None
    domain: str = "general"
    source: str = "unknown"

class DataPreprocessor:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.logger = self._setup_logger()
        
        # MOSDAC-specific intent mapping
        self.intent_keywords = {
            'data_search': ['data', 'dataset', 'search', 'find', 'locate', 'access'],
            'data_download': ['download', 'get', 'retrieve', 'fetch', 'obtain'],
            'mission_info': ['mission', 'satellite', 'spacecraft', 'launch', 'orbit'],
            'technical_support': ['help', 'support', 'error', 'problem', 'issue', 'troubleshoot'],
            'spatial_query': ['location', 'region', 'area', 'coordinates', 'latitude', 'longitude'],
            'temporal_query': ['time', 'date', 'period', 'duration', 'when', 'recent'],
            'product_specification': ['specification', 'format', 'resolution', 'accuracy', 'metadata'],
            'weather_info': ['weather', 'temperature', 'rainfall', 'cyclone', 'climate']
        }
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('data_preprocessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_squad_dataset(self, file_path: Path) -> List[ProcessedSample]:
        """Process SQuAD dataset"""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for article in data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    
                    for qa in paragraph['qas']:
                        question = qa['question']
                        
                        # Handle unanswerable questions in SQuAD v2
                        if qa.get('is_impossible', False):
                            answer = "I don't have enough information to answer this question."
                        else:
                            answers = qa.get('answers', [])
                            answer = answers[0]['text'] if answers else "No answer available"
                        
                        # Extract entities and classify intent
                        entities = self._extract_entities(question)
                        intent = self._classify_intent(question)
                        
                        sample = ProcessedSample(
                            question=question,
                            answer=answer,
                            context=context,
                            entities=entities,
                            intent=intent,
                            domain="general",
                            source="squad"
                        )
                        samples.append(sample)
            
            self.logger.info(f"Processed {len(samples)} samples from SQuAD dataset")
            
        except Exception as e:
            self.logger.error(f"Error processing SQuAD dataset: {str(e)}")
        
        return samples
    
    def process_natural_questions(self, file_path: Path) -> List[ProcessedSample]:
        """Process Natural Questions dataset"""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    
                    question = data['question_text']
                    
                    # Extract answer from annotations
                    annotations = data.get('annotations', [])
                    if annotations and annotations[0].get('short_answers'):
                        # Use short answer if available
                        short_answer = annotations[0]['short_answers'][0]
                        answer_text = data['document_text'][
                            short_answer['start_token']:short_answer['end_token']
                        ]
                        answer = ' '.join(answer_text.split())
                    else:
                        answer = "No specific answer found in the document."
                    
                    # Use document text as context
                    context = data.get('document_text', '')[:2000]  # Limit context length
                    
                    entities = self._extract_entities(question)
                    intent = self._classify_intent(question)
                    
                    sample = ProcessedSample(
                        question=question,
                        answer=answer,
                        context=context,
                        entities=entities,
                        intent=intent,
                        domain="general",
                        source="natural_questions"
                    )
                    samples.append(sample)
            
            self.logger.info(f"Processed {len(samples)} samples from Natural Questions")
            
        except Exception as e:
            self.logger.error(f"Error processing Natural Questions: {str(e)}")
        
        return samples
    
    def process_arxiv_metadata(self, file_path: Path) -> List[ProcessedSample]:
        """Process ArXiv metadata for scientific QA"""
        samples = []
        
        try:
            df = pd.read_json(file_path, lines=True)
            
            for _, row in df.iterrows():
                title = row.get('title', '')
                abstract = row.get('abstract', '')
                categories = row.get('categories', '')
                
                # Generate question-answer pairs from scientific content
                if 'astro-ph' in categories or 'physics.geo-ph' in categories:
                    # Earth science related papers
                    questions = self._generate_scientific_questions(title, abstract)
                    
                    for question in questions:
                        entities = self._extract_entities(question)
                        intent = self._classify_intent(question)
                        
                        sample = ProcessedSample(
                            question=question,
                            answer=abstract[:500],  # Use abstract as answer
                            context=f"Title: {title}\nCategories: {categories}",
                            entities=entities,
                            intent=intent,
                            domain="scientific",
                            source="arxiv"
                        )
                        samples.append(sample)
            
            self.logger.info(f"Processed {len(samples)} samples from ArXiv metadata")
            
        except Exception as e:
            self.logger.error(f"Error processing ArXiv metadata: {str(e)}")
        
        return samples
    
    def process_nasa_cmr_metadata(self, file_path: Path) -> List[ProcessedSample]:
        """Process NASA CMR satellite metadata"""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry in data.get('feed', {}).get('entry', []):
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                
                # Generate satellite-specific questions
                questions = self._generate_satellite_questions(title, summary)
                
                for question in questions:
                    entities = self._extract_entities(question)
                    intent = self._classify_intent(question)
                    
                    sample = ProcessedSample(
                        question=question,
                        answer=summary,
                        context=f"Satellite Mission: {title}",
                        entities=entities,
                        intent=intent,
                        domain="satellite",
                        source="nasa_cmr"
                    )
                    samples.append(sample)
            
            self.logger.info(f"Processed {len(samples)} samples from NASA CMR")
            
        except Exception as e:
            self.logger.error(f"Error processing NASA CMR metadata: {str(e)}")
        
        return samples
    
    def process_geonames_data(self, file_path: Path) -> List[ProcessedSample]:
        """Process GeoNames geographical data"""
        samples = []
        
        try:
            # GeoNames format: geonameid, name, asciiname, alternatenames, latitude, longitude, ...
            df = pd.read_csv(file_path, sep='\t', header=None, low_memory=False)
            df.columns = ['geonameid', 'name', 'asciiname', 'alternatenames', 
                         'latitude', 'longitude', 'feature_class', 'feature_code',
                         'country_code', 'cc2', 'admin1_code', 'admin2_code',
                         'admin3_code', 'admin4_code', 'population', 'elevation',
                         'dem', 'timezone', 'modification_date']
            
            # Filter for Indian locations
            indian_locations = df[df['country_code'] == 'IN'].head(1000)  # Limit for processing
            
            for _, row in indian_locations.iterrows():
                name = row['name']
                lat = row['latitude']
                lon = row['longitude']
                
                # Generate location-based questions
                questions = [
                    f"What are the coordinates of {name}?",
                    f"Where is {name} located?",
                    f"What is the latitude and longitude of {name}?"
                ]
                
                answer = f"{name} is located at latitude {lat} and longitude {lon} in India."
                
                for question in questions:
                    entities = self._extract_entities(question)
                    intent = 'spatial_query'
                    
                    sample = ProcessedSample(
                        question=question,
                        answer=answer,
                        context=f"Location: {name}, Country: India",
                        entities=entities,
                        intent=intent,
                        domain="geospatial",
                        source="geonames"
                    )
                    samples.append(sample)
            
            self.logger.info(f"Processed {len(samples)} samples from GeoNames")
            
        except Exception as e:
            self.logger.error(f"Error processing GeoNames data: {str(e)}")
        
        return samples
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def _classify_intent(self, text: str) -> str:
        """Classify intent based on keywords"""
        text_lower = text.lower()
        
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'general_query'
    
    def _generate_scientific_questions(self, title: str, abstract: str) -> List[str]:
        """Generate questions from scientific papers"""
        questions = []
        
        # Basic question templates
        if 'satellite' in title.lower() or 'remote sensing' in title.lower():
            questions.extend([
                f"What is the research about {title.split(':')[0]}?",
                f"How does {title.split(':')[0]} work?",
                f"What are the applications of {title.split(':')[0]}?"
            ])
        
        return questions[:3]  # Limit to 3 questions per paper
    
    def _generate_satellite_questions(self, title: str, summary: str) -> List[str]:
        """Generate satellite-specific questions"""
        questions = []
        
        questions.extend([
            f"What is {title}?",
            f"What data does {title} provide?",
            f"How can I access data from {title}?",
            f"What are the specifications of {title}?"
        ])
        
        return questions
    
    def create_training_dataset(self, processed_samples: List[ProcessedSample], 
                              output_path: Path) -> None:
        """Create final training dataset"""
        training_data = []
        
        for sample in processed_samples:
            training_data.append({
                'question': sample.question,
                'answer': sample.answer,
                'context': sample.context,
                'entities': sample.entities,
                'intent': sample.intent,
                'domain': sample.domain,
                'source': sample.source
            })
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created training dataset with {len(training_data)} samples at {output_path}")
    
    def create_domain_specific_datasets(self, processed_samples: List[ProcessedSample], 
                                      output_dir: Path) -> None:
        """Create domain-specific training datasets"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by domain
        domain_samples = {}
        for sample in processed_samples:
            domain = sample.domain
            if domain not in domain_samples:
                domain_samples[domain] = []
            domain_samples[domain].append(sample)
        
        # Save each domain separately
        for domain, samples in domain_samples.items():
            domain_data = []
            for sample in samples:
                domain_data.append({
                    'question': sample.question,
                    'answer': sample.answer,
                    'context': sample.context,
                    'entities': sample.entities,
                    'intent': sample.intent,
                    'source': sample.source
                })
            
            domain_file = output_dir / f"{domain}_training_data.json"
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump(domain_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Created {domain} dataset with {len(domain_data)} samples")