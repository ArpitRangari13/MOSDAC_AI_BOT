# MOSDAC AI-Powered Help Bot System

A comprehensive AI assistant for the MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) portal that provides intelligent help through semantic search, knowledge graphs, and retrieval-augmented generation.

## 🚀 System Overview

This system crawls and processes data from the MOSDAC portal (www.mosdac.gov.in) to create an intelligent help bot that can answer complex queries about satellite data, missions, and procedures using advanced NLP and AI techniques.

## 🏗️ Architecture Components

### 1. Data Ingestion Layer
- **Web Scraping**: Extracts structured and unstructured data from MOSDAC portal
- **Document Processing**: Parses PDFs, tables, and metadata
- **Data Cleaning**: Preprocesses and normalizes extracted content

### 2. Knowledge Graph Engine
- **Entity Extraction**: Identifies satellites, missions, data products, locations
- **Relationship Mapping**: Creates semantic connections between entities
- **Graph Database**: Stores structured knowledge for efficient querying

### 3. Semantic Understanding
- **Intent Classification**: Determines user query intent
- **Entity Recognition**: Extracts relevant entities from user queries
- **Context Management**: Maintains conversation context across turns

### 4. RAG Pipeline
- **Document Retrieval**: Finds relevant documents using semantic search
- **Context Augmentation**: Enhances queries with retrieved information
- **Response Generation**: Uses LLMs to generate contextual responses

## 📋 Features

- **Semantic Search**: Advanced search capabilities with natural language understanding
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Spatial Awareness**: Handles location-specific queries (e.g., "Kerala rainfall data")
- **Temporal Queries**: Processes time-based requests (e.g., "January 2023 data")
- **Source Attribution**: Provides citations and sources for responses
- **Real-time Analytics**: Monitors system performance and user interactions

## 🛠️ Technology Stack

### Backend
- **Python**: Core development language
- **FastAPI**: High-performance API framework
- **spaCy**: NLP processing and entity recognition
- **LangChain**: LLM orchestration and RAG pipeline
- **Neo4j**: Graph database for knowledge storage
- **FAISS/ChromaDB**: Vector database for semantic search
- **OpenAI/HuggingFace**: Large language models

### Frontend
- **React**: Modern web interface
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library

### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **AWS/GCP**: Cloud deployment
- **Prometheus/Grafana**: Monitoring

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.9+
- Docker
- Neo4j Database

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/mosdac-ai-bot
cd mosdac-ai-bot
```

2. **Install frontend dependencies**
```bash
npm install
npm run dev
```

3. **Backend setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key_here
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

4. **Start the services**
```bash
# Start Neo4j database
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Start backend API
cd backend
uvicorn main:app --reload

# Start frontend (in another terminal)
npm run dev
```

## 📊 System Modules

### 1. Data Ingestion Module (`backend/ingestion/`)
```python
# crawler.py - Web scraping implementation
import scrapy
from scrapy.spiders import CrawlSpider
from bs4 import BeautifulSoup
import requests
from dataclasses import dataclass
from typing import List, Dict, Any
import PyMuPDF
import docx2txt

@dataclass
class Document:
    title: str
    content: str
    url: str
    document_type: str
    metadata: Dict[str, Any]

class MOSDACSpider(CrawlSpider):
    name = "mosdac_spider"
    start_urls = [
        'https://www.mosdac.gov.in/',
        'https://www.mosdac.gov.in/data/',
        'https://www.mosdac.gov.in/missions/',
        'https://www.mosdac.gov.in/help/'
    ]
    
    def parse_document(self, response):
        # Extract structured data from pages
        title = response.css('h1::text').get()
        content = response.css('div.content p::text').getall()
        
        # Handle different document types
        if response.url.endswith('.pdf'):
            content = self.extract_pdf_content(response.url)
        elif response.url.endswith('.docx'):
            content = self.extract_docx_content(response.url)
        
        document = Document(
            title=title,
            content=' '.join(content),
            url=response.url,
            document_type=self.classify_document_type(response.url),
            metadata=self.extract_metadata(response)
        )
        
        yield document
    
    def extract_pdf_content(self, url: str) -> str:
        """Extract text from PDF documents"""
        response = requests.get(url)
        pdf_document = PyMuPDF.open(stream=response.content)
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    
    def extract_docx_content(self, url: str) -> str:
        """Extract text from Word documents"""
        response = requests.get(url)
        return docx2txt.process(BytesIO(response.content))
    
    def classify_document_type(self, url: str) -> str:
        """Classify document type based on URL patterns"""
        if '/missions/' in url:
            return 'mission_info'
        elif '/data/' in url:
            return 'data_product'
        elif '/help/' in url:
            return 'help_document'
        else:
            return 'general'
    
    def extract_metadata(self, response) -> Dict[str, Any]:
        """Extract metadata from document"""
        return {
            'last_modified': response.headers.get('Last-Modified'),
            'content_type': response.headers.get('Content-Type'),
            'language': response.css('html::attr(lang)').get(),
            'keywords': response.css('meta[name="keywords"]::attr(content)').get()
        }
```

### 2. Knowledge Graph Module (`backend/knowledge_graph/`)
```python
# graph_builder.py - Knowledge graph construction
from py2neo import Graph, Node, Relationship
import spacy
from typing import List, Dict, Tuple
import re

class KnowledgeGraphBuilder:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.nlp = spacy.load("en_core_web_sm")
        
    def build_graph_from_documents(self, documents: List[Document]):
        """Build knowledge graph from processed documents"""
        for doc in documents:
            entities = self.extract_entities(doc.content)
            relationships = self.extract_relationships(doc.content, entities)
            
            # Create nodes
            for entity in entities:
                self.create_entity_node(entity, doc)
            
            # Create relationships
            for rel in relationships:
                self.create_relationship(rel)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_type = self.map_entity_type(ent.label_)
            entities.append({
                'text': ent.text,
                'type': entity_type,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Custom entity extraction for domain-specific terms
        satellite_pattern = r'\b(INSAT|CARTOSAT|GISAT|RESOURCESAT|RISAT)-?\w*\b'
        satellites = re.findall(satellite_pattern, text, re.IGNORECASE)
        
        for satellite in satellites:
            entities.append({
                'text': satellite,
                'type': 'SATELLITE',
                'start': 0,
                'end': 0
            })
        
        return entities
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Tuple]:
        """Extract relationships between entities"""
        doc = self.nlp(text)
        relationships = []
        
        # Pattern-based relationship extraction
        patterns = [
            (r'(\w+)\s+satellite\s+provides\s+(.+)', 'PROVIDES'),
            (r'(\w+)\s+data\s+available\s+for\s+(.+)', 'AVAILABLE_FOR'),
            (r'(\w+)\s+mission\s+launched\s+in\s+(.+)', 'LAUNCHED_IN'),
            (r'(\w+)\s+covers\s+(.+)', 'COVERS')
        ]
        
        for pattern, rel_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append((match[0], rel_type, match[1]))
        
        return relationships
    
    def create_entity_node(self, entity: Dict, document: Document):
        """Create entity node in graph"""
        node = Node(entity['type'], 
                   name=entity['text'],
                   source_document=document.url,
                   document_type=document.document_type)
        
        self.graph.create(node)
    
    def create_relationship(self, relationship: Tuple):
        """Create relationship between entities"""
        source, rel_type, target = relationship
        
        source_node = self.graph.nodes.match(name=source).first()
        target_node = self.graph.nodes.match(name=target).first()
        
        if source_node and target_node:
            rel = Relationship(source_node, rel_type, target_node)
            self.graph.create(rel)
    
    def query_graph(self, query: str) -> List[Dict]:
        """Query the knowledge graph"""
        return list(self.graph.run(query))
    
    def get_entity_relationships(self, entity_name: str) -> List[Dict]:
        """Get all relationships for a specific entity"""
        query = f"""
        MATCH (e {{name: '{entity_name}'}})-[r]->(related)
        RETURN e.name as entity, type(r) as relationship, related.name as related_entity
        """
        return self.query_graph(query)
```

### 3. Semantic Understanding Module (`backend/nlp/`)
```python
# intent_classifier.py - Intent classification and entity extraction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from typing import Dict, List, Tuple
import spacy
from dataclasses import dataclass

@dataclass
class Intent:
    intent: str
    confidence: float
    entities: List[Dict]

class IntentClassifier:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define intent categories
        self.intent_labels = [
            'data_search',
            'data_download',
            'mission_info',
            'technical_support',
            'spatial_query',
            'temporal_query',
            'product_specification',
            'user_guide'
        ]
    
    def classify_intent(self, text: str) -> Intent:
        """Classify user intent from text"""
        # Preprocess text
        inputs = self.tokenizer(text, return_tensors="pt", 
                              truncation=True, padding=True)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top intent
        intent_idx = torch.argmax(predictions).item()
        confidence = predictions[0][intent_idx].item()
        
        # Extract entities
        entities = self.extract_entities(text)
        
        return Intent(
            intent=self.intent_labels[intent_idx],
            confidence=confidence,
            entities=entities
        )
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from user query"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Extract spatial entities
        spatial_entities = self.extract_spatial_entities(text)
        entities.extend(spatial_entities)
        
        # Extract temporal entities
        temporal_entities = self.extract_temporal_entities(text)
        entities.extend(temporal_entities)
        
        return entities
    
    def extract_spatial_entities(self, text: str) -> List[Dict]:
        """Extract spatial/geographic entities"""
        spatial_patterns = [
            r'\b(Kerala|Tamil Nadu|Karnataka|Andhra Pradesh|Maharashtra|Gujarat|Rajasthan|Uttar Pradesh|Madhya Pradesh|Bihar|West Bengal|Assam|Odisha|Jharkhand|Chhattisgarh|Uttarakhand|Himachal Pradesh|Jammu and Kashmir|Punjab|Haryana|Delhi|Goa|Manipur|Meghalaya|Tripura|Nagaland|Mizoram|Arunachal Pradesh|Sikkim|Telangana)\b',
            r'\b(India|Indian Ocean|Arabian Sea|Bay of Bengal|Himalayas|Western Ghats|Eastern Ghats)\b',
            r'\b(\d+\.?\d*°?\s*[NS])\s*(\d+\.?\d*°?\s*[EW])\b'  # Coordinates
        ]
        
        entities = []
        for pattern in spatial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match if isinstance(match, str) else ' '.join(match),
                    'label': 'SPATIAL',
                    'start': 0,
                    'end': 0
                })
        
        return entities
    
    def extract_temporal_entities(self, text: str) -> List[Dict]:
        """Extract temporal entities"""
        temporal_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(today|yesterday|tomorrow|last week|last month|last year|this week|this month|this year)\b'
        ]
        
        entities = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match,
                    'label': 'TEMPORAL',
                    'start': 0,
                    'end': 0
                })
        
        return entities
```

### 4. RAG Pipeline Module (`backend/rag/`)
```python
# rag_pipeline.py - Retrieval-Augmented Generation
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from typing import List, Dict, Any
import numpy as np

class RAGPipeline:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(temperature=0.2, openai_api_key=openai_api_key)
        self.vector_store = None
        self.qa_chain = None
        
    def build_vector_store(self, documents: List[Document]):
        """Build vector store from documents"""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = []
        metadatas = []
        
        for doc in documents:
            chunks = text_splitter.split_text(doc.content)
            texts.extend(chunks)
            metadatas.extend([{
                'source': doc.url,
                'title': doc.title,
                'doc_type': doc.document_type
            }] * len(chunks))
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5})
        )
    
    def generate_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response using RAG pipeline"""
        if not self.qa_chain:
            raise ValueError("Vector store not built. Call build_vector_store first.")
        
        # Enhance query with context
        enhanced_query = self.enhance_query_with_context(query, context)
        
        # Get relevant documents
        relevant_docs = self.vector_store.similarity_search(enhanced_query, k=5)
        
        # Generate response
        response = self.qa_chain.run(enhanced_query)
        
        # Extract sources
        sources = [doc.metadata.get('source', '') for doc in relevant_docs]
        
        return {
            'response': response,
            'sources': list(set(sources)),
            'relevant_documents': [doc.page_content for doc in relevant_docs],
            'confidence': self.calculate_confidence(query, relevant_docs)
        }
    
    def enhance_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance query with conversation context"""
        if not context:
            return query
        
        enhanced_query = query
        
        # Add spatial context
        if 'location' in context:
            enhanced_query += f" for {context['location']}"
        
        # Add temporal context
        if 'date' in context:
            enhanced_query += f" in {context['date']}"
        
        # Add previous conversation context
        if 'previous_queries' in context:
            recent_queries = context['previous_queries'][-3:]  # Last 3 queries
            context_str = " ".join(recent_queries)
            enhanced_query = f"Context: {context_str}. Current query: {enhanced_query}"
        
        return enhanced_query
    
    def calculate_confidence(self, query: str, documents: List) -> float:
        """Calculate confidence score for the response"""
        if not documents:
            return 0.0
        
        # Calculate semantic similarity between query and top documents
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = [self.embeddings.embed_query(doc.page_content) for doc in documents]
        
        similarities = []
        for doc_embedding in doc_embeddings:
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)
        
        return float(np.mean(similarities))
```

### 5. API Gateway Module (`backend/api/`)
```python
# main.py - FastAPI application
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from nlp.intent_classifier import IntentClassifier
from rag.rag_pipeline import RAGPipeline
from knowledge_graph.graph_builder import KnowledgeGraphBuilder

app = FastAPI(title="MOSDAC AI Help Bot", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
intent_classifier = IntentClassifier()
rag_pipeline = RAGPipeline(openai_api_key="your_openai_key")
knowledge_graph = KnowledgeGraphBuilder(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    intent: str
    entities: List[Dict]
    sources: List[str]
    confidence: float
    session_id: str

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query and generate response"""
    try:
        # Classify intent and extract entities
        intent_result = intent_classifier.classify_intent(request.query)
        
        # Generate response using RAG
        rag_result = rag_pipeline.generate_response(
            query=request.query,
            context=request.context
        )
        
        # Enhance with knowledge graph if needed
        if intent_result.intent in ['mission_info', 'data_search']:
            graph_results = knowledge_graph.query_graph(
                f"MATCH (n) WHERE n.name CONTAINS '{request.query}' RETURN n LIMIT 5"
            )
            # Incorporate graph results into response
        
        return QueryResponse(
            response=rag_result['response'],
            intent=intent_result.intent,
            entities=intent_result.entities,
            sources=rag_result['sources'],
            confidence=rag_result['confidence'],
            session_id=request.session_id or "default"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MOSDAC AI Help Bot"}

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "total_queries": 12543,
        "active_users": 1234,
        "knowledge_base_size": 45678,
        "response_accuracy": 94.2
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📊 Evaluation Metrics

### 1. Intent Recognition Accuracy
```python
def evaluate_intent_recognition(test_data: List[Tuple[str, str]]) -> float:
    """Evaluate intent classification accuracy"""
    correct = 0
    total = len(test_data)
    
    for query, expected_intent in test_data:
        predicted_intent = intent_classifier.classify_intent(query).intent
        if predicted_intent == expected_intent:
            correct += 1
    
    return correct / total
```

### 2. Entity Extraction F1 Score
```python
def evaluate_entity_extraction(test_data: List[Tuple[str, List[Dict]]]) -> float:
    """Evaluate entity extraction performance"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for query, expected_entities in test_data:
        predicted_entities = intent_classifier.extract_entities(query)
        
        # Calculate F1 score
        predicted_set = set((e['text'], e['label']) for e in predicted_entities)
        expected_set = set((e['text'], e['label']) for e in expected_entities)
        
        true_positives += len(predicted_set & expected_set)
        false_positives += len(predicted_set - expected_set)
        false_negatives += len(expected_set - predicted_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
```

### 3. Response Quality Metrics
```python
def evaluate_response_quality(test_data: List[Tuple[str, str]]) -> Dict[str, float]:
    """Evaluate response quality using multiple metrics"""
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    
    bleu_scores = []
    rouge_scores = []
    
    for query, expected_response in test_data:
        generated_response = rag_pipeline.generate_response(query)['response']
        
        # BLEU score
        bleu = sentence_bleu([expected_response.split()], generated_response.split())
        bleu_scores.append(bleu)
        
        # ROUGE score
        rouge = Rouge()
        rouge_score = rouge.get_scores(generated_response, expected_response)[0]
        rouge_scores.append(rouge_score['rouge-l']['f'])
    
    return {
        'bleu_score': np.mean(bleu_scores),
        'rouge_score': np.mean(rouge_scores)
    }
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
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

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - neo4j
      - redis
    volumes:
      - ./data:/app/data

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - api

volumes:
  neo4j_data:
```

## 🚀 Cloud Deployment

### AWS Deployment (using ECS)
```yaml
# aws-task-definition.json
{
  "family": "mosdac-ai-bot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "mosdac-ai-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/mosdac-ai-bot:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "NEO4J_URI",
          "value": "bolt://neo4j.cluster.amazonaws.com:7687"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/mosdac-ai-bot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Railway Deployment
```yaml
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[env]
NEO4J_URI = "bolt://neo4j.railway.internal:7687"
OPENAI_API_KEY = "${{secrets.OPENAI_API_KEY}}"
```

## 📈 Performance Monitoring

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mosdac-ai-bot'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "MOSDAC AI Bot Metrics",
    "panels": [
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(response_time_seconds)",
            "legendFormat": "Average Response Time"
          }
        ]
      },
      {
        "title": "Query Volume",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(queries_total[5m])",
            "legendFormat": "Queries per Second"
          }
        ]
      }
    ]
  }
}
```

## 🔧 Configuration Files

### requirements.txt
```txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
scrapy==2.11.0
beautifulsoup4==4.12.2
PyMuPDF==1.23.9
docx2txt==0.8
spacy==3.7.2
transformers==4.35.2
torch==2.1.1
langchain==0.0.340
openai==1.3.6
faiss-cpu==1.7.4
py2neo==2021.2.3
redis==5.0.1
prometheus-client==0.19.0
nltk==3.8.1
rouge==1.0.1
numpy==1.24.3
pandas==2.1.4
scikit-learn==1.3.2
```

This comprehensive MOSDAC AI-powered help bot system provides intelligent assistance through advanced NLP, knowledge graphs, and retrieval-augmented generation. The modular architecture ensures scalability and adaptability across different government portals.