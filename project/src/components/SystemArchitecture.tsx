import React from 'react';
import { Database, Brain, MessageSquare, Cloud, Search, FileText, Network, Zap } from 'lucide-react';

const SystemArchitecture: React.FC = () => {
  const components = [
    {
      title: "Data Ingestion Layer",
      icon: Database,
      description: "Crawls and extracts structured/unstructured data from MOSDAC portal",
      technologies: ["Python", "Scrapy", "BeautifulSoup", "PyMuPDF", "Textract"],
      color: "from-blue-500 to-blue-600"
    },
    {
      title: "Knowledge Graph Engine",
      icon: Network,
      description: "Builds semantic relationships between entities and concepts",
      technologies: ["Neo4j", "spaCy", "NetworkX", "RDF/OWL", "SPARQL"],
      color: "from-green-500 to-green-600"
    },
    {
      title: "Semantic Understanding",
      icon: Brain,
      description: "NLP models for intent classification and entity extraction",
      technologies: ["spaCy", "Transformers", "BERT", "Rasa NLU", "scikit-learn"],
      color: "from-purple-500 to-purple-600"
    },
    {
      title: "Vector Store & Search",
      icon: Search,
      description: "Efficient semantic search and document retrieval",
      technologies: ["FAISS", "ChromaDB", "Elasticsearch", "Sentence-BERT", "Pinecone"],
      color: "from-orange-500 to-orange-600"
    },
    {
      title: "RAG Pipeline",
      icon: Zap,
      description: "Retrieval-Augmented Generation with LLMs",
      technologies: ["LangChain", "OpenAI GPT", "Mistral", "HuggingFace", "NVIDIA NeMo"],
      color: "from-red-500 to-red-600"
    },
    {
      title: "Chat Interface",
      icon: MessageSquare,
      description: "User-friendly conversational interface",
      technologies: ["React", "TypeScript", "Tailwind CSS", "WebSocket", "Socket.io"],
      color: "from-teal-500 to-teal-600"
    },
    {
      title: "API Gateway",
      icon: Cloud,
      description: "Scalable backend services and deployment",
      technologies: ["FastAPI", "Docker", "Kubernetes", "AWS/GCP", "Railway"],
      color: "from-indigo-500 to-indigo-600"
    },
    {
      title: "Analytics & Monitoring",
      icon: FileText,
      description: "Performance metrics and system evaluation",
      technologies: ["Prometheus", "Grafana", "MLflow", "Weights & Biases", "ELK Stack"],
      color: "from-yellow-500 to-yellow-600"
    }
  ];

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            MOSDAC AI System Architecture
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Comprehensive AI-powered help bot system with knowledge graph integration,
            semantic search, and retrieval-augmented generation capabilities
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
          {components.map((component, index) => (
            <div
              key={index}
              className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300 overflow-hidden group"
            >
              <div className={`bg-gradient-to-r ${component.color} p-6 text-white`}>
                <component.icon className="w-8 h-8 mb-3" />
                <h3 className="text-xl font-semibold mb-2">{component.title}</h3>
              </div>
              
              <div className="p-6">
                <p className="text-gray-600 mb-4 text-sm leading-relaxed">
                  {component.description}
                </p>
                
                <div className="space-y-2">
                  <h4 className="font-medium text-gray-900 text-sm">Technologies:</h4>
                  <div className="flex flex-wrap gap-1">
                    {component.technologies.map((tech, techIndex) => (
                      <span
                        key={techIndex}
                        className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-12 bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Data Flow Architecture</h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-blue-600">1. Data Ingestion</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• Web scraping of MOSDAC portal</p>
                <p>• PDF and document parsing</p>
                <p>• Metadata extraction</p>
                <p>• Data cleaning and preprocessing</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-green-600">2. Knowledge Processing</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• Entity recognition and linking</p>
                <p>• Relationship extraction</p>
                <p>• Knowledge graph construction</p>
                <p>• Vector embeddings generation</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-purple-600">3. Query Processing</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• Intent classification</p>
                <p>• Entity extraction</p>
                <p>• Semantic search</p>
                <p>• Response generation</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemArchitecture;