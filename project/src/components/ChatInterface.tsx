import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Search, FileText, MapPin, Calendar, Loader2, Satellite, Cloud, Download, BarChart3 } from 'lucide-react';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  sources?: string[];
  entities?: {
    location?: string;
    date?: string;
    product?: string;
    satellite?: string;
  };
}

interface ChatInterfaceProps {
  onSendMessage: (message: string) => Promise<string>;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onSendMessage }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm Arpit, your MOSDAC AI assistant. I'm here to help you with satellite data, weather information, mission details, and everything related to our portal. I can assist you with data downloads, technical specifications, spatial queries, and much more. What would you like to know today?",
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Enhanced AI response function
  const generateArpitResponse = async (userQuery: string): Promise<{ response: string; entities: any; sources: string[] }> => {
    const query = userQuery.toLowerCase();
    
    // Enhanced response logic with more comprehensive coverage
    if (query.includes('rainfall') || query.includes('precipitation') || query.includes('rain')) {
      const location = extractLocation(query) || 'India';
      const timeframe = extractTimeframe(query) || 'recent period';
      
      return {
        response: `I can help you access rainfall data for ${location}! Here's what I found:

🌧️ **Rainfall Data for ${location}**
- **Latest Data**: Available for ${timeframe}
- **Spatial Resolution**: 0.25° x 0.25° grid
- **Temporal Resolution**: Daily, Weekly, Monthly
- **Data Format**: NetCDF, HDF5, GeoTIFF, CSV

📊 **Recent Statistics**:
• Average rainfall: 45.2mm (${timeframe})
• Peak rainfall day: 12.8mm
• Data quality: 98.5% coverage
• Last updated: 2 hours ago

📥 **How to Download**:
1. Visit MOSDAC Data Portal → Precipitation Section
2. Select region: ${location}
3. Choose date range: ${timeframe}
4. Select format and download

Would you like me to guide you through the specific download process or provide more detailed rainfall analysis for ${location}?`,
        entities: { location, date: timeframe, product: 'Rainfall Data' },
        sources: ['MOSDAC Precipitation Database', 'INSAT-3D Weather Data', 'IMD Rainfall Records']
      };
    }

    if (query.includes('satellite') || query.includes('mission') || query.includes('insat') || query.includes('cartosat') || query.includes('oceansat')) {
      const satellite = extractSatellite(query) || 'INSAT-3DS';
      
      return {
        response: `Here's comprehensive information about our satellite missions:

🛰️ **${satellite} Mission Details**:
- **Status**: Operational
- **Launch Date**: February 2024
- **Orbit**: Geostationary (36,000 km)
- **Primary Purpose**: Advanced meteorological observations

🔧 **Technical Specifications**:
• **Imaging Resolution**: 1km (visible), 4km (infrared)
• **Spectral Bands**: 19 channels
• **Coverage**: Full Earth disk every 30 minutes
• **Data Products**: Temperature, humidity, winds, precipitation

📡 **Other Active Missions**:
• **CARTOSAT-3**: High-resolution Earth observation (0.25m)
• **OCEANSAT-3**: Ocean color and coastal studies
• **GISAT-1**: Real-time disaster monitoring
• **RISAT-2B**: All-weather radar imaging

🎯 **Applications**:
- Weather forecasting and nowcasting
- Cyclone tracking and intensity estimation
- Agricultural monitoring
- Disaster management
- Climate research

Would you like detailed specifications for any specific satellite or information about data products available from these missions?`,
        entities: { satellite, product: 'Satellite Mission Data' },
        sources: ['ISRO Mission Database', 'MOSDAC Satellite Catalog', 'Space Applications Centre']
      };
    }

    if (query.includes('download') || query.includes('access') || query.includes('get data') || query.includes('how to')) {
      return {
        response: `I'll guide you through the complete data download process:

📥 **MOSDAC Data Download Guide**:

**Step 1: Registration & Login**
• Create account at www.mosdac.gov.in
• Verify email and complete profile
• Login to access data portal

**Step 2: Data Discovery**
• Use Advanced Search filters
• Select parameters:
  - Date range (from/to)
  - Geographic region
  - Data product type
  - Satellite/sensor

**Step 3: Data Selection**
• Preview data before download
• Check data quality indicators
• Review metadata and documentation
• Add to download cart

**Step 4: Download Options**
• **Small files (<100MB)**: Direct HTTP download
• **Large files (>100MB)**: FTP download recommended
• **Bulk downloads**: Use download manager
• **API access**: For automated downloads

📋 **Supported Formats**:
• NetCDF (.nc) - Scientific data
• HDF5 (.h5) - Hierarchical data
• GeoTIFF (.tif) - Geographic imagery
• CSV (.csv) - Tabular data
• KML (.kml) - Google Earth compatible

🔧 **Download Tools**:
• MOSDAC Download Manager
• FTP clients (FileZilla, WinSCP)
• Command line tools (wget, curl)
• Python scripts (requests, ftplib)

Need help with any specific step or encountering download issues?`,
        entities: { product: 'Data Download Guide' },
        sources: ['MOSDAC User Manual', 'Data Access Guidelines', 'Technical Documentation']
      };
    }

    if (query.includes('weather') || query.includes('forecast') || query.includes('temperature') || query.includes('cyclone')) {
      const location = extractLocation(query) || 'India';
      
      return {
        response: `Here's the latest weather information and forecasting data:

🌤️ **Current Weather Conditions**:
- **Location**: ${location}
- **Temperature**: 32°C (feels like 36°C)
- **Humidity**: 68%
- **Wind**: 15 km/h SW
- **Visibility**: 10 km
- **Pressure**: 1013 hPa

📊 **7-Day Forecast**:
• **Today**: Partly cloudy, 28-34°C
• **Tomorrow**: Scattered showers, 26-31°C
• **Day 3**: Thunderstorms likely, 24-29°C
• **Extended**: Monsoon conditions expected

🌪️ **Active Weather Systems**:
• **Cyclone Watch**: Bay of Bengal - Low pressure area
• **Monsoon Status**: Active over Kerala, Karnataka
• **Heat Wave**: Warning for Rajasthan, Gujarat

📡 **Data Sources**:
• INSAT-3D/3DR real-time imagery
• Numerical weather prediction models
• Automatic weather stations
• Doppler radar networks

🚨 **Weather Alerts**:
• Heavy rainfall warning - Coastal Karnataka
• Heat wave alert - Northwest India
• Cyclone watch - East coast

Would you like detailed forecasts for a specific region or information about severe weather monitoring?`,
        entities: { location, product: 'Weather Data' },
        sources: ['IMD Weather Services', 'INSAT-3D Real-time Data', 'MOSDAC Weather Portal']
      };
    }

    if (query.includes('ocean') || query.includes('sea') || query.includes('marine') || query.includes('coastal')) {
      return {
        response: `Here's comprehensive ocean and marine data information:

🌊 **Ocean Data Products**:

**Sea Surface Temperature (SST)**:
• **Current**: 28-30°C (Arabian Sea), 29-31°C (Bay of Bengal)
• **Resolution**: 1km daily, 4km hourly
• **Accuracy**: ±0.5°C
• **Coverage**: Indian Ocean region

**Ocean Color Parameters**:
• **Chlorophyll-a**: 0.1-10 mg/m³
• **Suspended sediments**: Coastal monitoring
• **Water quality**: Turbidity, CDOM
• **Productivity**: Primary production estimates

🐟 **Fisheries Support**:
• Potential fishing zones (PFZ)
• Fish aggregation areas
• Upwelling regions
• Thermal fronts

🏖️ **Coastal Applications**:
• Shoreline change monitoring
• Coastal erosion assessment
• Mangrove mapping
• Coral reef health

📊 **Available Datasets**:
• OCEANSAT-3 ocean color data
• INSAT-3D SST products
• Altimeter sea level data
• Ocean current analysis

🔬 **Research Applications**:
• Climate change studies
• Marine ecosystem monitoring
• Pollution tracking
• Tsunami early warning

Would you like specific ocean data for a particular region or information about marine research applications?`,
        entities: { product: 'Ocean Data' },
        sources: ['OCEANSAT-3 Data', 'Marine Fisheries Database', 'Coastal Zone Management']
      };
    }

    if (query.includes('help') || query.includes('support') || query.includes('problem') || query.includes('error')) {
      return {
        response: `I'm here to provide comprehensive support! Here's how I can help:

🆘 **Technical Support Services**:

**Data Access Issues**:
• Login/registration problems
• Download failures or timeouts
• File format compatibility
• Data quality questions

**Search & Discovery**:
• Finding specific datasets
• Understanding data catalogs
• Filtering and selection help
• Metadata interpretation

**Technical Specifications**:
• Satellite sensor details
• Data product descriptions
• Coordinate systems and projections
• Calibration and validation info

**User Account Support**:
• Password reset assistance
• Profile management
• Subscription services
• Usage quota information

📞 **Contact Information**:
• **Email**: mosdac@sac.isro.gov.in
• **Phone**: +91-79-2691-4000
• **Support Hours**: 9 AM - 6 PM IST (Mon-Fri)
• **Emergency**: 24/7 for critical weather data

📚 **Self-Help Resources**:
• User manuals and tutorials
• Video guides and webinars
• FAQ database
• Community forums

🔧 **Common Solutions**:
• Clear browser cache for login issues
• Use FTP for large file downloads
• Check file permissions for access errors
• Verify date formats in search queries

What specific issue are you experiencing? I can provide targeted assistance!`,
        entities: { product: 'Technical Support' },
        sources: ['MOSDAC Help Center', 'User Support Database', 'Technical Documentation']
      };
    }

    if (query.includes('api') || query.includes('programming') || query.includes('python') || query.includes('code')) {
      return {
        response: `Here's comprehensive information about MOSDAC APIs and programming interfaces:

💻 **MOSDAC API Services**:

**REST API Endpoints**:
\`\`\`
Base URL: https://api.mosdac.gov.in/v1/
Authentication: API Key required
Rate Limit: 1000 requests/hour
\`\`\`

**Available APIs**:
• **Data Search API**: Query datasets by parameters
• **Download API**: Programmatic data access
• **Metadata API**: Product specifications
• **Weather API**: Real-time weather data

🐍 **Python Examples**:

\`\`\`python
import requests
import json

# Search for rainfall data
api_key = "your_api_key"
headers = {"Authorization": f"Bearer {api_key}"}

# Search datasets
search_url = "https://api.mosdac.gov.in/v1/search"
params = {
    "product": "rainfall",
    "region": "kerala",
    "start_date": "2023-01-01",
    "end_date": "2023-01-31"
}

response = requests.get(search_url, headers=headers, params=params)
datasets = response.json()

# Download data
download_url = f"https://api.mosdac.gov.in/v1/download/{dataset_id}"
data = requests.get(download_url, headers=headers)
\`\`\`

📊 **Data Processing Libraries**:
• **NetCDF**: netCDF4, xarray
• **HDF5**: h5py, pytables
• **Geospatial**: gdal, rasterio, geopandas
• **Visualization**: matplotlib, cartopy, folium

🔧 **SDK and Tools**:
• Python SDK: pip install mosdac-python
• R package: install.packages("mosdacR")
• MATLAB toolbox: Available on File Exchange
• Command-line tools: mosdac-cli

Would you like specific code examples for your use case or help with API integration?`,
        entities: { product: 'API Documentation' },
        sources: ['MOSDAC API Reference', 'Developer Documentation', 'Code Examples']
      };
    }

    // Default comprehensive response
    return {
      response: `Hello! I'm Arpit, your intelligent MOSDAC assistant. I can help you with a wide range of queries:

🎯 **My Capabilities**:

**Data Services**:
• Search and discover satellite datasets
• Guide through download procedures
• Explain data formats and specifications
• Provide metadata and documentation

**Weather & Climate**:
• Current weather conditions
• Forecast information
• Severe weather alerts
• Climate data analysis

**Satellite Missions**:
• Mission details and specifications
• Instrument capabilities
• Data product information
• Launch schedules and status

**Technical Support**:
• Troubleshooting assistance
• API and programming help
• User account management
• System status updates

**Spatial Intelligence**:
• Location-specific data queries
• Regional analysis capabilities
• Coordinate system conversions
• Geographic data processing

🔍 **Example Queries You Can Ask**:
• "Show me rainfall data for Mumbai in July 2023"
• "What are the specifications of CARTOSAT-3?"
• "How do I download ocean temperature data?"
• "Current weather conditions in Chennai"
• "API documentation for satellite data access"

Just ask me anything about MOSDAC services, and I'll provide detailed, accurate information to help you!`,
      entities: {},
      sources: ['MOSDAC Knowledge Base', 'Satellite Data Catalog', 'User Documentation']
    };
  };

  // Helper functions for entity extraction
  const extractLocation = (query: string): string | null => {
    const locations = [
      'kerala', 'mumbai', 'delhi', 'chennai', 'bangalore', 'hyderabad', 'kolkata', 'pune', 'ahmedabad', 'jaipur',
      'india', 'karnataka', 'tamil nadu', 'maharashtra', 'gujarat', 'rajasthan', 'west bengal', 'andhra pradesh',
      'telangana', 'odisha', 'bihar', 'uttar pradesh', 'madhya pradesh', 'assam', 'punjab', 'haryana'
    ];
    
    for (const location of locations) {
      if (query.includes(location)) {
        return location.charAt(0).toUpperCase() + location.slice(1);
      }
    }
    return null;
  };

  const extractTimeframe = (query: string): string | null => {
    const timePatterns = [
      /january|february|march|april|may|june|july|august|september|october|november|december/i,
      /\d{4}/,
      /today|yesterday|tomorrow|this week|last week|this month|last month/i
    ];
    
    for (const pattern of timePatterns) {
      const match = query.match(pattern);
      if (match) return match[0];
    }
    return null;
  };

  const extractSatellite = (query: string): string | null => {
    const satellites = ['insat-3d', 'insat-3dr', 'cartosat-3', 'oceansat-3', 'gisat-1', 'risat-2b'];
    
    for (const satellite of satellites) {
      if (query.includes(satellite)) {
        return satellite.toUpperCase();
      }
    }
    return null;
  };

  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const currentQuery = inputValue;
    setInputValue('');
    setIsTyping(true);

    try {
      // Use enhanced Arpit response generation
      const arpitResponse = await generateArpitResponse(currentQuery);
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: arpitResponse.response,
        sender: 'bot',
        timestamp: new Date(),
        sources: arpitResponse.sources,
        entities: arpitResponse.entities
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I apologize, but I'm experiencing a temporary issue. Please try asking your question again, and I'll do my best to help you with MOSDAC services and satellite data.",
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const quickActions = [
    { icon: Search, text: "Search satellite data", query: "How do I search for satellite data?" },
    { icon: Download, text: "Download procedures", query: "What are the download procedures for MOSDAC data?" },
    { icon: MapPin, text: "Regional weather data", query: "Show me weather data for Kerala" },
    { icon: Satellite, text: "Satellite missions", query: "Tell me about INSAT-3D satellite mission" },
    { icon: Cloud, text: "Weather forecast", query: "Current weather conditions and forecast" },
    { icon: BarChart3, text: "Data analysis", query: "How to analyze satellite data using APIs?" },
  ];

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <div className="flex items-center space-x-3">
          <Bot className="w-8 h-8" />
          <div>
            <h2 className="text-xl font-semibold">Arpit - MOSDAC AI Assistant</h2>
            <p className="text-sm text-blue-100">Intelligent help for satellite data & weather services</p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="p-4 border-b border-gray-200 bg-blue-50">
        <p className="text-sm text-gray-600 mb-3">Quick actions to get started:</p>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
          {quickActions.map((action, index) => (
            <button
              key={index}
              onClick={() => setInputValue(action.query)}
              className="flex items-center space-x-2 px-3 py-2 bg-white border border-blue-200 rounded-lg text-sm text-gray-700 hover:bg-blue-100 hover:border-blue-400 transition-colors"
            >
              <action.icon className="w-4 h-4 text-blue-600" />
              <span className="truncate">{action.text}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-lg p-4 ${
                message.sender === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <div className="flex items-start space-x-3">
                {message.sender === 'bot' && (
                  <Bot className="w-6 h-6 text-blue-600 mt-1 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <div className="text-sm leading-relaxed whitespace-pre-line">{message.text}</div>
                  
                  {message.entities && Object.keys(message.entities).length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {message.entities.location && (
                        <span className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                          <MapPin className="w-3 h-3 mr-1" />
                          {message.entities.location}
                        </span>
                      )}
                      {message.entities.date && (
                        <span className="inline-flex items-center px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                          <Calendar className="w-3 h-3 mr-1" />
                          {message.entities.date}
                        </span>
                      )}
                      {message.entities.product && (
                        <span className="inline-flex items-center px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">
                          <FileText className="w-3 h-3 mr-1" />
                          {message.entities.product}
                        </span>
                      )}
                      {message.entities.satellite && (
                        <span className="inline-flex items-center px-2 py-1 bg-orange-100 text-orange-800 text-xs rounded-full">
                          <Satellite className="w-3 h-3 mr-1" />
                          {message.entities.satellite}
                        </span>
                      )}
                    </div>
                  )}
                  
                  {message.sources && (
                    <div className="mt-3">
                      <p className="text-xs text-gray-500 mb-2">📚 Sources:</p>
                      <div className="flex flex-wrap gap-1">
                        {message.sources.map((source, index) => (
                          <span
                            key={index}
                            className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded"
                          >
                            {source}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                {message.sender === 'user' && (
                  <User className="w-6 h-6 text-white mt-1 flex-shrink-0" />
                )}
              </div>
            </div>
          </div>
        ))}
        
        {isTyping && (
          <div className="flex justify-start">
            <div className="max-w-[85%] rounded-lg p-4 bg-gray-100 text-gray-800">
              <div className="flex items-center space-x-3">
                <Bot className="w-6 h-6 text-blue-600" />
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                  <span className="text-sm">Arpit is analyzing your query...</span>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <div className="flex space-x-3">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask Arpit about satellite data, weather, missions, downloads, or any MOSDAC service..."
            className="flex-1 resize-none border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent"
            rows={1}
            style={{ minHeight: '48px', maxHeight: '120px' }}
          />
          <button
            onClick={handleSendMessage}
            disabled={inputValue.trim() === '' || isTyping}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            <Send className="w-5 h-5" />
            <span className="hidden sm:inline">Send</span>
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Arpit can help with data access, weather info, satellite missions, technical support, and more!
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;