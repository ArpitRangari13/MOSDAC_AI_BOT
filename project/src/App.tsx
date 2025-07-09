import React, { useState } from 'react';
import { Search, User, GraduationCap, Atom, Users, Satellite, Sun, Eye, Bot, MessageSquare, Menu, X, Download, Calendar, MapPin, FileText, BarChart3, Globe, Cloud, Zap, Database, TrendingUp, AlertTriangle, Bell, Settings, Home, Info, Phone, Mail } from 'lucide-react';
import ChatInterface from './components/ChatInterface';

function App() {
  const [showChat, setShowChat] = useState(false);
  const [activeSection, setActiveSection] = useState('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Mock function to simulate AI response
  const handleSendMessage = async (message: string): Promise<string> => {
    // This function is now handled by the enhanced ChatInterface component
    // The actual AI logic is implemented in ChatInterface.tsx
    return "Response handled by Arpit AI";
  };

  const navigationItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'data', label: 'Data Products', icon: Database },
    { id: 'missions', label: 'Missions', icon: Satellite },
    { id: 'tools', label: 'Tools', icon: Settings },
    { id: 'about', label: 'About', icon: Info },
    { id: 'contact', label: 'Contact', icon: Phone },
  ];

  const renderContent = () => {
    switch (activeSection) {
      case 'home':
        return <HomeContent />;
      case 'data':
        return <DataProductsContent />;
      case 'missions':
        return <MissionsContent />;
      case 'tools':
        return <ToolsContent />;
      case 'about':
        return <AboutContent />;
      case 'contact':
        return <ContactContent />;
      default:
        return <HomeContent />;
    }
  };

  const HomeContent = () => (
    <>
      {/* Hero Section */}
      <section className="relative bg-gradient-to-r from-blue-900 via-blue-800 to-blue-900 text-white overflow-hidden">
        <div className="absolute inset-0 bg-black opacity-20"></div>
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{
            backgroundImage: `url('https://images.pexels.com/photos/87651/earth-blue-planet-globe-planet-87651.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080&fit=crop')`
          }}
        ></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="max-w-3xl">
            <h1 className="text-5xl font-bold mb-6 leading-tight">
              Meteorological &<br />
              Oceanographic Satellite Data
            </h1>
            <p className="text-xl mb-8 text-blue-100">
              Access real-time satellite data, weather forecasts, and scientific resources powered by ISRO.
            </p>
            
            {/* Search Bar */}
            <div className="relative max-w-lg">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search datasets, FAQs..."
                className="w-full pl-12 pr-4 py-4 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-300 focus:outline-none"
              />
            </div>
          </div>
        </div>
      </section>

      {/* User Categories */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-white rounded-xl shadow-lg p-8 text-center hover:shadow-xl transition-shadow cursor-pointer">
            <GraduationCap className="w-16 h-16 text-blue-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Students</h3>
            <p className="text-gray-600">
              Educational resources & tutorials for students learning Earth sciences
            </p>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-8 text-center hover:shadow-xl transition-shadow cursor-pointer">
            <Atom className="w-16 h-16 text-blue-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Researchers</h3>
            <p className="text-gray-600">
              Advanced tools and datasets for scientific research and analysis
            </p>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-8 text-center hover:shadow-xl transition-shadow cursor-pointer">
            <Users className="w-16 h-16 text-blue-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-4">General Public</h3>
            <p className="text-gray-600">
              Weather information, alerts, and visualization tools for public users
            </p>
          </div>
        </div>
      </section>

      {/* Content Sections */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Data */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Recent Data</h2>
            <div className="space-y-4">
              <div className="bg-gradient-to-br from-blue-500 to-green-400 rounded-lg p-4 text-white">
                <div 
                  className="w-full h-32 bg-cover bg-center rounded mb-4"
                  style={{
                    backgroundImage: `url('https://images.pexels.com/photos/355935/pexels-photo-355935.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&fit=crop')`
                  }}
                ></div>
                <h3 className="font-bold">INSAT-3D | Apr 24 2024</h3>
                <p className="text-sm opacity-90">Brightness Temperature</p>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-700">Ocean Color Data</span>
                  <Download className="w-4 h-4 text-blue-600" />
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-700">Cyclone Tracking</span>
                  <Download className="w-4 h-4 text-blue-600" />
                </div>
              </div>
            </div>
          </div>

          {/* Weather & Alerts */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Weather & Alerts</h2>
            <div className="text-center mb-6">
              <Sun className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
              <div className="text-4xl font-bold text-gray-900 mb-2">32°C</div>
              <div className="text-lg text-gray-600 mb-1">New Delhi</div>
              <div className="text-sm text-gray-500 mb-4">Clear sky</div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center p-3 bg-yellow-50 rounded-lg">
                <AlertTriangle className="w-5 h-5 text-yellow-600 mr-3" />
                <span className="text-sm text-yellow-800">Heat Wave Warning - Rajasthan</span>
              </div>
              <div className="flex items-center p-3 bg-blue-50 rounded-lg">
                <Cloud className="w-5 h-5 text-blue-600 mr-3" />
                <span className="text-sm text-blue-800">Monsoon Update - Kerala</span>
              </div>
            </div>
          </div>

          {/* Quick Access Tools */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Quick Access Tools</h2>
            <div className="space-y-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <Eye className="w-12 h-12 text-blue-600 mx-auto mb-3" />
                <h3 className="text-lg font-bold text-gray-900 mb-2">Satellite View</h3>
                <p className="text-sm text-gray-600 mb-3">
                  Explore live and archived satellite imagery
                </p>
                <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm">
                  Launch Tool
                </button>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <button className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                  <BarChart3 className="w-6 h-6 text-gray-600 mx-auto mb-1" />
                  <span className="text-xs text-gray-700">Analytics</span>
                </button>
                <button className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                  <Globe className="w-6 h-6 text-gray-600 mx-auto mb-1" />
                  <span className="text-xs text-gray-700">Map View</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Statistics Section */}
      <section className="bg-blue-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">MOSDAC by Numbers</h2>
            <p className="text-blue-200">Serving the scientific community with reliable satellite data</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">15+</div>
              <div className="text-blue-200">Active Satellites</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">50TB</div>
              <div className="text-blue-200">Data Archive</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">10K+</div>
              <div className="text-blue-200">Registered Users</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold mb-2">24/7</div>
              <div className="text-blue-200">Data Availability</div>
            </div>
          </div>
        </div>
      </section>
    </>
  );

  const DataProductsContent = () => (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Data Products</h1>
        <p className="text-xl text-gray-600">Comprehensive satellite data products for various applications</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Data Categories */}
        <div className="lg:col-span-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              { title: "Ocean Color", icon: Globe, description: "Chlorophyll, SST, and ocean productivity data", count: "1,245 datasets" },
              { title: "Atmospheric", icon: Cloud, description: "Temperature, humidity, and atmospheric profiles", count: "2,156 datasets" },
              { title: "Land Surface", icon: MapPin, description: "Vegetation indices, land cover, and surface temperature", count: "3,421 datasets" },
              { title: "Weather", icon: Sun, description: "Precipitation, wind patterns, and weather forecasts", count: "1,876 datasets" },
              { title: "Cyclone Tracking", icon: Zap, description: "Storm monitoring and cyclone path prediction", count: "567 datasets" },
              { title: "Climate Data", icon: TrendingUp, description: "Long-term climate records and trends", count: "892 datasets" }
            ].map((category, index) => (
              <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow cursor-pointer">
                <category.icon className="w-12 h-12 text-blue-600 mb-4" />
                <h3 className="text-xl font-bold text-gray-900 mb-2">{category.title}</h3>
                <p className="text-gray-600 mb-3">{category.description}</p>
                <div className="text-sm text-blue-600 font-medium">{category.count}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Search and Filters */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Search Data</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
                <div className="flex space-x-2">
                  <input type="date" className="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500" />
                  <input type="date" className="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500" />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Region</label>
                <select className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500">
                  <option>Select Region</option>
                  <option>India</option>
                  <option>Indian Ocean</option>
                  <option>Arabian Sea</option>
                  <option>Bay of Bengal</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Satellite</label>
                <select className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500">
                  <option>All Satellites</option>
                  <option>INSAT-3D</option>
                  <option>INSAT-3DR</option>
                  <option>CARTOSAT-3</option>
                  <option>OCEANSAT-3</option>
                </select>
              </div>
              <button className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors">
                Search Data
              </button>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Popular Downloads</h3>
            <div className="space-y-3">
              {[
                "INSAT-3D Temperature Data",
                "Ocean Color Chlorophyll",
                "Monsoon Rainfall Analysis",
                "Cyclone Track Data"
              ].map((item, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-700">{item}</span>
                  <Download className="w-4 h-4 text-blue-600" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const MissionsContent = () => (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Satellite Missions</h1>
        <p className="text-xl text-gray-600">Explore ISRO's satellite missions providing meteorological and oceanographic data</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {[
          {
            name: "INSAT-3DS",
            status: "Active",
            launched: "February 2024",
            purpose: "Advanced meteorological observations",
            image: "https://images.pexels.com/photos/586063/pexels-photo-586063.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&fit=crop",
            features: ["Enhanced imaging", "Cyclone tracking", "Weather forecasting", "Climate monitoring"]
          },
          {
            name: "CARTOSAT-3",
            status: "Active",
            launched: "November 2019",
            purpose: "High-resolution Earth observation",
            image: "https://images.pexels.com/photos/586063/pexels-photo-586063.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&fit=crop",
            features: ["0.25m resolution", "Urban planning", "Disaster management", "Mapping applications"]
          },
          {
            name: "OCEANSAT-3",
            status: "Active",
            launched: "November 2021",
            purpose: "Ocean color and coastal studies",
            image: "https://images.pexels.com/photos/586063/pexels-photo-586063.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&fit=crop",
            features: ["Ocean color monitoring", "Coastal zone studies", "Fisheries support", "Water quality assessment"]
          },
          {
            name: "GISAT-1",
            status: "Planned",
            launched: "2024",
            purpose: "Geostationary Earth observation",
            image: "https://images.pexels.com/photos/586063/pexels-photo-586063.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&fit=crop",
            features: ["Real-time monitoring", "Disaster management", "Agricultural monitoring", "Environmental studies"]
          }
        ].map((mission, index) => (
          <div key={index} className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow">
            <div 
              className="h-48 bg-cover bg-center"
              style={{ backgroundImage: `url('${mission.image}')` }}
            ></div>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-2xl font-bold text-gray-900">{mission.name}</h3>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  mission.status === 'Active' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                }`}>
                  {mission.status}
                </span>
              </div>
              <p className="text-gray-600 mb-4">{mission.purpose}</p>
              <div className="mb-4">
                <span className="text-sm text-gray-500">Launched: </span>
                <span className="text-sm font-medium text-gray-900">{mission.launched}</span>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-gray-900">Key Features:</h4>
                <div className="flex flex-wrap gap-2">
                  {mission.features.map((feature, featureIndex) => (
                    <span key={featureIndex} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                      {feature}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const ToolsContent = () => (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Analysis Tools</h1>
        <p className="text-xl text-gray-600">Powerful tools for satellite data analysis and visualization</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {[
          {
            name: "Satellite Viewer",
            icon: Eye,
            description: "Interactive satellite imagery viewer with real-time data",
            features: ["Real-time imagery", "Multi-spectral analysis", "Time series animation"]
          },
          {
            name: "Data Analytics",
            icon: BarChart3,
            description: "Advanced analytics platform for satellite data processing",
            features: ["Statistical analysis", "Trend detection", "Custom algorithms"]
          },
          {
            name: "Weather Forecasting",
            icon: Cloud,
            description: "Numerical weather prediction models and forecasting tools",
            features: ["7-day forecasts", "Severe weather alerts", "Model comparison"]
          },
          {
            name: "Ocean Monitoring",
            icon: Globe,
            description: "Comprehensive ocean color and temperature analysis",
            features: ["SST mapping", "Chlorophyll analysis", "Current patterns"]
          },
          {
            name: "Cyclone Tracker",
            icon: Zap,
            description: "Real-time cyclone tracking and intensity analysis",
            features: ["Storm tracking", "Intensity forecasts", "Path prediction"]
          },
          {
            name: "Climate Analysis",
            icon: TrendingUp,
            description: "Long-term climate data analysis and trend visualization",
            features: ["Climate indices", "Anomaly detection", "Trend analysis"]
          }
        ].map((tool, index) => (
          <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
            <tool.icon className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold text-gray-900 mb-3">{tool.name}</h3>
            <p className="text-gray-600 mb-4">{tool.description}</p>
            <div className="space-y-2 mb-6">
              {tool.features.map((feature, featureIndex) => (
                <div key={featureIndex} className="flex items-center text-sm text-gray-700">
                  <div className="w-2 h-2 bg-blue-600 rounded-full mr-2"></div>
                  {feature}
                </div>
              ))}
            </div>
            <button className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors">
              Launch Tool
            </button>
          </div>
        ))}
      </div>
    </div>
  );

  const AboutContent = () => (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">About MOSDAC</h1>
        <p className="text-xl text-gray-600">Meteorological and Oceanographic Satellite Data Archival Centre</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
        <div className="space-y-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Our Mission</h2>
            <p className="text-gray-600 leading-relaxed">
              MOSDAC serves as the primary data archival and dissemination center for meteorological and oceanographic 
              satellite data from ISRO's satellite missions. We provide comprehensive data services to support weather 
              forecasting, climate research, and oceanographic studies.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Key Objectives</h2>
            <ul className="space-y-3 text-gray-600">
              <li className="flex items-start">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                Archive and maintain satellite data from ISRO missions
              </li>
              <li className="flex items-start">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                Provide easy access to meteorological and oceanographic data
              </li>
              <li className="flex items-start">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                Support research and operational applications
              </li>
              <li className="flex items-start">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                Facilitate data discovery and visualization
              </li>
            </ul>
          </div>

          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Data Services</h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-2">Real-time Data</h3>
                <p className="text-sm text-gray-600">Live satellite feeds and near real-time processing</p>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-2">Archive Data</h3>
                <p className="text-sm text-gray-600">Historical data spanning multiple decades</p>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-2">Processed Products</h3>
                <p className="text-sm text-gray-600">Value-added products and derived parameters</p>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-900 mb-2">Custom Processing</h3>
                <p className="text-sm text-gray-600">Tailored data products for specific requirements</p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div 
            className="h-64 bg-cover bg-center rounded-xl"
            style={{
              backgroundImage: `url('https://images.pexels.com/photos/586063/pexels-photo-586063.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop')`
            }}
          ></div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Quick Facts</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Established</span>
                <span className="font-medium text-gray-900">2008</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Data Archive Size</span>
                <span className="font-medium text-gray-900">50+ TB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Active Satellites</span>
                <span className="font-medium text-gray-900">15+</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Registered Users</span>
                <span className="font-medium text-gray-900">10,000+</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Data Products</span>
                <span className="font-medium text-gray-900">100+</span>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white rounded-xl p-6">
            <h3 className="text-xl font-bold mb-3">Partnership Opportunities</h3>
            <p className="text-blue-100 mb-4">
              Collaborate with MOSDAC for research projects, data sharing agreements, and capacity building programs.
            </p>
            <button className="bg-white text-blue-600 px-4 py-2 rounded-lg hover:bg-blue-50 transition-colors">
              Learn More
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const ContactContent = () => (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Contact Us</h1>
        <p className="text-xl text-gray-600">Get in touch with our team for support and inquiries</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
        <div className="space-y-8">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Get in Touch</h2>
            <div className="space-y-4">
              <div className="flex items-start space-x-4">
                <Mail className="w-6 h-6 text-blue-600 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900">Email</h3>
                  <p className="text-gray-600">mosdac@sac.isro.gov.in</p>
                  <p className="text-gray-600">support@mosdac.gov.in</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <Phone className="w-6 h-6 text-blue-600 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900">Phone</h3>
                  <p className="text-gray-600">+91-79-2691-4000</p>
                  <p className="text-gray-600">+91-79-2691-4001 (Support)</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <MapPin className="w-6 h-6 text-blue-600 mt-1" />
                <div>
                  <h3 className="font-semibold text-gray-900">Address</h3>
                  <p className="text-gray-600">
                    Space Applications Centre (SAC)<br />
                    Indian Space Research Organisation<br />
                    Ahmedabad - 380015, Gujarat, India
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Support Hours</h2>
            <div className="bg-blue-50 rounded-lg p-6">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-700">Monday - Friday</span>
                  <span className="font-medium text-gray-900">9:00 AM - 6:00 PM IST</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-700">Saturday</span>
                  <span className="font-medium text-gray-900">9:00 AM - 1:00 PM IST</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-700">Sunday</span>
                  <span className="font-medium text-gray-900">Closed</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Send us a Message</h2>
          <form className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">First Name</label>
                <input 
                  type="text" 
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your first name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Last Name</label>
                <input 
                  type="text" 
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your last name"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
              <input 
                type="email" 
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your email address"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Subject</label>
              <select className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                <option>Select a subject</option>
                <option>Data Access Support</option>
                <option>Technical Issues</option>
                <option>Research Collaboration</option>
                <option>General Inquiry</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Message</label>
              <textarea 
                rows={5}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your message"
              ></textarea>
            </div>
            <button 
              type="submit"
              className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              Send Message
            </button>
          </form>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-orange-500 rounded flex items-center justify-center">
                <span className="text-white font-bold text-sm">ISRO</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-900">MOSDAC</h1>
            </div>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex space-x-8">
              {navigationItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActiveSection(item.id)}
                  className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-colors ${
                    activeSection === item.id
                      ? 'bg-blue-100 text-blue-600'
                      : 'text-gray-700 hover:text-blue-600 hover:bg-gray-100'
                  }`}
                >
                  <item.icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>

            {/* Search Bar */}
            <div className="hidden lg:flex flex-1 max-w-lg mx-8">
              <div className="relative w-full">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  placeholder="Search datasets, FAQs..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            {/* Right side buttons */}
            <div className="flex items-center space-x-4">
              <button className="bg-blue-900 text-white px-6 py-2 rounded-lg hover:bg-blue-800 transition-colors">
                Login
              </button>
              
              {/* Mobile menu button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 rounded-lg text-gray-600 hover:bg-gray-100"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-white border-t border-gray-200">
            <div className="px-4 py-2 space-y-1">
              {navigationItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveSection(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                    activeSection === item.id
                      ? 'bg-blue-100 text-blue-600'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <item.icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main>
        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-8 h-8 bg-orange-500 rounded flex items-center justify-center">
                  <span className="text-white font-bold text-sm">ISRO</span>
                </div>
                <h3 className="text-xl font-bold">MOSDAC</h3>
              </div>
              <p className="text-gray-400 text-sm">
                Meteorological and Oceanographic Satellite Data Archival Centre
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Quick Links</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Data Products</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Satellite Missions</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Analysis Tools</a></li>
                <li><a href="#" className="hover:text-white transition-colors">User Guide</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Help Center</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
                <li><a href="#" className="hover:text-white transition-colors">API Reference</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Contact Support</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Connect</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>mosdac@sac.isro.gov.in</li>
                <li>+91-79-2691-4000</li>
                <li>Ahmedabad, Gujarat</li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400">
            <p>&copy; 2024 MOSDAC - Space Applications Centre, ISRO. All rights reserved.</p>
          </div>
        </div>
      </footer>

      {/* AI Chat Bot Floating Button */}
      <button
        onClick={() => setShowChat(true)}
        className="fixed bottom-6 right-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110 z-50 group"
      >
        <Bot className="w-6 h-6 group-hover:animate-pulse" />
        <div className="absolute -top-2 -right-2 w-4 h-4 bg-green-500 rounded-full animate-pulse"></div>
      </button>
      
      {/* Floating tooltip */}
      {!showChat && (
        <div className="fixed bottom-20 right-6 bg-gray-900 text-white px-3 py-2 rounded-lg text-sm opacity-0 hover:opacity-100 transition-opacity z-40 pointer-events-none">
          Chat with Arpit AI
        </div>
      )}

      {/* Chat Interface Modal */}
      {showChat && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl h-[80vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center">
                  <Bot className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Arpit - MOSDAC AI Assistant</h2>
                  <p className="text-sm text-gray-600">Your intelligent companion for satellite data & weather services</p>
                </div>
              </div>
              <button
                onClick={() => setShowChat(false)}
                className="text-gray-400 hover:text-gray-600 text-2xl"
              >
                ×
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <ChatInterface onSendMessage={handleSendMessage} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;