import React, { useState } from 'react';
import { Activity, Users, MessageSquare, Database, TrendingUp, AlertTriangle, CheckCircle, Clock } from 'lucide-react';

const AdminDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const metrics = [
    {
      title: "Total Queries",
      value: "12,543",
      change: "+8.2%",
      icon: MessageSquare,
      color: "text-blue-600"
    },
    {
      title: "Active Users",
      value: "1,234",
      change: "+12.5%",
      icon: Users,
      color: "text-green-600"
    },
    {
      title: "Knowledge Base Size",
      value: "45,678",
      change: "+5.3%",
      icon: Database,
      color: "text-purple-600"
    },
    {
      title: "Response Accuracy",
      value: "94.2%",
      change: "+2.1%",
      icon: TrendingUp,
      color: "text-orange-600"
    }
  ];

  const recentQueries = [
    {
      id: 1,
      query: "How to download rainfall data for Kerala?",
      status: "resolved",
      timestamp: "2 min ago",
      accuracy: 95
    },
    {
      id: 2,
      query: "What are the latest satellite missions?",
      status: "resolved",
      timestamp: "5 min ago",
      accuracy: 88
    },
    {
      id: 3,
      query: "Technical specifications of INSAT-3D",
      status: "pending",
      timestamp: "8 min ago",
      accuracy: 92
    }
  ];

  const systemHealth = [
    { service: "API Gateway", status: "healthy", uptime: "99.9%" },
    { service: "Knowledge Graph", status: "healthy", uptime: "99.7%" },
    { service: "Vector Store", status: "warning", uptime: "98.5%" },
    { service: "LLM Service", status: "healthy", uptime: "99.8%" }
  ];

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Admin Dashboard</h1>
          <p className="text-gray-600">Monitor and manage your MOSDAC AI system</p>
        </div>

        {/* Tabs */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8">
              {['overview', 'analytics', 'system'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {metrics.map((metric, index) => (
                <div key={index} className="bg-white rounded-lg shadow-sm p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">{metric.title}</p>
                      <p className="text-2xl font-semibold text-gray-900">{metric.value}</p>
                    </div>
                    <metric.icon className={`w-8 h-8 ${metric.color}`} />
                  </div>
                  <div className="mt-4">
                    <span className="text-sm text-green-600">{metric.change} from last month</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Recent Queries */}
            <div className="bg-white rounded-lg shadow-sm">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-900">Recent Queries</h2>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  {recentQueries.map((query) => (
                    <div key={query.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex-1">
                        <p className="text-sm text-gray-900">{query.query}</p>
                        <p className="text-xs text-gray-500 mt-1">{query.timestamp}</p>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="text-sm text-gray-600">
                          Accuracy: {query.accuracy}%
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          query.status === 'resolved' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {query.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'system' && (
          <div className="space-y-8">
            {/* System Health */}
            <div className="bg-white rounded-lg shadow-sm">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-900">System Health</h2>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {systemHealth.map((service, index) => (
                    <div key={index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                      <div className="flex items-center space-x-3">
                        {service.status === 'healthy' ? (
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 text-yellow-500" />
                        )}
                        <span className="font-medium text-gray-900">{service.service}</span>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-gray-900">Uptime: {service.uptime}</div>
                        <div className={`text-xs ${
                          service.status === 'healthy' ? 'text-green-600' : 'text-yellow-600'
                        }`}>
                          {service.status}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-white rounded-lg shadow-sm">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-900">Performance Metrics</h2>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <Activity className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                    <div className="text-2xl font-semibold text-gray-900">1.2s</div>
                    <div className="text-sm text-gray-600">Avg Response Time</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <TrendingUp className="w-8 h-8 text-green-600 mx-auto mb-2" />
                    <div className="text-2xl font-semibold text-gray-900">94.2%</div>
                    <div className="text-sm text-gray-600">Intent Recognition</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <Clock className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                    <div className="text-2xl font-semibold text-gray-900">99.7%</div>
                    <div className="text-sm text-gray-600">System Uptime</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdminDashboard;