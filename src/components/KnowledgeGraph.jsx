import React, { useState, useEffect, useCallback } from 'react';
import { ForceGraph2D } from 'react-force-graph-2d';
import { Search, Filter, Plus, Download, Settings, Shield, AlertTriangle } from 'lucide-react';

// API service for NeuralBlitz
const NeuralBlitzAPI = {
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
  
  async searchConcepts(query, limit = 10) {
    const response = await fetch(`${this.baseURL}/concepts/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit })
    });
    return response.json();
  },
  
  async getConcept(id) {
    const response = await fetch(`${this.baseURL}/concepts/${id}`);
    return response.json();
  },
  
  async getRelatedConcepts(id, depth = 1) {
    const response = await fetch(`${this.baseURL}/concepts/${id}/related?depth=${depth}`);
    return response.json();
  },
  
  async findConnections(startId, endId) {
    const response = await fetch(`${this.baseURL}/concepts/connections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start_concept: startId, end_concept: endId })
    });
    return response.json();
  }
};

// Main Knowledge Graph Component
const KnowledgeGraph = () => {
  const [concepts, setConcepts] = useState([]);
  const [selectedConcept, setSelectedConcept] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    relationshipType: 'all',
    riskLevel: 'all',
    dateRange: 'all'
  });
  const [auditInfo, setAuditInfo] = useState(null);

  // Transform concept data for graph visualization
  const transformToGraphData = useCallback((conceptList) => {
    const nodes = conceptList.map(concept => ({
      id: concept.id || concept.concept_id,
      name: concept.data?.title || concept.id,
      data: concept,
      group: concept.data?.category || 'default',
      color: getNodeColor(concept),
      size: getNodeSize(concept)
    }));

    const links = [];
    conceptList.forEach(concept => {
      if (concept.connections && Array.isArray(concept.connections)) {
        concept.connections.forEach(connection => {
          links.push({
            source: concept.id || concept.concept_id,
            target: connection.target_concept,
            type: connection.relation || 'related',
            strength: connection.weight || 1,
            color: getLinkColor(connection)
          });
        });
      }
    });

    return { nodes, links };
  }, []);

  // Get node color based on concept properties
  const getNodeColor = (concept) => {
    const riskLevel = concept.metadata?.risk_level;
    switch (riskLevel) {
      case 'HIGH': return '#ef4444';
      case 'MEDIUM': return '#f59e0b';
      case 'LOW': return '#10b981';
      default: return '#3b82f6';
    }
  };

  // Get node size based on connections or importance
  const getNodeSize = (concept) => {
    const connectionCount = concept.connections?.length || 0;
    return Math.max(5, Math.min(20, 5 + connectionCount));
  };

  // Get link color based on relationship type
  const getLinkColor = (connection) => {
    switch (connection.relation) {
      case 'depends_on': return '#8b5cf6';
      case 'relates_to': return '#06b6d4';
      case 'conflicts_with': return '#ef4444';
      default: return '#94a3b8';
    }
  };

  // Search concepts
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    try {
      const results = await NeuralBlitzAPI.searchConcepts(searchQuery, 50);
      if (results.results) {
        setConcepts(results.results);
        const graphData = transformToGraphData(results.results);
        setGraphData(graphData);
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  }, [searchQuery, transformToGraphData]);

  // Load concept details
  const loadConceptDetails = useCallback(async (conceptId) => {
    try {
      const concept = await NeuralBlitzAPI.getConcept(conceptId);
      setSelectedConcept(concept);
      
      // Load related concepts
      const related = await NeuralBlitzAPI.getRelatedConcepts(conceptId, 2);
      if (related.related_concepts && related.related_concepts.length > 0) {
        // Load full details for related concepts
        const relatedConcepts = await Promise.all(
          related.related_concepts.map(id => NeuralBlitzAPI.getConcept(id))
        );
        
        const allConcepts = [concept, ...relatedConcepts];
        setConcepts(allConcepts);
        const graphData = transformToGraphData(allConcepts);
        setGraphData(graphData);
      }
    } catch (error) {
      console.error('Error loading concept details:', error);
    }
  }, [transformToGraphData]);

  // Handle node click in graph
  const handleNodeClick = useCallback((node) => {
    loadConceptDetails(node.id);
  }, [loadConceptDetails]);

  // Handle node hover for tooltips
  const handleNodeHover = useCallback((node) => {
    if (node) {
      // Show tooltip with concept preview
      const tooltip = document.getElementById('tooltip');
      if (tooltip) {
        tooltip.style.display = 'block';
        tooltip.style.left = `${event.clientX + 10}px`;
        tooltip.style.top = `${event.clientY - 30}px`;
        tooltip.innerHTML = `
          <div class="font-semibold">${node.name}</div>
          <div class="text-sm">${node.data?.data?.description || 'No description'}</div>
          <div class="text-xs">Connections: ${node.data?.connections?.length || 0}</div>
        `;
      }
    } else {
      const tooltip = document.getElementById('tooltip');
      if (tooltip) {
        tooltip.style.display = 'none';
      }
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-xl font-semibold text-gray-900">NeuralBlitz Knowledge Graph</h1>
            </div>
            
            {/* Search Bar */}
            <div className="flex items-center space-x-4">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search concepts..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  className="w-64 px-4 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="absolute right-2 top-2.5 text-gray-400 hover:text-gray-600 disabled:opacity-50"
                >
                  <Search className="h-5 w-5" />
                </button>
              </div>
              
              <button className="p-2 text-gray-400 hover:text-gray-600">
                <Filter className="h-5 w-5" />
              </button>
              <button className="p-2 text-gray-400 hover:text-gray-600">
                <Plus className="h-5 w-5" />
              </button>
              <button className="p-2 text-gray-400 hover:text-gray-600">
                <Download className="h-5 w-5" />
              </button>
              <button className="p-2 text-gray-400 hover:text-gray-600">
                <Settings className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-screen pt-16">
        {/* Sidebar - Concept Details */}
        <div className="w-96 bg-white border-r border-gray-200 overflow-y-auto">
          {selectedConcept ? (
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">{selectedConcept.data?.title || selectedConcept.id}</h2>
                <span className={`px-2 py-1 text-xs rounded-full ${getRiskLevelClass(selectedConcept.metadata?.risk_level)}`}>
                  {selectedConcept.metadata?.risk_level || 'LOW'}
                </span>
              </div>
              
              {selectedConcept.data?.description && (
                <p className="text-gray-600 mb-4">{selectedConcept.data.description}</p>
              )}
              
              {/* Metadata */}
              <div className="space-y-3 mb-6">
                <div className="text-sm">
                  <span className="font-medium">ID:</span>
                  <span className="ml-2 text-gray-600">{selectedConcept.id}</span>
                </div>
                
                <div className="text-sm">
                  <span className="font-medium">Created:</span>
                  <span className="ml-2 text-gray-600">
                    {new Date(selectedConcept.created_at).toLocaleDateString()}
                  </span>
                </div>
                
                <div className="text-sm">
                  <span className="font-medium">Connections:</span>
                  <span className="ml-2 text-gray-600">
                    {selectedConcept.connections?.length || 0} relationships
                  </span>
                </div>
              </div>

              {/* Relationships */}
              {selectedConcept.connections && selectedConcept.connections.length > 0 && (
                <div className="mb-6">
                  <h3 className="font-medium text-gray-900 mb-3">Relationships</h3>
                  <div className="space-y-2">
                    {selectedConcept.connections.map((connection, index) => (
                      <div key={index} className="p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-sm">{connection.relation}</span>
                          <span className="text-xs text-gray-500">Weight: {connection.weight || 1}</span>
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          {connection.target_concept}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Audit Trail Info */}
              {auditInfo && (
                <div className="border-t border-gray-200 pt-4">
                  <h3 className="font-medium text-gray-900 mb-3">Audit Information</h3>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="font-medium">Trace ID:</span>
                      <span className="ml-2 text-gray-600">{auditInfo.trace_id}</span>
                    </div>
                    <div>
                      <span className="font-medium">GoldenDAG:</span>
                      <span className="ml-2 text-gray-600 text-xs">{auditInfo.golden_dag}</span>
                    </div>
                    <div className="flex items-center">
                      <span className="font-medium">Verified:</span>
                      <span className={`ml-2 ${auditInfo.verified ? 'text-green-600' : 'text-red-600'}`}>
                        {auditInfo.verified ? '✓ Verified' : '⚠ Not Verified'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-6 text-center text-gray-500">
              <Shield className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p>Select a concept to view details</p>
            </div>
          )}
        </div>

        {/* Main Graph View */}
        <div className="flex-1 relative">
          {graphData.nodes.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Search className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Start Exploring</h3>
                <p className="text-gray-600 mb-4">Search for concepts to visualize their relationships</p>
                <div className="text-sm text-gray-500">
                  <p>Try searching for:</p>
                  <ul className="mt-2 space-y-1">
                    <li>• "AI safety"</li>
                    <li>• "Machine learning"</li>
                    <li>• "Ethical governance"</li>
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <ForceGraph2D
              graphData={graphData}
              nodeLabel="name"
              nodeColor="color"
              nodeVal="size"
              linkColor="color"
              linkWidth="strength"
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
              enableNodeDrag={true}
              enableZoomPanInteraction={true}
              cooldownTicks={100}
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.3}
            />
          )}
          
          {/* Tooltip */}
          <div id="tooltip" className="absolute hidden bg-gray-900 text-white p-2 rounded-lg text-sm pointer-events-none z-50" />
        </div>

        {/* Stats Sidebar */}
        <div className="w-64 bg-white border-l border-gray-200 overflow-y-auto">
          <div className="p-4">
            <h3 className="font-medium text-gray-900 mb-4">Knowledge Graph Stats</h3>
            
            <div className="space-y-4">
              <div className="bg-blue-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{graphData.nodes.length}</div>
                <div className="text-sm text-blue-800">Total Concepts</div>
              </div>
              
              <div className="bg-green-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{graphData.links.length}</div>
                <div className="text-sm text-green-800">Relationships</div>
              </div>
              
              <div className="bg-purple-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {graphData.links.length > 0 ? (graphData.links.length / graphData.nodes.length).toFixed(1) : '0'}
                </div>
                <div className="text-sm text-purple-800">Avg. Connections</div>
              </div>
            </div>

            {/* Filters */}
            <div className="mt-6">
              <h3 className="font-medium text-gray-900 mb-3">Filters</h3>
              
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-gray-600">Relationship Type</label>
                  <select 
                    value={filters.relationshipType}
                    onChange={(e) => setFilters({...filters, relationshipType: e.target.value})}
                    className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
                  >
                    <option value="all">All Types</option>
                    <option value="depends_on">Depends On</option>
                    <option value="relates_to">Relates To</option>
                    <option value="conflicts_with">Conflicts With</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-sm text-gray-600">Risk Level</label>
                  <select 
                    value={filters.riskLevel}
                    onChange={(e) => setFilters({...filters, riskLevel: e.target.value})}
                    className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
                  >
                    <option value="all">All Levels</option>
                    <option value="low">Low Risk</option>
                    <option value="medium">Medium Risk</option>
                    <option value="high">High Risk</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="mt-6">
              <h3 className="font-medium text-gray-900 mb-3">Quick Actions</h3>
              <div className="space-y-2">
                <button className="w-full px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700">
                  Add New Concept
                </button>
                <button className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50">
                  Export Graph
                </button>
                <button className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50">
                  Generate Report
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Helper function for risk level styling
  function getRiskLevelClass(level) {
    switch (level) {
      case 'HIGH': return 'bg-red-100 text-red-800';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800';
      case 'LOW': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }
};

export default KnowledgeGraph;