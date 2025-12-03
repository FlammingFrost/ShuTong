import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes timeout for long-running operations
});

// API service functions
export const api = {
  // Get analysis data for overview charts
  async getAnalysisData() {
    try {
      const response = await apiClient.get('/api/analysis');
      return response.data;
    } catch (error) {
      console.error('Error fetching analysis data:', error);
      throw error;
    }
  },

  // Generate a math problem
  async generateProblem(topic, difficulty, problemType) {
    try {
      const response = await apiClient.post('/api/generate-problem', {
        topic,
        difficulty,
        problemType,
      });
      return response.data;
    } catch (error) {
      console.error('Error generating problem:', error);
      throw error;
    }
  },

  // Run the agent pipeline
  async runAgent(mathProblem, maxIterations = 3) {
    try {
      const response = await apiClient.post('/api/run-agent', {
        math_problem: mathProblem,
        max_iterations: maxIterations,
      });
      return response.data;
    } catch (error) {
      console.error('Error running agent:', error);
      throw error;
    }
  },

  // Stream agent progress (for real-time updates)
  async streamAgentProgress(mathProblem, maxIterations, onProgress) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/run-agent-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          math_problem: mathProblem,
          max_iterations: maxIterations,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start agent stream');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              onProgress(data);
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error streaming agent progress:', error);
      throw error;
    }
  },

  // Load demo result (pre-loaded for quick demonstration)
  async loadDemoResult() {
    try {
      const response = await apiClient.get('/api/demo-result');
      return response.data;
    } catch (error) {
      console.error('Error loading demo result:', error);
      throw error;
    }
  },

  // Get run history
  async getRunHistory() {
    try {
      const response = await apiClient.get('/api/runs');
      return response.data;
    } catch (error) {
      console.error('Error fetching run history:', error);
      throw error;
    }
  },

  // Get specific run details
  async getRunDetails(runId) {
    try {
      const response = await apiClient.get(`/api/runs/${runId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching run details:', error);
      throw error;
    }
  },
};

export default apiClient;
