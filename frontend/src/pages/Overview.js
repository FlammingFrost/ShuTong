import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import Card from '../components/Card';
import LoadingSpinner from '../components/LoadingSpinner';
import Alert from '../components/Alert';
import { getModelColor, formatPercentage, formatCurrency } from '../utils/helpers';
import { api } from '../services/api';

const Overview = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('exact'); // 'exact', 'correct', 'cost'

  useEffect(() => {
    fetchAnalysisData();
  }, []);

  const fetchAnalysisData = async () => {
    try {
      setLoading(true);
      
      // Fetch real data from API
      const response = await api.getAnalysisData();
      setData(response);
      setError(null);
    } catch (err) {
      setError('Failed to load analysis data. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getChartData = () => {
    if (!data) return [];

    return data.models.map(model => ({
      name: model.displayName,
      modelKey: model.name,
      // Exact metrics (only accuracy)
      exactAccuracy: model.exactAccuracy * 100,
      exactAccuracyPositive: model.exactAccuracyPositive * 100,
      exactAccuracyNegative: model.exactAccuracyNegative * 100,
      // Correct metrics
      correctAccuracy: model.correctAccuracy * 100,
      correctPrecision: model.correctPrecision * 100,
      correctRecall: model.correctRecall * 100,
      correctF1: model.correctF1 * 100,
      // Cost metrics
      costPerExactMatch: model.costPerExactMatch,
      costPerCorrectMatch: model.costPerCorrectMatch,
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <LoadingSpinner size="lg" text="Loading analysis data..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-8">
        <Alert type="error" title="Error" message={error} />
      </div>
    );
  }

  const chartData = getChartData();

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold gradient-text mb-4">Model Performance Overview</h1>
        <p className="text-gray-600 max-w-3xl mx-auto">
          Compare different models' performance across exact match accuracy, correct match accuracy, and cost efficiency.
          Select metrics below to explore detailed comparisons.
        </p>
      </div>

      {/* Metric Selector */}
      <Card className="bg-gradient-to-r from-primary-50 to-accent-50">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Select Metric Category
            </label>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => setSelectedMetric('exact')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  selectedMetric === 'exact'
                    ? 'bg-primary-500 text-white shadow-lg scale-105'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                📊 Exact Match Metrics
              </button>
              <button
                onClick={() => setSelectedMetric('correct')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  selectedMetric === 'correct'
                    ? 'bg-primary-500 text-white shadow-lg scale-105'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                ✅ Correct Match Metrics
              </button>
              <button
                onClick={() => setSelectedMetric('cost')}
                className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  selectedMetric === 'cost'
                    ? 'bg-primary-500 text-white shadow-lg scale-105'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                💰 Cost Efficiency
              </button>
            </div>
          </div>
        </div>
      </Card>

      {/* Chart */}
      <Card>
        <h2 className="text-2xl font-bold text-gray-800 mb-6">
          {selectedMetric === 'exact' && '📊 Exact Match Performance'}
          {selectedMetric === 'correct' && '✅ Correct Match Performance'}
          {selectedMetric === 'cost' && '💰 Cost Efficiency Comparison'}
        </h2>
        
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="name"
              angle={-45}
              textAnchor="end"
              height={80}
              tick={{ fill: '#4b5563', fontSize: 14 }}
            />
            <YAxis
              label={{ 
                value: selectedMetric === 'cost' ? 'Cost (USD)' : 'Percentage (%)', 
                angle: -90, 
                position: 'insideLeft', 
                style: { fill: '#4b5563' } 
              }}
              tick={{ fill: '#4b5563' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                padding: '12px',
              }}
              formatter={(value, name) => {
                if (selectedMetric === 'cost') {
                  return [formatCurrency(value), name];
                }
                return [`${value.toFixed(2)}%`, name];
              }}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            
            {/* Render different bars based on selected metric */}
            {selectedMetric === 'exact' && (
              <>
                <Bar 
                  dataKey="exactAccuracy" 
                  name="Overall Accuracy" 
                  fill="#0ea5e9" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `${value.toFixed(1)}%`, fontSize: 11, fill: '#374151' }}
                />
                <Bar 
                  dataKey="exactAccuracyPositive" 
                  name="Accuracy (Positive)" 
                  fill="#8b5cf6" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `${value.toFixed(1)}%`, fontSize: 11, fill: '#374151' }}
                />
                <Bar 
                  dataKey="exactAccuracyNegative" 
                  name="Accuracy (Negative)" 
                  fill="#10b981" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `${value.toFixed(1)}%`, fontSize: 11, fill: '#374151' }}
                />
              </>
            )}
            
            {selectedMetric === 'correct' && (
              <>
                <Bar 
                  dataKey="correctAccuracy" 
                  name="Accuracy" 
                  fill="#0ea5e9" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `${value.toFixed(1)}%`, fontSize: 11, fill: '#374151' }}
                />
                <Bar 
                  dataKey="correctPrecision" 
                  name="Precision" 
                  fill="#8b5cf6" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `${value.toFixed(1)}%`, fontSize: 11, fill: '#374151' }}
                />
                <Bar 
                  dataKey="correctRecall" 
                  name="Recall" 
                  fill="#f59e0b" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `${value.toFixed(1)}%`, fontSize: 11, fill: '#374151' }}
                />
                <Bar 
                  dataKey="correctF1" 
                  name="F1-Score" 
                  fill="#10b981" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `${value.toFixed(1)}%`, fontSize: 11, fill: '#374151' }}
                />
              </>
            )}
            
            {selectedMetric === 'cost' && (
              <>
                <Bar 
                  dataKey="costPerExactMatch" 
                  name="Cost per Exact Match" 
                  fill="#ef4444" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `$${value.toFixed(4)}`, fontSize: 11, fill: '#374151' }}
                />
                <Bar 
                  dataKey="costPerCorrectMatch" 
                  name="Cost per Correct Match" 
                  fill="#f59e0b" 
                  radius={[8, 8, 0, 0]}
                  label={{ position: 'top', formatter: (value) => `$${value.toFixed(4)}`, fontSize: 11, fill: '#374151' }}
                />
              </>
            )}
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">
            <strong>Note:</strong>
            {selectedMetric === 'exact' && ' Exact match means the predicted error step number exactly matches the ground truth. Shows overall accuracy, accuracy for positive cases (error exists), and accuracy for negative cases (no error).'}
            {selectedMetric === 'correct' && ' Correct match means both predictions identify an error (or both identify no error), regardless of exact step number. Shows accuracy, precision, recall, and F1-score.'}
            {selectedMetric === 'cost' && ' Cost efficiency is calculated as total cost divided by number of successful matches. Lower is better.'}
          </p>
        </div>
      </Card>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {data.models.map(model => (
          <Card key={model.name} hover className="bg-gradient-to-br from-white to-gray-50">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-gray-800">{model.displayName}</h3>
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: getModelColor(model.name) }}
                />
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Exact Accuracy:</span>
                  <span className="font-semibold text-gray-800">
                    {formatPercentage(model.exactAccuracy)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Correct Accuracy:</span>
                  <span className="font-semibold text-gray-800">
                    {formatPercentage(model.correctAccuracy)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Cost/Exact:</span>
                  <span className="font-semibold text-primary-600">
                    {formatCurrency(model.costPerExactMatch)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Samples:</span>
                  <span className="font-semibold text-gray-800">{model.totalSamples}</span>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default Overview;
