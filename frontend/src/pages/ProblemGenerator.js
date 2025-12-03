import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Card from '../components/Card';
import Button from '../components/Button';
import Alert from '../components/Alert';
import LatexRenderer from '../components/LatexRenderer';
import LoadingSpinner from '../components/LoadingSpinner';
import { Sparkles, Copy, Send, RotateCw, PenTool } from 'lucide-react';
import { convertLatex } from '../utils/helpers';
import { api } from '../services/api';

const ProblemGenerator = () => {
  const navigate = useNavigate();
  const [topic, setTopic] = useState('');
  const [difficulty, setDifficulty] = useState('Undergraduate');
  const [problemType, setProblemType] = useState('Proof');
  const [generatedProblem, setGeneratedProblem] = useState('');
  const [editedProblem, setEditedProblem] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const difficultyLevels = [
    'Undergraduate',
    'Graduate',
    'Advanced Graduate',
    'Research Level',
  ];

  const problemTypes = [
    'Proof',
    'Calculation',
    'Application',
    'Conceptual',
    'Mixed',
  ];

  const handleGenerate = async () => {
    if (!topic.trim()) {
      setError('Please enter a topic first!');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      // Call the real API to generate problem using GPT-5-mini
      const result = await api.generateProblem(topic, difficulty, problemType);
      
      setGeneratedProblem(result.problem);
      setEditedProblem(result.problem);
      setSuccess(true);
    } catch (err) {
      setError('Failed to generate problem. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(editedProblem);
    alert('Problem copied to clipboard!');
  };

  const handleSendToSolver = () => {
    if (!editedProblem.trim()) return;
    // Store problem in sessionStorage or state management
    sessionStorage.setItem('currentProblem', editedProblem);
    navigate('/solver');
  };

  const handleReset = () => {
    setGeneratedProblem('');
    setEditedProblem('');
    setTopic('');
    setError(null);
    setSuccess(false);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-accent-500 to-primary-500 rounded-2xl mb-4 shadow-lg">
          <PenTool className="text-white" size={32} />
        </div>
        <h1 className="text-4xl font-bold gradient-text mb-4">Math Problem Generator</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Generate challenging math problems on any topic using AI. Customize difficulty level
          and problem type to match your needs.
        </p>
      </div>

      {/* Input Form */}
      <Card className="bg-gradient-to-br from-white to-primary-50">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <Sparkles className="mr-2 text-accent-500" size={24} />
          Problem Configuration
        </h2>

        <div className="space-y-6">
          {/* Topic Input */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Math Topic or Concept *
            </label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., probability theory, calculus, linear algebra, differential equations"
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:outline-none transition-colors"
            />
          </div>

          {/* Difficulty and Type */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Difficulty Level */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Difficulty Level
              </label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:outline-none transition-colors bg-white"
              >
                {difficultyLevels.map(level => (
                  <option key={level} value={level}>{level}</option>
                ))}
              </select>
            </div>

            {/* Problem Type */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Problem Type
              </label>
              <select
                value={problemType}
                onChange={(e) => setProblemType(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:outline-none transition-colors bg-white"
              >
                {problemTypes.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Generate Button */}
          <div className="flex gap-3">
            <Button
              onClick={handleGenerate}
              disabled={loading || !topic.trim()}
              loading={loading}
              className="flex-1"
              size="lg"
            >
              <Sparkles size={20} />
              <span>Generate Problem</span>
            </Button>
            
            {generatedProblem && (
              <Button
                onClick={handleReset}
                variant="outline"
                size="lg"
              >
                <RotateCw size={20} />
              </Button>
            )}
          </div>
        </div>
      </Card>

      {/* Alerts */}
      {error && <Alert type="error" message={error} />}
      {success && <Alert type="success" message="Problem generated successfully! You can edit it below." />}

      {/* Generated Problem */}
      {loading && (
        <Card className="flex items-center justify-center py-12">
          <LoadingSpinner size="lg" text="Generating math problem..." />
        </Card>
      )}

      {generatedProblem && !loading && (
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-gray-800">Generated Problem</h2>
            <div className="flex gap-2">
              <Button onClick={handleCopy} variant="outline" size="sm">
                <Copy size={16} />
                <span>Copy</span>
              </Button>
              <Button onClick={handleSendToSolver} variant="primary" size="sm">
                <Send size={16} />
                <span>Send to Solver</span>
              </Button>
            </div>
          </div>

          {/* Rendered Preview */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-gray-700">Rendered Preview</h3>
              <span className="text-xs text-gray-500 bg-blue-50 px-3 py-1 rounded-full">
                LaTeX Rendering
              </span>
            </div>
            <div className="p-6 bg-gradient-to-br from-gray-50 to-blue-50 rounded-lg border-2 border-blue-100">
              <LatexRenderer content={convertLatex(editedProblem)} className="text-gray-800 leading-relaxed" />
            </div>
          </div>

          {/* Editable Text Area */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Edit Problem (supports LaTeX with $ and $$)
            </label>
            <textarea
              value={editedProblem}
              onChange={(e) => setEditedProblem(e.target.value)}
              rows={12}
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:outline-none transition-colors font-mono text-sm"
            />
          </div>
        </Card>
      )}

      {/* Example Topics */}
      {!generatedProblem && !loading && (
        <Card className="bg-gradient-to-br from-accent-50 to-primary-50">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">💡 Example Topics</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              'Probability Theory',
              'Linear Algebra',
              'Calculus',
              'Differential Equations',
              'Real Analysis',
              'Complex Analysis',
              'Number Theory',
              'Abstract Algebra',
              'Topology',
            ].map(exampleTopic => (
              <button
                key={exampleTopic}
                onClick={() => setTopic(exampleTopic)}
                className="px-4 py-2 bg-white rounded-lg text-sm font-medium text-gray-700 hover:bg-primary-100 hover:text-primary-700 transition-colors shadow-sm hover:shadow-md"
              >
                {exampleTopic}
              </button>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default ProblemGenerator;
