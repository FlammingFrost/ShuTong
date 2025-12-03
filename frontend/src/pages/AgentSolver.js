import React, { useState, useEffect, useRef } from 'react';
import Card from '../components/Card';
import Button from '../components/Button';
import Alert from '../components/Alert';
import LatexRenderer from '../components/LatexRenderer';
import { Play, CheckCircle, XCircle, AlertCircle, Clock, TrendingUp, Zap } from 'lucide-react';
import { convertLatex } from '../utils/helpers';
import { api } from '../services/api';

const AgentSolver = () => {
  const [problem, setProblem] = useState('');
  const [maxIterations, setMaxIterations] = useState(3);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  
  // Real-time progress tracking
  const [currentStage, setCurrentStage] = useState(''); // 'initial', 'critique', 'refine'
  const [solutionSteps, setSolutionSteps] = useState([]);
  const [critiques, setCritiques] = useState([]);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [finalResult, setFinalResult] = useState(null);
  const [identifiedErrors, setIdentifiedErrors] = useState([]);
  
  const resultsRef = useRef(null);

  // Load problem from session storage if available
  useEffect(() => {
    const savedProblem = sessionStorage.getItem('currentProblem');
    if (savedProblem) {
      setProblem(savedProblem);
      sessionStorage.removeItem('currentProblem');
    }
  }, []);

  // Auto-scroll to latest update
  useEffect(() => {
    if (resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [solutionSteps, critiques, currentStage, isRunning]);

  const handleRun = async () => {
    if (!problem.trim()) {
      setError('Please enter a math problem first!');
      return;
    }

    setIsRunning(true);
    setError(null);
    setCurrentStage('initializing');
    setSolutionSteps([]);
    setCritiques([]);
    setCurrentIteration(0);
    setFinalResult(null);
    setIdentifiedErrors([]);

    try {
      // Use real API streaming
      await api.streamAgentProgress(problem, maxIterations, handleProgressUpdate);
      
    } catch (err) {
      setError('Failed to run agent. Please try again.');
      console.error(err);
    } finally {
      setIsRunning(false);
      setCurrentStage('completed');
    }
  };

  const handleLoadDemo = async () => {
    setIsRunning(true);
    setError(null);
    setCurrentStage('initializing');
    setSolutionSteps([]);
    setCritiques([]);
    setCurrentIteration(0);
    setFinalResult(null);
    setIdentifiedErrors([]);

    try {
      // Load pre-loaded demo result
      const result = await api.loadDemoResult();
      
      // Set the problem
      if (result.math_problem) {
        setProblem(result.math_problem);
      }
      
      // Simulate quick progression through stages
      setCurrentStage('initial');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Set solution steps
      setSolutionSteps(result.solution_steps || []);
      await new Promise(resolve => setTimeout(resolve, 300));
      
      setCurrentStage('critique');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Set critiques
      setCritiques(result.all_critiques || []);
      await new Promise(resolve => setTimeout(resolve, 300));
      
      setCurrentStage('completed');
      
      // Set final result
      setFinalResult({
        final_solution: result.final_solution || '',
        iteration_count: result.iteration_count || 1,
        knowledge_points: result.knowledge_points || []
      });
      
      setCurrentIteration(result.iteration_count || 1);
      
    } catch (err) {
      setError('Failed to load demo. Please try again.');
      console.error(err);
    } finally {
      setIsRunning(false);
    }
  };

  const handleProgressUpdate = (data) => {
    const { stage } = data;
    
    console.log('Progress update:', stage, data); // Debug logging

    switch (stage) {
      case 'initializing':
        setCurrentStage('initializing');
        console.log('Stage: Initializing');
        break;
      
      case 'initial':
        setCurrentStage('initial');
        console.log('Stage: Generating initial solution');
        break;
      
      case 'solution_step':
        // Add solution step progressively in real-time
        const newStep = data.step;
        console.log('Received solution step:', newStep.step_number, newStep.description);
        setSolutionSteps(prev => {
          const existing = prev.find(s => s.step_number === newStep.step_number);
          if (existing) {
            // Update existing step
            console.log('Updating existing step:', newStep.step_number);
            return prev.map(s => s.step_number === newStep.step_number ? newStep : s);
          }
          // Add new step
          console.log('Adding new step:', newStep.step_number);
          return [...prev, newStep];
        });
        break;
      
      case 'critique':
        setCurrentStage('critique');
        setCurrentIteration(data.iteration || 1);
        console.log('Stage: Critiquing, iteration:', data.iteration);
        break;
      
      case 'critique_result':
        // Add critique result in real-time
        const newCritique = data.critique;
        console.log('Received critique for step:', newCritique.step_number, 
          'Correct:', newCritique.is_logically_correct && newCritique.is_calculation_correct);
        setCritiques(prev => {
          const existing = prev.find(c => c.step_number === newCritique.step_number);
          if (existing) {
            // Update existing critique
            console.log('Updating existing critique:', newCritique.step_number);
            return prev.map(c => c.step_number === newCritique.step_number ? newCritique : c);
          }
          // Add new critique
          console.log('Adding new critique:', newCritique.step_number);
          return [...prev, newCritique];
        });
        break;
      
      case 'refine':
        setCurrentStage('refine');
        setCurrentIteration(data.iteration || 2);
        console.log('Stage: Refining, iteration:', data.iteration);
        // Store identified errors from previous iteration
        if (data.errors && data.errors.length > 0) {
          setIdentifiedErrors(data.errors);
          console.log('Identified errors:', data.errors);
        } else {
          // Extract errors from current critiques if not provided
          const errors = critiques
            .filter(c => !c.is_logically_correct || !c.is_calculation_correct)
            .map(c => {
              const issues = [];
              if (!c.is_logically_correct) {
                issues.push(`Step ${c.step_number} - Logic Issue: ${c.logic_feedback}`);
              }
              if (!c.is_calculation_correct) {
                issues.push(`Step ${c.step_number} - Calculation Issue: ${c.calculation_feedback}`);
              }
              return issues.join('; ');
            });
          setIdentifiedErrors(errors);
        }
        // Clear previous steps and critiques for refinement
        setSolutionSteps([]);
        setCritiques([]);
        break;
      
      case 'completed':
        setCurrentStage('completed');
        console.log('Stage: Completed');
        if (data.result) {
          // Set final result with all data
          setFinalResult({
            final_solution: data.result.final_solution || '',
            iteration_count: data.result.iteration_count || currentIteration,
            knowledge_points: data.result.knowledge_points || [],
          });
          
          // Ensure we have all final steps
          if (data.result.solution_steps && data.result.solution_steps.length > 0) {
            setSolutionSteps(data.result.solution_steps);
          }
          
          // Ensure we have all final critiques
          if (data.result.all_critiques && data.result.all_critiques.length > 0) {
            setCritiques(data.result.all_critiques);
          }
        }
        break;
      
      case 'error':
        setError(data.error || 'An error occurred during processing');
        console.error('Error:', data.error);
        break;
      
      default:
        console.log('Unknown stage:', stage, data);
    }
  };

  const getStageDisplay = () => {
    const stages = {
      initializing: { icon: Clock, text: 'Initializing agent...', color: 'text-blue-500' },
      initial: { icon: Play, text: 'Generating initial solution...', color: 'text-primary-500' },
      critique: { icon: AlertCircle, text: `Analyzing solution (Iteration ${currentIteration})...`, color: 'text-accent-500' },
      refine: { icon: TrendingUp, text: `Refining solution (Iteration ${currentIteration})...`, color: 'text-yellow-500' },
      completed: { icon: CheckCircle, text: 'Solution completed!', color: 'text-green-500' },
    };

    const stage = stages[currentStage] || stages.initializing;
    const Icon = stage.icon;

    return (
      <div className="flex items-center space-x-3">
        <Icon className={`${stage.color} animate-pulse`} size={24} />
        <span className="font-semibold text-gray-700">{stage.text}</span>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto space-y-2">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-primary-500 to-accent-500 rounded-2xl mb-1 shadow-lg">
          <Play className="text-white" size={32} />
        </div>
        <h1 className="text-3xl font-bold gradient-text mb-1">Agent Solver</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Enter a math problem and watch the AI agent solve it step-by-step with real-time critique and refinement.
        </p>
      </div>

      {/* Problem Input */}
      <Card className="bg-gradient-to-br from-white to-primary-50">
        <h2 className="text-xl font-bold text-gray-800 mb-1">Math Problem</h2>
        
        <div className="space-y-2">
          <textarea
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            placeholder="Enter your math problem here (supports LaTeX with $ and $$)..."
            rows={5}
            disabled={isRunning}
            className="w-full px-4 py-2 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:outline-none transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
          />

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <label className="text-sm font-semibold text-gray-700">
                Max Refinement Iterations:
              </label>
              <select
                value={maxIterations}
                onChange={(e) => setMaxIterations(Number(e.target.value))}
                disabled={isRunning}
                className="px-3 py-2 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:outline-none bg-white disabled:bg-gray-100 disabled:cursor-not-allowed"
              >
                {[1, 2, 3, 4, 5].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>

            <div className="flex gap-3">
              <Button
                onClick={handleLoadDemo}
                disabled={isRunning}
                variant="outline"
                size="lg"
              >
                <Zap size={20} />
                <span>Load Demo</span>
              </Button>
              
              <Button
                onClick={handleRun}
                disabled={isRunning || !problem.trim()}
                loading={isRunning}
                size="lg"
              >
                <Play size={20} />
                <span>Run Agent</span>
              </Button>
            </div>
          </div>
        </div>
      </Card>

      {/* Error Alert */}
      {error && <Alert type="error" message={error} />}

      {/* Progress Indicator */}
      {isRunning && (
        <Card className="bg-gradient-to-r from-primary-50 to-accent-50">
          {getStageDisplay()}
          <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-gradient-to-r from-primary-500 to-accent-500 h-2 rounded-full transition-all duration-500 animate-pulse"
              style={{
                width: currentStage === 'initializing' ? '20%' :
                       currentStage === 'initial' ? '40%' :
                       currentStage === 'critique' ? '60%' :
                       currentStage === 'refine' ? '80%' :
                       '100%'
              }}
            />
          </div>
          
          {/* Show identified errors during refinement */}
          {currentStage === 'refine' && identifiedErrors.length > 0 && (
            <div className="mt-2 p-3 bg-red-50 border-l-4 border-red-400 rounded">
              <h4 className="font-semibold text-red-800 mb-2 flex items-center">
                <XCircle size={18} className="mr-2" />
                Identified Issues in Previous Iteration
              </h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-red-700">
                {identifiedErrors.map((error, idx) => (
                  <li key={idx}>{error}</li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      )}

      {/* Real-time Solution Steps */}
      {solutionSteps.length > 0 && (
        <Card>
          <h2 className="text-xl font-bold text-gray-800 mb-2">📝 Solution Steps</h2>
          
          <div className="space-y-2">
            {solutionSteps.map((step, idx) => {
              const critique = critiques.find(c => c.step_number === step.step_number);
              
              return (
                <div key={step.step_number} className="border-l-4 border-primary-300 pl-3 py-1">
                  <h3 className="text-base font-semibold text-gray-800 mb-1">
                    Step {step.step_number}: {step.description}
                  </h3>
                  
                  <div className="bg-gray-50 p-2 rounded-lg mb-1">
                    <LatexRenderer content={convertLatex(step.content)} />
                  </div>

                  {/* Critique Display */}
                  {critique && (
                    <div className="ml-3 p-2 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200 animate-fadeIn">
                      <h4 className="font-semibold text-gray-800 mb-1 flex items-center">
                        <AlertCircle size={18} className="mr-2 text-accent-500" />
                        Critique Analysis
                      </h4>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                        <div>
                          <div className="flex items-center space-x-2 mb-2">
                            {critique.is_logically_correct ? (
                              <CheckCircle size={16} className="text-green-500" />
                            ) : (
                              <XCircle size={16} className="text-red-500" />
                            )}
                            <span className="font-semibold">Logic:</span>
                          </div>
                          <p className="text-gray-700">{critique.logic_feedback}</p>
                        </div>
                        
                        <div>
                          <div className="flex items-center space-x-2 mb-2">
                            {critique.is_calculation_correct ? (
                              <CheckCircle size={16} className="text-green-500" />
                            ) : (
                              <XCircle size={16} className="text-red-500" />
                            )}
                            <span className="font-semibold">Calculation:</span>
                          </div>
                          <p className="text-gray-700">{critique.calculation_feedback}</p>
                        </div>
                      </div>

                      {critique.knowledge_points && critique.knowledge_points.length > 0 && (
                        <div className="mt-1 pt-1 border-t border-blue-200">
                          <p className="text-xs font-semibold text-gray-600 mb-1">Knowledge Points:</p>
                          <div className="flex flex-wrap gap-2">
                            {critique.knowledge_points.map((kp, i) => (
                              <span key={i} className="px-2 py-1 bg-white rounded-full text-xs text-gray-700 border border-blue-200">
                                {kp}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Final Result Summary */}
      {finalResult && (
        <Card className="bg-gradient-to-br from-green-50 to-blue-50 border-2 border-green-200">
          <div className="flex items-center space-x-3 mb-2">
            <CheckCircle className="text-green-500" size={28} />
            <h2 className="text-xl font-bold text-gray-800">Solution Complete!</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-2">
            <div className="bg-white p-2 rounded-lg shadow-sm">
              <p className="text-sm text-gray-600 mb-1">Total Steps</p>
              <p className="text-xl font-bold text-primary-600">{solutionSteps.length}</p>
            </div>
            <div className="bg-white p-3 rounded-lg shadow-sm">
              <p className="text-sm text-gray-600 mb-1">Refinement Iterations</p>
              <p className="text-xl font-bold text-accent-600">{finalResult.iteration_count}</p>
            </div>
            <div className="bg-white p-3 rounded-lg shadow-sm">
              <p className="text-sm text-gray-600 mb-1">Knowledge Points</p>
              <p className="text-xl font-bold text-green-600">{finalResult.knowledge_points.length}</p>
            </div>
          </div>

          <div className="bg-white p-3 rounded-lg">
            <h3 className="text-base font-semibold text-gray-800 mb-2">📚 Knowledge Points Summary</h3>
            <div className="flex flex-wrap gap-2">
              {finalResult.knowledge_points.map((kp, i) => (
                <span key={i} className="px-3 py-2 bg-gradient-to-r from-primary-100 to-accent-100 rounded-lg text-sm font-medium text-gray-700">
                  {kp}
                </span>
              ))}
            </div>
          </div>
        </Card>
      )}

      <div ref={resultsRef} />
    </div>
  );
};

export default AgentSolver;
