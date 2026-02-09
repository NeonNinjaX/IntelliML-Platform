'use client';

import { useState } from 'react';
import { analyzeBusinessProblem, refineBusinessContext } from '@/lib/api';

interface BusinessProblemWizardProps {
  onComplete: (context: any) => void;
}

export default function BusinessProblemWizard({ onComplete }: BusinessProblemWizardProps) {
  const [step, setStep] = useState<'describe' | 'questions' | 'summary'>('describe');
  const [description, setDescription] = useState('');
  const [analysis, setAnalysis] = useState<any>(null);
  const [questions, setQuestions] = useState<string[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [isProcessing, setIsProcessing] = useState(false);

  const handleDescribe = async () => {
    if (!description.trim()) return;

    setIsProcessing(true);
    try {
      const result = await analyzeBusinessProblem(description);
      setAnalysis(result.analysis);
      setQuestions(result.questions);
      setStep('questions');
    } catch (error) {
      console.error('Failed to analyze problem:', error);
      alert('Failed to analyze problem. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAnswerChange = (question: string, answer: string) => {
    setAnswers(prev => ({ ...prev, [question]: answer }));
  };

  const handleComplete = async () => {
    setIsProcessing(true);
    try {
      const result = await refineBusinessContext(answers);
      setAnalysis(result.context);
      setStep('summary');
    } catch (error) {
      console.error('Failed to refine context:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleProceed = () => {
    onComplete(analysis);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress Indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <StepIndicator active={step === 'describe'} completed={step !== 'describe'} label="Describe Problem" />
          <div className="flex-1 h-1 bg-gray-300 mx-2"></div>
          <StepIndicator active={step === 'questions'} completed={step === 'summary'} label="Answer Questions" />
          <div className="flex-1 h-1 bg-gray-300 mx-2"></div>
          <StepIndicator active={step === 'summary'} completed={false} label="Review & Start" />
        </div>
      </div>

      {/* Step 1: Describe Problem */}
      {step === 'describe' && (
        <div className="card animate-fadeIn">
          <div className="text-center mb-6">
            <div className="text-6xl mb-4">🎯</div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              What's Your Business Challenge?
            </h2>
            <p className="text-gray-600">
              Describe your business problem in your own words. I'll help you solve it with data.
            </p>
          </div>

          <div className="space-y-4">
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Example: We're losing customers every month and need to understand why and predict which customers are likely to leave..."
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
              rows={6}
            />

            <button
              onClick={handleDescribe}
              disabled={!description.trim() || isProcessing}
              className="w-full btn-primary text-lg py-4 disabled:opacity-50"
            >
              {isProcessing ? (
                <span className="flex items-center justify-center">
                  <span className="animate-spin mr-2">⚙️</span>
                  Analyzing...
                </span>
              ) : (
                '🔍 Analyze My Problem'
              )}
            </button>
          </div>

          {/* Examples */}
          <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2">💡 Example Problems:</h4>
            <ul className="space-y-1 text-sm text-blue-800">
              <li>• "Predict which customers will churn next month"</li>
              <li>• "Forecast sales for the next quarter"</li>
              <li>• "Detect fraudulent transactions in real-time"</li>
              <li>• "Optimize product pricing for maximum revenue"</li>
              <li>• "Segment customers for targeted marketing"</li>
            </ul>
          </div>
        </div>
      )}

      {/* Step 2: Answer Questions */}
      {step === 'questions' && (
        <div className="card animate-fadeIn">
          <div className="text-center mb-6">
            <div className="text-6xl mb-4">❓</div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Help Me Understand Better
            </h2>
            <p className="text-gray-600">
              Answer these questions so I can tailor the solution to your needs
            </p>
          </div>

          <div className="space-y-6">
            {questions.map((question, idx) => (
              <div key={idx} className="space-y-2">
                <label className="block font-medium text-gray-900">
                  {idx + 1}. {question}
                </label>
                <textarea
                  value={answers[question] || ''}
                  onChange={(e) => handleAnswerChange(question, e.target.value)}
                  placeholder="Your answer..."
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 resize-none"
                  rows={3}
                />
              </div>
            ))}
          </div>

          <div className="mt-6 flex space-x-4">
            <button
              onClick={() => setStep('describe')}
              className="flex-1 px-6 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded-lg"
            >
              ← Back
            </button>
            <button
              onClick={handleComplete}
              disabled={isProcessing}
              className="flex-1 btn-primary py-3"
            >
              {isProcessing ? 'Processing...' : 'Continue →'}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Summary */}
      {step === 'summary' && analysis && (
        <div className="space-y-6 animate-fadeIn">
          <div className="card bg-gradient-to-r from-green-50 to-blue-50 border-2 border-green-200">
            <div className="text-center mb-6">
              <div className="text-6xl mb-4">✅</div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Perfect! Here's What I Understand
              </h2>
            </div>

            <div className="space-y-4">
              <InfoCard icon="🏢" label="Industry" value={analysis.industry} />
              <InfoCard icon="🎯" label="Problem Type" value={analysis.problem_type?.replace(/_/g, ' ')} />
              <InfoCard icon="💡" label="Business Goal" value={analysis.business_goal} />
              <InfoCard icon="📈" label="Success Metric" value={analysis.success_metric} />
              <InfoCard icon="🎲" label="Target Variable" value={analysis.target_variable} />
            </div>

            {analysis.recommended_approach && (
              <div className="mt-6 p-4 bg-white rounded-lg border border-blue-200">
                <h4 className="font-semibold text-blue-900 mb-2">🔍 Recommended Approach:</h4>
                <p className="text-gray-700">{analysis.recommended_approach}</p>
              </div>
            )}

            {analysis.data_requirements && (
              <div className="mt-4 p-4 bg-white rounded-lg border border-purple-200">
                <h4 className="font-semibold text-purple-900 mb-2">📋 Data You'll Need:</h4>
                <ul className="space-y-1">
                  {analysis.data_requirements.map((req: string, idx: number) => (
                    <li key={idx} className="text-gray-700 text-sm">• {req}</li>
                  ))}
                </ul>
              </div>
            )}

            {analysis.potential_challenges && (
              <div className="mt-4 p-4 bg-white rounded-lg border border-yellow-200">
                <h4 className="font-semibold text-yellow-900 mb-2">⚠️ Potential Challenges:</h4>
                <ul className="space-y-1">
                  {analysis.potential_challenges.map((challenge: string, idx: number) => (
                    <li key={idx} className="text-gray-700 text-sm">• {challenge}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="flex space-x-4">
            <button
              onClick={() => setStep('questions')}
              className="px-6 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded-lg"
            >
              ← Modify Answers
            </button>
            <button
              onClick={handleProceed}
              className="flex-1 btn-primary text-lg py-4"
            >
              🚀 Start Building Solution
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function StepIndicator({ active, completed, label }: { active: boolean; completed: boolean; label: string }) {
  return (
    <div className="flex flex-col items-center">
      <div className={`
        w-12 h-12 rounded-full flex items-center justify-center font-bold
        ${completed ? 'bg-green-500 text-white' : active ? 'bg-purple-600 text-white' : 'bg-gray-300 text-gray-600'}
      `}>
        {completed ? '✓' : active ? '●' : '○'}
      </div>
      <div className="mt-2 text-xs font-medium text-gray-600">{label}</div>
    </div>
  );
}

function InfoCard({ icon, label, value }: { icon: string; label: string; value: string }) {
  return (
    <div className="flex items-start space-x-3 p-3 bg-white rounded-lg border border-gray-200">
      <span className="text-2xl">{icon}</span>
      <div className="flex-1">
        <div className="text-sm font-semibold text-gray-600">{label}</div>
        <div className="text-lg text-gray-900 capitalize">{value}</div>
      </div>
    </div>
  );
}