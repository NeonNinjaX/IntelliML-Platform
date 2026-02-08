'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

interface ModelComparisonProps {
  results: any;
}

const SparkleIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
    <path d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
  </svg>
);

const ChartIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const TrophyIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
    <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
  </svg>
);

const TargetIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

const LightBulbIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

export default function ModelComparison({ results }: ModelComparisonProps) {
  if (!results) return null;

  // Handle various potential API response structures
  const models = results.results?.results || results.results || [];
  const best_model = results.results?.best_model || results.best_model;
  const problem_type = results.results?.problem_type || results.problem_type || results.results?.model_type || results.model_type; // 'classification' or 'regression'
  const explanation = results.explanation;
  const suggestions = results.suggestions || results.results?.suggestions || [];

  if (!models || models.length === 0) {
    return (
      <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 p-6">
        <p className="text-gray-400">No model results available</p>
      </div>
    );
  }

  const isClassification = problem_type === 'classification';

  return (
    <div className="space-y-6 animate-fadeIn">

      {/* 1. KEY METRICS CARDS */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-slate-900 rounded-xl border border-white/5 p-4 shadow-lg shadow-blue-500/5 hover:border-blue-500/30 transition-all">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Best Model</div>
          <div className="flex items-center gap-2 mb-1">
            <TrophyIcon />
            <div className="text-lg font-bold text-white truncate" title={best_model.model_name}>
              {best_model.model_name}
            </div>
          </div>
          <div className="text-xs text-emerald-400 font-mono">Rank #1</div>
        </div>

        <div className="bg-slate-900 rounded-xl border border-white/5 p-4 shadow-lg shadow-blue-500/5 hover:border-blue-500/30 transition-all">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
            {isClassification ? 'Accuracy' : 'R² Score'}
          </div>
          <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
            {(best_model.test_score ?? best_model.score) !== undefined
              ? ((best_model.test_score ?? best_model.score) * (isClassification ? 100 : 1)).toFixed(2)
              : 'N/A'}
            {isClassification && '%'}
          </div>
          <div className="text-xs text-gray-500 mt-1">Primary Metric</div>
        </div>

        <div className="bg-slate-900 rounded-xl border border-white/5 p-4 shadow-lg shadow-blue-500/5 hover:border-blue-500/30 transition-all">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
            {isClassification ? 'F1 Score' : 'RMSE'}
          </div>
          <div className="text-2xl font-bold text-white">
            {best_model.metrics ? (
              isClassification
                ? (best_model.metrics.f1 !== undefined ? (best_model.metrics.f1 * 100).toFixed(2) + '%' : 'N/A')
                : (best_model.metrics.rmse !== undefined ? best_model.metrics.rmse.toFixed(4) : 'N/A')
            ) : 'N/A'}
          </div>
          <div className="text-xs text-gray-500 mt-1">{isClassification ? 'Weighted Average' : 'Root Mean Sq. Error'}</div>
        </div>

        <div className="bg-slate-900 rounded-xl border border-white/5 p-4 shadow-lg shadow-blue-500/5 hover:border-blue-500/30 transition-all">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
            {isClassification ? 'Precision' : 'MAE'}
          </div>
          <div className="text-2xl font-bold text-white">
            {best_model.metrics ? (
              isClassification
                ? (best_model.metrics.precision !== undefined ? (best_model.metrics.precision * 100).toFixed(2) + '%' : 'N/A')
                : (best_model.metrics.mae !== undefined ? best_model.metrics.mae.toFixed(4) : 'N/A')
            ) : 'N/A'}
          </div>
          <div className="text-xs text-gray-500 mt-1">{isClassification ? 'Weighted Precision' : 'Mean Abs. Error'}</div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">

        {/* 2. CONFUSION MATRIX (Classification Only) */}
        {isClassification && best_model.confusion_matrix && (
          <div className="bg-slate-900 rounded-xl border border-white/5 p-6 shadow-lg">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center text-blue-400">
                <TargetIcon />
              </div>
              <h3 className="text-lg font-bold text-white">Confusion Matrix</h3>
            </div>

            <div className="overflow-x-auto">
              <div className="inline-block min-w-full align-middle">
                <table className="min-w-full divide-y divide-white/5 border border-white/5">
                  <thead>
                    <tr>
                      <th className="px-3 py-2 text-xs font-medium text-gray-500 bg-slate-950"></th>
                      {best_model.confusion_matrix_labels?.map((label: string, i: number) => (
                        <th key={i} className="px-3 py-2 text-xs font-medium text-gray-400 bg-slate-950 uppercase">
                          Pred: {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {best_model.confusion_matrix.map((row: any, i: number) => (
                      <tr key={i}>
                        <td className="px-3 py-2 text-xs font-medium text-gray-400 bg-slate-950 uppercase whitespace-nowrap">
                          Actual: {row.Actual}
                        </td>
                        {best_model.confusion_matrix_labels?.map((colLabel: string, j: number) => {
                          const val = row[colLabel];
                          // Simple heatmap intensity logic
                          const maxVal = Math.max(...best_model.confusion_matrix.map((r: any) =>
                            Math.max(...best_model.confusion_matrix_labels.map((c: string) => r[c]))
                          ));
                          const intensity = val / maxVal;
                          const bg = `rgba(6, 182, 212, ${intensity * 0.5})`; // cyan base

                          return (
                            <td key={j} className="px-3 py-2 text-sm text-center text-white" style={{ backgroundColor: val > 0 ? bg : 'transparent' }}>
                              {val}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 3. ROC CURVE (Classification Only) */}
        {isClassification && best_model.roc_curve && best_model.roc_curve.length > 0 && (
          <div className="bg-slate-900 rounded-xl border border-white/5 p-6 shadow-lg">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center text-purple-400">
                <ChartIcon />
              </div>
              <h3 className="text-lg font-bold text-white">ROC Curve</h3>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={best_model.roc_curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                  <XAxis
                    dataKey="x"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                    labelStyle={{ color: '#9ca3af' }}
                    formatter={(value: number) => value.toFixed(3)}
                  />
                  <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#6b7280" strokeDasharray="3 3" />
                  <Line
                    type="monotone"
                    dataKey="y"
                    stroke="#06b6d4"
                    strokeWidth={2}
                    dot={false}
                    name="ROC"
                    activeDot={{ r: 6, fill: '#fff' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* 4. SUGGESTIONS */}
      {suggestions.length > 0 && (
        <div className="bg-gradient-to-r from-amber-950/30 to-slate-900 border border-amber-500/20 rounded-xl p-6 shadow-lg">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center text-amber-500 animate-pulse">
              <LightBulbIcon />
            </div>
            <h3 className="text-lg font-bold text-white">
              Suggestions for Improvement
            </h3>
          </div>
          <ul className="space-y-3">
            {suggestions.map((tip: string, idx: number) => (
              <li key={idx} className="flex items-start gap-3 text-sm text-gray-300">
                <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0"></span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 5. FULL COMPARISON TABLE */}
      <div className="bg-slate-900 rounded-xl border border-blue-500/10 overflow-hidden shadow-lg shadow-blue-500/5">
        <div className="flex items-center gap-3 p-5 border-b border-white/5 bg-slate-950/50">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center text-white shadow-lg shadow-blue-500/20">
            <ChartIcon />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">
              Model Comparison
            </h3>
            <p className="text-xs text-gray-500">Full performance breakdown</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead className="bg-slate-950">
              <tr>
                <th className="px-6 py-4 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Rank
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  Model Architecture
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  {isClassification ? 'Accuracy' : 'R² Score'}
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  {isClassification ? 'F1 Score' : 'RMSE'}
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">
                  {isClassification ? 'Precision' : 'MAE'}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 bg-slate-900">
              {models.map((model: any, idx: number) => (
                <tr
                  key={idx}
                  className={idx === 0 ? 'bg-emerald-500/5 hover:bg-emerald-500/10 transition-colors' : 'hover:bg-slate-800 transition-colors'}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {idx === 0 ? (
                      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center text-white shadow-lg shadow-amber-500/20">
                        <TrophyIcon />
                      </div>
                    ) : (
                      <span className="text-gray-500 font-mono ml-2">#{idx + 1}</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-semibold text-white">
                      {model.model_name}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm font-bold text-white">
                      {isClassification
                        ? ((model.test_score ?? model.score ?? 0) * 100).toFixed(2) + '%'
                        : Number(model.test_score ?? model.score ?? 0).toFixed(4)
                      }
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                    {model.metrics ? (
                      isClassification
                        ? (model.metrics.f1 !== undefined ? (model.metrics.f1 * 100).toFixed(2) + '%' : '-')
                        : (model.metrics.rmse !== undefined ? model.metrics.rmse.toFixed(4) : '-')
                    ) : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                    {model.metrics ? (
                      isClassification
                        ? (model.metrics.precision !== undefined ? (model.metrics.precision * 100).toFixed(2) + '%' : '-')
                        : (model.metrics.mae !== undefined ? model.metrics.mae.toFixed(4) : '-')
                    ) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* AI Explanation Text */}
      {explanation && (
        <div className="bg-gradient-to-r from-indigo-950/50 to-slate-900 border border-indigo-500/20 rounded-xl p-6 shadow-lg shadow-indigo-500/5">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/20">
              <SparkleIcon />
            </div>
            <h3 className="text-lg font-bold text-white">
              AI Explanation
            </h3>
          </div>
          <p className="text-gray-300 leading-relaxed text-sm md:text-base">{explanation}</p>
        </div>
      )}

    </div>
  );
}