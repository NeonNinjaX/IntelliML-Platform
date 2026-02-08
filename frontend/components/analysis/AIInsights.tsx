'use client';

interface AIInsightsProps {
  insights: string | { insights: string; timestamp: string } | null;
}

const BotIcon = () => (
  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
    <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H6a2 2 0 01-2-2v-1H3a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2zM9 16a1 1 0 100-2 1 1 0 000 2zm6 0a1 1 0 100-2 1 1 0 000 2z" />
  </svg>
);

export default function AIInsights({ insights }: AIInsightsProps) {
  if (!insights) return null;

  const insightText = typeof insights === 'string' ? insights : insights?.insights;

  if (!insightText) return null;

  return (
    <div className="bg-gradient-to-br from-slate-900/50 to-blue-900/50 backdrop-blur-sm rounded-xl border border-blue-500/30 p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white shadow-lg shadow-blue-500/30">
          <BotIcon />
        </div>
        <h3 className="text-lg font-semibold text-white">
          AI-Generated Insights
        </h3>
      </div>

      <div className="prose prose-invert prose-sm max-w-none">
        <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">
          {insightText}
        </p>
      </div>
    </div>
  );
}