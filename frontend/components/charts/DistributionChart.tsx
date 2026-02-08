'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from 'recharts';

interface DistributionChartProps {
  data: {
    column: string;
    bins: number[];
    counts: number[];
    mean: number;
    median: number;
  };
}

import { downloadChart } from '@/lib/downloadChart';

export default function DistributionChart({ data }: DistributionChartProps) {
  const chartId = `chart-dist-${data.column.replace(/\s+/g, '-')}`;

  // Transform data for Recharts
  const chartData = data.bins.map((bin, idx) => ({
    bin: bin.toFixed(2),
    count: data.counts[idx],
    binValue: bin,
  }));

  return (
    <div id={chartId} className="bg-slate-900 rounded-xl border border-blue-500/10 p-5 shadow-lg shadow-blue-500/5">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h4 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="w-1 h-5 bg-blue-500 rounded-full"></span>
            {data.column}
          </h4>
          <p className="text-xs text-gray-500 ml-3">Distribution Analysis</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center space-x-3 text-xs">
            <div className="flex flex-col items-end">
              <span className="text-gray-500 uppercase tracking-wider font-bold text-[10px]">Mean</span>
              <span className="text-blue-400 font-mono font-medium">{data.mean.toFixed(2)}</span>
            </div>
            <div className="w-px h-6 bg-white/10"></div>
            <div className="flex flex-col items-end">
              <span className="text-gray-500 uppercase tracking-wider font-bold text-[10px]">Median</span>
              <span className="text-cyan-400 font-mono font-medium">{data.median.toFixed(2)}</span>
            </div>
          </div>
          <button
            onClick={() => downloadChart(chartId, `distribution-${data.column}`)}
            className="p-2 hover:bg-white/5 rounded-lg text-gray-400 hover:text-white transition-colors"
            title="Download Chart"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="bin"
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            angle={-45}
            textAnchor="end"
            height={60}
            stroke="#334155"
          />
          <YAxis
            tick={{ fontSize: 11, fill: '#94a3b8' }}
            stroke="#334155"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f172a',
              border: '1px solid #1e293b',
              borderRadius: '8px',
              color: '#f8fafc',
              boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
            }}
            itemStyle={{ color: '#cbd5e1' }}
            labelStyle={{ color: '#94a3b8', marginBottom: '0.5rem' }}
          />
          <Bar dataKey="count" fill="#3b82f6" opacity={0.8} radius={[4, 4, 0, 0]} name="Frequency" />
          <Line
            type="monotone"
            dataKey="count"
            stroke="#22d3ee"
            strokeWidth={3}
            dot={{ r: 4, fill: '#22d3ee', strokeWidth: 0 }}
            activeDot={{ r: 6, fill: '#fff' }}
            name="Trend"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}