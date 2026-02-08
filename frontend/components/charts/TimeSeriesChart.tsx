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
  Area,
  ComposedChart,
} from 'recharts';

interface TimeSeriesChartProps {
  data: {
    date_column: string;
    value_column: string;
    dates: string[];
    values: number[];
  };
}

import { downloadChart } from '@/lib/downloadChart';

export default function TimeSeriesChart({ data }: TimeSeriesChartProps) {
  const chartId = `chart-ts-${data.date_column}-${data.value_column}`.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-_]/g, '');

  const chartData = data.dates.map((date, idx) => ({
    date: date,
    value: data.values[idx],
  }));

  // Sample data if too many points
  const sampledData =
    chartData.length > 100
      ? chartData.filter((_, idx) => idx % Math.ceil(chartData.length / 100) === 0)
      : chartData;

  return (
    <div id={chartId} className="bg-slate-900 rounded-xl border border-blue-500/10 p-5 shadow-lg shadow-blue-500/5">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h4 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="w-1 h-5 bg-indigo-500 rounded-full"></span>
            Time Series Analysis
          </h4>
          <p className="text-xs text-gray-500 ml-3">{data.value_column} over {data.date_column}</p>
        </div>
        <button
          onClick={() => downloadChart(chartId, `timeseries-${data.value_column}`)}
          className="p-2 hover:bg-white/5 rounded-lg text-gray-400 hover:text-white transition-colors"
          title="Download Chart"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        </button>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={sampledData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            angle={-45}
            textAnchor="end"
            height={60}
            stroke="#334155"
          />
          <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} stroke="#334155" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f172a',
              border: '1px solid #1e293b',
              borderRadius: '8px',
              color: '#f8fafc',
            }}
            itemStyle={{ color: '#cbd5e1' }}
          />
          <Legend />
          <Area
            type="monotone"
            dataKey="value"
            fill="#8b5cf6"
            fillOpacity={0.1}
            stroke="none"
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#8b5cf6"
            strokeWidth={2}
            dot={false}
            name={data.value_column}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}