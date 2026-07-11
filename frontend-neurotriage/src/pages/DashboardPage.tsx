import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts';
import {
  Activity, Upload, Clock, TrendingUp, Brain, ArrowRight,
} from 'lucide-react';
import { getHistory } from '../lib/api';
import type { AnalysisSummary } from '../types/api';
import { PageTransition } from '../components/PageTransition';
import { RoutingBadge } from '../components/RoutingBanner';
import { Disclaimer } from '../components/Disclaimer';
import { CardSkeleton, ChartSkeleton } from '../components/Skeleton';

const ROUTING_COLORS: Record<string, string> = {
  urgent: '#DC2626',
  'needs-review': '#D97706',
  'auto-clear': '#16A34A',
};

const CLASS_COLORS: Record<string, string> = {
  glioma: '#635BFF',
  meningioma: '#7C6CFF',
  notumor: '#16A34A',
  pituitary: '#D97706',
};

function CustomTooltip({ active, payload, label }: Record<string, unknown>) {
  if (!active || !payload || !(payload as Array<unknown>).length) return null;
  const p = (payload as Array<{ value: number; fill: string }>)[0];
  return (
    <div
      className="px-3 py-2 rounded-xl text-sm"
      style={{
        background: 'var(--surface-elevated)',
        border: '1px solid var(--border)',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        color: 'var(--text-primary)',
      }}
    >
      <span className="font-medium">{label as string}: </span>
      <span style={{ color: p.fill }}>{p.value}</span>
    </div>
  );
}

function PieTooltip({ active, payload }: Record<string, unknown>) {
  if (!active || !payload || !(payload as Array<unknown>).length) return null;
  const p = (payload as Array<{ name: string; value: number; fill: string }>)[0];
  return (
    <div
      className="px-3 py-2 rounded-xl text-sm"
      style={{
        background: 'var(--surface-elevated)',
        border: '1px solid var(--border)',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        color: 'var(--text-primary)',
      }}
    >
      <span className="font-medium capitalize">{p.name}: </span>
      <span style={{ color: p.fill }}>{p.value}</span>
    </div>
  );
}

function AnimatedStat({ value, label, icon }: { value: string | number; label: string; icon: React.ReactNode }) {
  return (
    <motion.div
      className="card p-5"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-start justify-between mb-3">
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center"
          style={{
            background: 'linear-gradient(135deg, rgba(124,108,255,0.15), rgba(79,70,229,0.15))',
            color: 'var(--color-accent)',
          }}
        >
          {icon}
        </div>
      </div>
      <p
        className="text-3xl md:text-4xl font-bold tracking-tight mb-1 gradient-text"
        style={{ fontFamily: 'var(--font-display)' }}
      >
        {value}
      </p>
      <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{label}</p>
    </motion.div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div
        className="w-20 h-20 rounded-2xl flex items-center justify-center mb-6"
        style={{
          background: 'linear-gradient(135deg, rgba(124,108,255,0.12), rgba(79,70,229,0.12))',
          color: 'var(--color-accent)',
        }}
      >
        <Brain size={36} />
      </div>
      <h2
        className="text-xl font-semibold mb-2"
        style={{ fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}
      >
        No analyses yet
      </h2>
      <p className="text-sm mb-6 max-w-xs" style={{ color: 'var(--text-secondary)' }}>
        Upload a brain MRI scan to run the multi-agent triage pipeline and see results here.
      </p>
      <Link
        to="/upload"
        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold text-white transition-all duration-150 active:scale-[0.98]"
        style={{ background: 'linear-gradient(135deg, #7C6CFF, #4F46E5)' }}
      >
        <Upload size={15} />
        Upload your first scan
      </Link>
    </div>
  );
}

const staggerContainer = {
  animate: { transition: { staggerChildren: 0.07 } },
};
const staggerItem = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
};

export function DashboardPage() {
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHistory()
      .then(setHistory)
      .catch(() => setError('Failed to load history. Is the backend running?'))
      .finally(() => setLoading(false));
  }, []);

  const totalRuns = history.length;
  const avgConfidence =
    totalRuns > 0
      ? Math.round((history.reduce((sum, h) => sum + h.top_confidence, 0) / totalRuns) * 100)
      : 0;

  const routingBreakdown = Object.entries(
    history.reduce<Record<string, number>>((acc, h) => {
      acc[h.routing] = (acc[h.routing] ?? 0) + 1;
      return acc;
    }, {})
  ).map(([name, value]) => ({ name, value }));

  const classBreakdown = Object.entries(
    history.reduce<Record<string, number>>((acc, h) => {
      acc[h.top_class] = (acc[h.top_class] ?? 0) + 1;
      return acc;
    }, {})
  ).map(([name, value]) => ({ name, value }));

  const recent = history.slice(0, 5);

  if (loading) {
    return (
      <PageTransition>
        <div className="max-w-7xl mx-auto px-6 md:px-8 py-8">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ChartSkeleton />
            <ChartSkeleton />
          </div>
        </div>
      </PageTransition>
    );
  }

  if (error) {
    return (
      <PageTransition>
        <div className="max-w-7xl mx-auto px-6 md:px-8 py-8">
          <div
            className="card p-8 text-center"
            style={{ borderColor: 'var(--color-danger-border)' }}
          >
            <p className="text-sm font-medium" style={{ color: 'var(--color-danger)' }}>{error}</p>
          </div>
        </div>
      </PageTransition>
    );
  }

  if (totalRuns === 0) {
    return (
      <PageTransition>
        <div className="max-w-7xl mx-auto px-6 md:px-8 py-8">
          <EmptyState />
        </div>
      </PageTransition>
    );
  }

  return (
    <PageTransition>
      <div className="max-w-7xl mx-auto px-6 md:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1
            className="text-2xl md:text-3xl font-bold tracking-tight mb-1"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}
          >
            Dashboard
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Aggregate view of all analysis runs
          </p>
        </div>

        {/* Stats row */}
        <motion.div
          className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6"
          variants={staggerContainer}
          initial="initial"
          animate="animate"
        >
          <motion.div variants={staggerItem}>
            <AnimatedStat value={totalRuns} label="Total analyses" icon={<Activity size={17} />} />
          </motion.div>
          <motion.div variants={staggerItem}>
            <AnimatedStat value={`${avgConfidence}%`} label="Avg. confidence" icon={<TrendingUp size={17} />} />
          </motion.div>
          <motion.div variants={staggerItem}>
            <AnimatedStat
              value={history.filter(h => h.routing === 'urgent').length}
              label="Urgent cases"
              icon={<Brain size={17} />}
            />
          </motion.div>
        </motion.div>

        {/* Charts row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Routing donut */}
          <motion.div
            className="card p-5"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            <h3
              className="text-sm font-semibold mb-4"
              style={{ color: 'var(--text-primary)' }}
            >
              Routing Breakdown
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={routingBreakdown}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {routingBreakdown.map((entry) => (
                    <Cell key={entry.name} fill={ROUTING_COLORS[entry.name] ?? '#A1A1AA'} />
                  ))}
                </Pie>
                <Tooltip content={<PieTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-x-4 gap-y-2 justify-center mt-2">
              {routingBreakdown.map(e => (
                <div key={e.name} className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ROUTING_COLORS[e.name] ?? '#A1A1AA' }} />
                  <span className="text-xs capitalize" style={{ color: 'var(--text-secondary)' }}>
                    {e.name} ({e.value})
                  </span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Classification bar chart */}
          <motion.div
            className="card p-5"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h3
              className="text-sm font-semibold mb-4"
              style={{ color: 'var(--text-primary)' }}
            >
              Classification Breakdown
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={classBreakdown} barSize={28}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                  axisLine={false}
                  tickLine={false}
                  width={24}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {classBreakdown.map((entry) => (
                    <Cell key={entry.name} fill={CLASS_COLORS[entry.name] ?? '#635BFF'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* Recent activity */}
        <motion.div
          className="card overflow-hidden"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
        >
          <div className="flex items-center justify-between px-5 py-4"
            style={{ borderBottom: '1px solid var(--border)' }}
          >
            <div className="flex items-center gap-2">
              <Clock size={15} style={{ color: 'var(--color-accent)' }} />
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                Recent Activity
              </h3>
            </div>
            <Link
              to="/history"
              className="text-xs font-medium flex items-center gap-1 hover:gap-1.5 transition-all"
              style={{ color: 'var(--color-accent)' }}
            >
              View all <ArrowRight size={12} />
            </Link>
          </div>

          <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
            {recent.map((run) => (
              <Link
                key={run.run_id}
                to={`/results/${run.run_id}`}
                className="flex items-center justify-between px-5 py-3.5 hover:bg-[var(--surface-elevated)] transition-colors group"
              >
                <div className="flex items-center gap-3">
                  <div>
                    <p
                      className="text-xs mb-0.5"
                      style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}
                    >
                      {run.run_id.slice(0, 8)}…
                    </p>
                    <p className="text-sm font-medium capitalize" style={{ color: 'var(--text-primary)' }}>
                      {run.top_class}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-sm font-semibold" style={{ color: 'var(--color-accent)' }}>
                    {Math.round(run.top_confidence * 100)}%
                  </span>
                  <RoutingBadge routing={run.routing} />
                  <ArrowRight
                    size={14}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ color: 'var(--text-secondary)' }}
                  />
                </div>
              </Link>
            ))}
          </div>
        </motion.div>

        <Disclaimer />
      </div>
    </PageTransition>
  );
}
