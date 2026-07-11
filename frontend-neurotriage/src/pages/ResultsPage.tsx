import { useEffect, useState } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart, Bar, Cell, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts';
import {
  Brain, Bot, FileText, GitMerge, Server,
  CheckCircle, AlertCircle, RefreshCw, ImageIcon, ChevronRight,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getResult } from '../lib/api';
import type { AnalysisResult } from '../types/api';
import { PageTransition } from '../components/PageTransition';
import { RoutingBanner } from '../components/RoutingBanner';
import { Disclaimer } from '../components/Disclaimer';
import { CardSkeleton, ChartSkeleton } from '../components/Skeleton';

const CLASS_COLORS: Record<string, string> = {
  glioma: '#635BFF',
  meningioma: '#7C6CFF',
  notumor: '#16A34A',
  pituitary: '#D97706',
};

const AGENT_ICONS: Record<string, React.ReactNode> = {
  vision: <Brain size={16} />,
  retrieval: <GitMerge size={16} />,
  drafting: <FileText size={16} />,
  critic: <CheckCircle size={16} />,
  orchestrator: <Server size={16} />,
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
      <span className="font-medium capitalize">{label as string}: </span>
      <span style={{ color: p.fill }}>{(p.value * 100).toFixed(1)}%</span>
    </div>
  );
}

interface TabsProps {
  tabs: { id: string; label: string; icon: React.ReactNode }[];
  activeTab: string;
  onChange: (id: string) => void;
}

function Tabs({ tabs, activeTab, onChange }: TabsProps) {
  return (
    <div
      className="flex items-center gap-1 p-1 rounded-xl"
      style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)' }}
      role="tablist"
    >
      {tabs.map(tab => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          role="tab"
          aria-selected={tab.id === activeTab}
          className="relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-150 select-none z-10"
          style={{
            color: tab.id === activeTab ? 'var(--text-primary)' : 'var(--text-secondary)',
          }}
        >
          {tab.id === activeTab && (
            <motion.div
              layoutId="tab-indicator"
              className="absolute inset-0 rounded-lg"
              style={{
                background: 'var(--surface)',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              }}
              transition={{ type: 'spring', bounce: 0.2, duration: 0.35 }}
            />
          )}
          <span className="relative z-10 flex items-center gap-1.5">
            {tab.icon}
            {tab.label}
          </span>
        </button>
      ))}
    </div>
  );
}

interface ReasoningTraceProps {
  trace: AnalysisResult['reasoning_trace'];
}

function ReasoningTrace({ trace }: ReasoningTraceProps) {
  if (trace.length === 0) {
    return (
      <div className="flex flex-col items-center py-12 text-center">
        <Bot size={32} style={{ color: 'var(--text-secondary)' }} className="mb-3 opacity-40" />
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>No reasoning trace available</p>
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Vertical line */}
      <div
        className="absolute left-5 top-5 bottom-5 w-0.5"
        style={{ background: `linear-gradient(to bottom, var(--color-accent), transparent)` }}
      />
      <div className="space-y-6">
        {trace.map((step, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.08 }}
            className="flex gap-4"
          >
            {/* Agent icon circle */}
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 relative z-10"
              style={{
                background: 'linear-gradient(135deg, rgba(124,108,255,0.15), rgba(79,70,229,0.15))',
                border: '1px solid rgba(99,91,255,0.3)',
                color: 'var(--color-accent)',
              }}
            >
              {AGENT_ICONS[step.agent.toLowerCase()] ?? <Bot size={16} />}
            </div>

            <div className="flex-1 min-w-0 pb-2">
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="text-xs font-semibold uppercase tracking-wider"
                  style={{ color: 'var(--color-accent)' }}
                >
                  {step.agent}
                </span>
                <ChevronRight size={12} style={{ color: 'var(--text-secondary)' }} />
              </div>
              <p className="text-sm font-medium mb-1.5" style={{ color: 'var(--text-primary)' }}>
                {step.summary}
              </p>
              <p
                className="text-xs p-2.5 rounded-lg"
                style={{
                  color: 'var(--text-secondary)',
                  background: 'var(--surface-elevated)',
                  border: '1px solid var(--border)',
                  fontFamily: 'var(--font-mono)',
                  lineHeight: 1.6,
                }}
              >
                {step.key_evidence}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

interface SystemMetadataProps {
  result: AnalysisResult;
}

function SystemMetadata({ result }: SystemMetadataProps) {
  const modelEntries = Object.entries(result.models_used);

  return (
    <div className="space-y-5">
      {/* Models grid */}
      <div>
        <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-secondary)' }}>
          Models Used
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {modelEntries.map(([role, model]) => (
            <div
              key={role}
              className="flex items-start gap-2.5 p-3 rounded-xl"
              style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)' }}
            >
              <div style={{ color: 'var(--color-accent)' }} className="mt-0.5">
                {AGENT_ICONS[role] ?? <Server size={14} />}
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold capitalize" style={{ color: 'var(--text-primary)' }}>{role}</p>
                <p
                  className="text-xs truncate"
                  style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}
                >
                  {model}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3">
        <div
          className="p-4 rounded-xl text-center"
          style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)' }}
        >
          <p
            className="text-2xl font-bold mb-1"
            style={{ color: 'var(--color-accent)', fontFamily: 'var(--font-display)' }}
          >
            {result.critic_revision_count}
          </p>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Critic revisions</p>
        </div>
        <div
          className="p-4 rounded-xl text-center"
          style={{ background: 'var(--surface-elevated)', border: '1px solid var(--border)' }}
        >
          <p
            className="text-2xl font-bold mb-1"
            style={{ color: 'var(--color-accent)', fontFamily: 'var(--font-display)' }}
          >
            {result.reasoning_trace.length}
          </p>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Agent steps</p>
        </div>
      </div>

      {/* Run ID */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-secondary)' }}>
          Run ID
        </p>
        <p
          className="text-xs p-2.5 rounded-lg break-all"
          style={{
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-secondary)',
            background: 'var(--surface-elevated)',
            border: '1px solid var(--border)',
          }}
        >
          {result.run_id}
        </p>
      </div>
    </div>
  );
}

const TABS = [
  { id: 'report', label: 'Clinical Report', icon: <FileText size={14} /> },
  { id: 'trace', label: 'Reasoning Trace', icon: <GitMerge size={14} /> },
  { id: 'meta', label: 'System Metadata', icon: <Server size={14} /> },
];

const staggerContainer = {
  animate: { transition: { staggerChildren: 0.08 } },
};
const staggerItem = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
};

export function ResultsPage() {
  const { runId } = useParams<{ runId: string }>();
  const location = useLocation();
  const [result, setResult] = useState<AnalysisResult | null>(
    (location.state as { result?: AnalysisResult })?.result ?? null
  );
  const [loading, setLoading] = useState(!result);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('report');

  useEffect(() => {
    if (result) return;
    if (!runId) {
      setError('No run ID provided.');
      setLoading(false);
      return;
    }
    getResult(runId)
      .then(setResult)
      .catch((err: { response?: { status?: number } }) => {
        if (err.response?.status === 404) {
          setError('Analysis not found. It may still be running or the ID is invalid.');
        } else {
          setError('Failed to load results. Please try again.');
        }
      })
      .finally(() => setLoading(false));
  }, [runId, result]);

  if (loading) {
    return (
      <PageTransition>
        <div className="max-w-7xl mx-auto px-6 md:px-8 py-8 space-y-4">
          <CardSkeleton />
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <div className="lg:col-span-2"><ChartSkeleton /></div>
            <div className="lg:col-span-3"><CardSkeleton /></div>
          </div>
        </div>
      </PageTransition>
    );
  }

  if (error || !result) {
    return (
      <PageTransition>
        <div className="max-w-7xl mx-auto px-6 md:px-8 py-8">
          <div
            className="card p-10 text-center flex flex-col items-center gap-4"
            style={{ borderColor: 'var(--color-danger-border)' }}
          >
            <AlertCircle size={36} style={{ color: 'var(--color-danger)', opacity: 0.6 }} />
            <div>
              <h2 className="text-lg font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
                Result Not Found
              </h2>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                {error ?? 'This result could not be loaded.'}
              </p>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
              style={{
                background: 'var(--surface-elevated)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
              }}
            >
              <RefreshCw size={14} />
              Retry
            </button>
          </div>
        </div>
      </PageTransition>
    );
  }

  const predictions = result.predictions;
  const predData = Object.entries(predictions).map(([name, value]) => ({ name, value }));
  const topConfidencePct = Math.round(result.confidence * 100);

  return (
    <PageTransition>
      <div className="max-w-7xl mx-auto px-6 md:px-8 py-8">
        {/* Routing Banner */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <RoutingBanner result={result} />
        </motion.div>

        <motion.div
          className="grid grid-cols-1 lg:grid-cols-5 gap-5 mb-5"
          variants={staggerContainer}
          initial="initial"
          animate="animate"
        >
          {/* Classification card — 2 cols */}
          <motion.div variants={staggerItem} className="lg:col-span-2">
            <div className="card p-5 h-full">
              <p className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: 'var(--text-secondary)' }}>
                Classification
              </p>

              {/* Hero stat */}
              <div className="flex items-end gap-3 mb-5">
                <span
                  className="text-4xl font-bold tracking-tight capitalize gradient-text"
                  style={{ fontFamily: 'var(--font-display)' }}
                >
                  {result.classification}
                </span>
                <span className="text-2xl font-bold mb-0.5" style={{ color: 'var(--text-secondary)' }}>
                  {topConfidencePct}%
                </span>
              </div>

              {/* Predictions bars */}
              <div className="space-y-2.5">
                {predData.map(({ name, value }) => (
                  <div key={name}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs font-medium capitalize" style={{ color: 'var(--text-primary)' }}>
                        {name}
                      </span>
                      <span
                        className="text-xs font-semibold"
                        style={{ fontFamily: 'var(--font-mono)', color: CLASS_COLORS[name] ?? 'var(--text-secondary)' }}
                      >
                        {(value * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full" style={{ background: 'var(--border)' }}>
                      <motion.div
                        className="h-2 rounded-full"
                        style={{ background: CLASS_COLORS[name] ?? 'var(--color-accent)' }}
                        initial={{ width: 0 }}
                        animate={{ width: `${value * 100}%` }}
                        transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Confidence chart — 3 cols */}
          <motion.div variants={staggerItem} className="lg:col-span-3">
            <div className="card p-5 h-full">
              <p className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: 'var(--text-secondary)' }}>
                Prediction Probabilities
              </p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={predData} barSize={36}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                    tick={{ fontSize: 11, fill: 'var(--text-secondary)' }}
                    axisLine={false}
                    tickLine={false}
                    width={36}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                    {predData.map(entry => (
                      <Cell key={entry.name} fill={CLASS_COLORS[entry.name] ?? '#635BFF'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        </motion.div>

        {/* Grad-CAM + Justification row */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 mb-5">
          {/* Grad-CAM — 2 cols */}
          <motion.div
            className="lg:col-span-2"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="card overflow-hidden h-full">
              <div className="px-5 py-4" style={{ borderBottom: '1px solid var(--border)' }}>
                <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                  Grad-CAM Heatmap
                </p>
              </div>
              <div
                className="flex items-center justify-center p-4"
                style={{ background: '#000', minHeight: 200 }}
              >
                {result.gradcam_image ? (
                  <img
                    src={`data:image/png;base64,${result.gradcam_image}`}
                    alt="Grad-CAM heatmap overlay"
                    className="w-full object-contain rounded-lg max-h-64"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-2 py-8">
                    <ImageIcon size={28} style={{ color: '#444' }} />
                    <p className="text-xs" style={{ color: '#666' }}>Heatmap not available</p>
                  </div>
                )}
              </div>
              <div className="px-5 py-3" style={{ borderTop: '1px solid var(--border)' }}>
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  Highlights regions that most influenced the{' '}
                  <span className="font-medium capitalize" style={{ color: 'var(--text-primary)' }}>
                    {result.classification}
                  </span>{' '}
                  classification.
                </p>
              </div>
            </div>
          </motion.div>

          {/* Orchestrator justification — 3 cols */}
          <motion.div
            className="lg:col-span-3"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
          >
            <div
              className="card h-full overflow-hidden"
              style={{ borderLeft: '3px solid var(--color-accent)' }}
            >
              <div className="px-5 py-4" style={{ borderBottom: '1px solid var(--border)' }}>
                <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                  Orchestrator Justification
                </p>
              </div>
              <div className="p-5">
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  {result.justification}
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Tabbed section */}
        <motion.div
          className="card overflow-hidden"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="p-5" style={{ borderBottom: '1px solid var(--border)' }}>
            <Tabs tabs={TABS} activeTab={activeTab} onChange={setActiveTab} />
          </div>

          <div className="p-5">
            <AnimatePresence mode="wait">
              {activeTab === 'report' && (
                <motion.div
                  key="report"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                  className="markdown-body"
                >
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {result.draft_report}
                  </ReactMarkdown>
                </motion.div>
              )}
              {activeTab === 'trace' && (
                <motion.div
                  key="trace"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                >
                  <ReasoningTrace trace={result.reasoning_trace} />
                </motion.div>
              )}
              {activeTab === 'meta' && (
                <motion.div
                  key="meta"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                >
                  <SystemMetadata result={result} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        <Disclaimer />
      </div>
    </PageTransition>
  );
}
