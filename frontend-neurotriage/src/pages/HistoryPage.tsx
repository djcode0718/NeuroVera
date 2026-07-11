import { useEffect, useState, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Clock, Upload, ArrowUpDown, ArrowRight, History } from 'lucide-react';
import { getHistory } from '../lib/api';
import type { AnalysisSummary } from '../types/api';
import { PageTransition } from '../components/PageTransition';
import { RoutingBadge } from '../components/RoutingBanner';
import { Disclaimer } from '../components/Disclaimer';
import { TableRowSkeleton } from '../components/Skeleton';

type SortDir = 'desc' | 'asc';
type RoutingFilter = 'all' | 'urgent' | 'needs-review' | 'auto-clear';

const FILTER_OPTIONS: { id: RoutingFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'urgent', label: 'Urgent' },
  { id: 'needs-review', label: 'Needs Review' },
  { id: 'auto-clear', label: 'Auto-Clear' },
];

function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function EmptyHistoryState() {
  return (
    <tr>
      <td colSpan={5}>
        <div className="flex flex-col items-center py-20 text-center">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
            style={{
              background: 'linear-gradient(135deg, rgba(124,108,255,0.12), rgba(79,70,229,0.12))',
              color: 'var(--color-accent)',
            }}
          >
            <History size={28} />
          </div>
          <h3
            className="text-base font-semibold mb-1.5"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}
          >
            No analyses found
          </h3>
          <p className="text-sm mb-5 max-w-xs" style={{ color: 'var(--text-secondary)' }}>
            Run your first analysis to see results here.
          </p>
          <Link
            to="/upload"
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold text-white"
            style={{ background: 'linear-gradient(135deg, #7C6CFF, #4F46E5)' }}
          >
            <Upload size={14} />
            Upload a scan
          </Link>
        </div>
      </td>
    </tr>
  );
}

export function HistoryPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [filter, setFilter] = useState<RoutingFilter>('all');

  useEffect(() => {
    getHistory()
      .then(setHistory)
      .catch(() => setError('Failed to load history. Is the backend running?'))
      .finally(() => setLoading(false));
  }, []);

  const sorted = useMemo(() => {
    const filtered = filter === 'all' ? history : history.filter(h => h.routing === filter);
    return [...filtered].sort((a, b) => {
      const diff = new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      return sortDir === 'desc' ? diff : -diff;
    });
  }, [history, sortDir, filter]);

  const toggleSort = () => setSortDir(d => (d === 'desc' ? 'asc' : 'desc'));

  return (
    <PageTransition>
      <div className="max-w-7xl mx-auto px-6 md:px-8 py-8">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-start justify-between">
            <div>
              <h1
                className="text-2xl md:text-3xl font-bold tracking-tight mb-1"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}
              >
                Analysis History
              </h1>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                All past MRI analysis runs — most recent first
              </p>
            </div>
            <Link
              to="/upload"
              className="hidden sm:flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold text-white"
              style={{ background: 'linear-gradient(135deg, #7C6CFF, #4F46E5)' }}
            >
              <Upload size={14} />
              New scan
            </Link>
          </div>
        </div>

        {/* Filter pills */}
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          {FILTER_OPTIONS.map(opt => (
            <button
              key={opt.id}
              onClick={() => setFilter(opt.id)}
              className="px-3.5 py-1.5 rounded-full text-xs font-medium transition-all duration-150"
              style={
                filter === opt.id
                  ? {
                      background: 'linear-gradient(135deg, #7C6CFF, #4F46E5)',
                      color: '#fff',
                      boxShadow: '0 2px 8px rgba(99,91,255,0.3)',
                    }
                  : {
                      background: 'var(--surface-elevated)',
                      border: '1px solid var(--border)',
                      color: 'var(--text-secondary)',
                    }
              }
            >
              {opt.label}
            </button>
          ))}
          {!loading && (
            <span className="text-xs ml-1" style={{ color: 'var(--text-secondary)' }}>
              {sorted.length} result{sorted.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {/* Error */}
        {error && (
          <div
            className="rounded-xl px-4 py-3 mb-4 text-sm"
            style={{
              background: 'var(--color-danger-bg)',
              border: '1px solid var(--color-danger-border)',
              color: 'var(--color-danger)',
            }}
          >
            {error}
          </div>
        )}

        {/* Table */}
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ background: 'var(--surface-elevated)', borderBottom: '1px solid var(--border)' }}>
                  <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                    Run ID
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                    Classification
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                    Confidence
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                    Routing
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                    <button
                      onClick={toggleSort}
                      className="flex items-center gap-1.5 hover:opacity-70 transition-opacity"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      <Clock size={12} />
                      Timestamp
                      <ArrowUpDown size={10} />
                    </button>
                  </th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y" style={{ borderColor: 'var(--border)' }}>
                {loading ? (
                  Array.from({ length: 6 }).map((_, i) => <TableRowSkeleton key={i} />)
                ) : sorted.length === 0 ? (
                  <EmptyHistoryState />
                ) : (
                  sorted.map((run, idx) => (
                    <motion.tr
                      key={run.run_id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.04 }}
                      className="group transition-colors"
                      style={{ cursor: 'pointer' }}
                      onClick={() => navigate(`/results/${run.run_id}`)}
                      onMouseEnter={e => (e.currentTarget.style.background = 'var(--surface-elevated)')}
                      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                    >
                      <td className="px-5 py-3.5">
                        <span
                          className="text-xs"
                          style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}
                        >
                          {run.run_id.slice(0, 8)}…
                        </span>
                      </td>
                      <td className="px-4 py-3.5">
                        <span className="font-medium capitalize" style={{ color: 'var(--text-primary)' }}>
                          {run.top_class}
                        </span>
                      </td>
                      <td className="px-4 py-3.5">
                        <span className="font-semibold" style={{ color: 'var(--color-accent)' }}>
                          {Math.round(run.top_confidence * 100)}%
                        </span>
                      </td>
                      <td className="px-4 py-3.5">
                        <RoutingBadge routing={run.routing} />
                      </td>
                      <td className="px-4 py-3.5">
                        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                          {formatDate(run.created_at)}
                        </span>
                      </td>
                      <td className="px-4 py-3.5">
                        <Link
                          to={`/results/${run.run_id}`}
                          onClick={e => e.stopPropagation()}
                          className="flex items-center gap-1 text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity"
                          style={{ color: 'var(--color-accent)' }}
                        >
                          View <ArrowRight size={12} />
                        </Link>
                      </td>
                    </motion.tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <Disclaimer />
      </div>
    </PageTransition>
  );
}
