import { NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Sun, Moon, Brain, Activity } from 'lucide-react';
import { useEffect, useState } from 'react';
import { getHealth } from '../lib/api';
import type { HealthCheck } from '../types/api';

interface NavbarProps {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

export function Navbar({ theme, toggleTheme }: NavbarProps) {
  const [health, setHealth] = useState<HealthCheck | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const h = await getHealth();
        if (!cancelled) {
          setHealth(h);
          setHealthy(h.status === 'ok' || h.status === 'healthy');
        }
      } catch {
        if (!cancelled) setHealthy(false);
      }
    };
    check();
    const interval = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { to: '/', label: 'Dashboard', end: true },
    { to: '/upload', label: 'Upload Scan' },
    { to: '/history', label: 'History' },
  ];

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 h-16"
      style={{
        backgroundColor: 'var(--surface)',
        borderBottom: '1px solid var(--border)',
        backdropFilter: 'blur(12px)',
      }}
    >
      <div className="max-w-7xl mx-auto h-full flex items-center justify-between px-6 md:px-8">
        {/* Logo */}
        <NavLink to="/" className="flex items-center gap-2.5 group select-none">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, #7C6CFF, #4F46E5)',
            }}
          >
            <Brain size={16} className="text-white" />
          </div>
          <span
            className="text-base font-bold tracking-tight"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}
          >
            Neuro<span style={{ color: 'var(--color-accent)' }}>Triage</span>
          </span>
        </NavLink>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {navItems.map(({ to, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                [
                  'relative px-4 py-2 text-sm font-medium rounded-lg transition-colors duration-150 select-none',
                  isActive
                    ? 'text-[var(--color-accent)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-elevated)]',
                ].join(' ')
              }
            >
              {({ isActive }) => (
                <>
                  {label}
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="absolute inset-0 rounded-lg"
                      style={{
                        background: 'linear-gradient(135deg, rgba(124,108,255,0.12), rgba(79,70,229,0.12))',
                        border: '1px solid rgba(99,91,255,0.2)',
                      }}
                      transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
                    />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </div>

        {/* Right side: health indicator + theme toggle */}
        <div className="flex items-center gap-3">
          {/* Health dot */}
          <div className="flex items-center gap-2" title={health ? `Backend: ${health.status}` : 'Checking backend...'}>
            <Activity size={13} style={{ color: 'var(--text-secondary)' }} />
            <div
              className={`w-2 h-2 rounded-full pulse-dot ${
                healthy === null
                  ? 'bg-yellow-400'
                  : healthy
                  ? 'bg-green-500'
                  : 'bg-red-500'
              }`}
            />
          </div>

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="w-9 h-9 flex items-center justify-center rounded-lg transition-colors duration-150"
            style={{
              backgroundColor: 'var(--surface-elevated)',
              border: '1px solid var(--border)',
              color: 'var(--text-secondary)',
            }}
            aria-label="Toggle theme"
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={theme}
                initial={{ opacity: 0, rotate: -30, scale: 0.7 }}
                animate={{ opacity: 1, rotate: 0, scale: 1 }}
                exit={{ opacity: 0, rotate: 30, scale: 0.7 }}
                transition={{ duration: 0.18 }}
              >
                {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
              </motion.div>
            </AnimatePresence>
          </button>
        </div>
      </div>
    </nav>
  );
}
