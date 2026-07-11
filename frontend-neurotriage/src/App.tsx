import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { Navbar } from './components/Navbar';
import { DashboardPage } from './pages/DashboardPage';
import { UploadPage } from './pages/UploadPage';
import { ResultsPage } from './pages/ResultsPage';
import { HistoryPage } from './pages/HistoryPage';
import { useTheme } from './hooks/useTheme';

function AnimatedRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait" initial={false}>
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/results/:runId" element={<ResultsPage />} />
        <Route path="/history" element={<HistoryPage />} />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  const [theme, toggleTheme] = useTheme();

  return (
    <BrowserRouter>
      <div className="min-h-screen" style={{ backgroundColor: 'var(--bg)' }}>
        <Navbar theme={theme} toggleTheme={toggleTheme} />
        {/* Offset for fixed nav */}
        <div className="pt-16">
          <AnimatedRoutes />
        </div>
      </div>
    </BrowserRouter>
  );
}
