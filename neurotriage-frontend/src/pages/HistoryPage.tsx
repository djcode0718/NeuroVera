/**
 * History page - display past analyses in a searchable table
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Disclaimer from '../components/Disclaimer';
import HistoryTable from '../components/HistoryTable';
import type { HistoryItem } from '../types/api';

export default function HistoryPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch('/api/history');
        if (!response.ok) {
          throw new Error(`Failed to fetch history: ${response.status}`);
        }
        const data = await response.json();
        setHistory(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch history');
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const handleRowClick = (item: HistoryItem) => {
    navigate(`/results?run_id=${item.run_id}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow">
        <div className="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">NeuroTriage</h1>
            <p className="text-gray-600">Analysis History</p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            New Analysis
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <Disclaimer />

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Past Analyses</h2>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded p-4 mb-4">
              <p className="text-red-800 font-semibold">Error</p>
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {isLoading ? (
            <div className="text-center py-8">
              <div className="inline-block">
                <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
              </div>
              <p className="text-gray-600 mt-2">Loading history...</p>
            </div>
          ) : (
            <HistoryTable items={history} onRowClick={handleRowClick} />
          )}
        </div>
      </main>
    </div>
  );
}
