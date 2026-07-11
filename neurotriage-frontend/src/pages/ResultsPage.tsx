/**
 * Results page - display analysis results with Grad-CAM visualization
 */

import { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import Disclaimer from '../components/Disclaimer';
import ResultsDisplay from '../components/ResultsDisplay';
import type { AnalysisResult } from '../types/api';

export default function ResultsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);

  useEffect(() => {
    const runId = searchParams.get('run_id') || sessionStorage.getItem('currentRunId');
    const uploadedImage = sessionStorage.getItem('uploadedImage');

    if (uploadedImage) {
      setUploadedImageUrl(uploadedImage);
    }

    if (!runId) {
      setError('No run ID provided');
      setIsLoading(false);
      return;
    }

    // Fetch results
    const fetchResults = async () => {
      try {
        const response = await fetch(`/api/results/${runId}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch results: ${response.status}`);
        }
        const data = await response.json();
        setResult(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch results');
      } finally {
        setIsLoading(false);
      }
    };

    // Poll until results are available (max 60 attempts, 1s each)
    let attempts = 0;
    const maxAttempts = 60;
    const pollInterval = setInterval(async () => {
      attempts++;
      await fetchResults();
      if (attempts >= maxAttempts) {
        clearInterval(pollInterval);
      }
    }, 1000);

    return () => clearInterval(pollInterval);
  }, [searchParams]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <header className="bg-white shadow">
          <div className="max-w-6xl mx-auto px-4 py-4">
            <h1 className="text-3xl font-bold text-gray-900">NeuroTriage</h1>
            <p className="text-gray-600">Brain MRI Analysis System</p>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-8">
          <Disclaimer />
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <div className="inline-block">
              <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
            </div>
            <p className="text-gray-600 mt-4">Analyzing MRI scan...</p>
          </div>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <header className="bg-white shadow">
          <div className="max-w-6xl mx-auto px-4 py-4">
            <h1 className="text-3xl font-bold text-gray-900">NeuroTriage</h1>
            <p className="text-gray-600">Brain MRI Analysis System</p>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-8">
          <Disclaimer />
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <p className="text-red-800 font-semibold">Error</p>
            <p className="text-red-700">{error}</p>
            <button
              onClick={() => navigate('/')}
              className="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
            >
              Back to Upload
            </button>
          </div>
        </main>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <header className="bg-white shadow">
          <div className="max-w-6xl mx-auto px-4 py-4">
            <h1 className="text-3xl font-bold text-gray-900">NeuroTriage</h1>
            <p className="text-gray-600">Brain MRI Analysis System</p>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-8">
          <Disclaimer />
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
            <p className="text-yellow-800 font-semibold">Still Processing</p>
            <p className="text-yellow-700">Your analysis is taking longer than expected.</p>
            <button
              onClick={() => navigate('/')}
              className="mt-4 bg-yellow-600 text-white px-4 py-2 rounded hover:bg-yellow-700"
            >
              Back to Upload
            </button>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow">
        <div className="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">NeuroTriage</h1>
            <p className="text-gray-600">Brain MRI Analysis Results</p>
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
        <ResultsDisplay result={result} uploadedImageUrl={uploadedImageUrl || undefined} />
      </main>
    </div>
  );
}
