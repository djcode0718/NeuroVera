/**
 * Upload page - drag-and-drop MRI scan upload with validation
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Disclaimer from '../components/Disclaimer';
import UploadZone from '../components/UploadZone';

export default function UploadPage() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelected = async (file: File) => {
    setError(null);
    setIsLoading(true);

    // File size validation (20MB)
    if (file.size > 20 * 1024 * 1024) {
      setError(`File size exceeds 20MB limit. Size: ${(file.size / (1024 * 1024)).toFixed(1)}MB`);
      setIsLoading(false);
      return;
    }

    // Create FormData for upload
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || `Upload failed: ${response.status}`);
      }

      const data = await response.json();
      const runId = data.run_id;

      // Store run_id and navigate to results page
      sessionStorage.setItem('currentRunId', runId);
      navigate(`/results?run_id=${runId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setIsLoading(false);
    }
  };

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

        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload MRI Scan</h2>

          <UploadZone onFileSelected={handleFileSelected} isLoading={isLoading} />

          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded p-4">
              <p className="text-red-800 font-semibold">Error</p>
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {isLoading && (
            <div className="mt-4 text-center">
              <div className="inline-block">
                <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
              </div>
              <p className="text-gray-600 mt-2">Processing your MRI scan...</p>
            </div>
          )}

          <div className="mt-8 p-4 bg-blue-50 rounded">
            <h3 className="font-semibold text-gray-900 mb-2">Accepted Formats</h3>
            <ul className="text-sm text-gray-700 list-disc list-inside">
              <li>JPEG (.jpg, .jpeg)</li>
              <li>PNG (.png)</li>
              <li>Maximum file size: 20MB</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}
