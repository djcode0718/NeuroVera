/**
 * Reusable results card component for displaying analysis results
 */

import type { AnalysisResult, AgentTraceEntry } from '../types/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ResultsDisplayProps {
  result: AnalysisResult;
  uploadedImageUrl?: string;
}

export default function ResultsDisplay({ result, uploadedImageUrl }: ResultsDisplayProps) {
  const getRoutingColor = (routing: string) => {
    switch (routing) {
      case 'auto-clear':
        return 'bg-green-100 border-green-400 text-green-800';
      case 'needs-review':
        return 'bg-yellow-100 border-yellow-400 text-yellow-800';
      case 'urgent':
        return 'bg-red-100 border-red-400 text-red-800';
      default:
        return 'bg-gray-100 border-gray-400 text-gray-800';
    }
  };

  const chartData = Object.entries(result.predictions).map(([className, probability]) => ({
    name: className.charAt(0).toUpperCase() + className.slice(1),
    probability: Math.round(probability * 100),
  }));

  return (
    <div className="space-y-6">
      {/* Classification and Confidence Section */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-gray-600 text-sm">Predicted Tumor Class</p>
            <p className="text-4xl font-bold text-gray-900 capitalize">
              {result.classification}
            </p>
          </div>
          <div className="text-right">
            <p className="text-gray-600 text-sm">Confidence</p>
            <p className="text-3xl font-bold text-blue-600">
              {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Routing Badge */}
        <div className={`inline-block border-l-4 px-3 py-2 rounded ${getRoutingColor(result.routing)}`}>
          <p className="font-semibold capitalize">{result.routing}</p>
        </div>
      </div>

      {/* Confidence Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Class Probabilities</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="probability" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Grad-CAM Heatmap */}
      {result.gradcam_image && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Grad-CAM Visualization</h3>
          <div className="flex gap-4">
            {uploadedImageUrl && (
              <div className="flex-1">
                <p className="text-sm text-gray-600 mb-2">Original Image</p>
                <img src={uploadedImageUrl} alt="Uploaded MRI" className="w-full rounded" />
              </div>
            )}
            <div className="flex-1">
              <p className="text-sm text-gray-600 mb-2">Grad-CAM Heatmap</p>
              <img
                src={`data:image/png;base64,${result.gradcam_image}`}
                alt="Grad-CAM Heatmap"
                className="w-full rounded"
              />
            </div>
          </div>
        </div>
      )}

      {/* Draft Report */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Clinical Report</h3>
        <div className="bg-gray-50 p-4 rounded text-gray-700 whitespace-pre-wrap text-sm max-h-80 overflow-y-auto">
          {result.draft_report}
        </div>
      </div>

      {/* Justification */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Routing Justification</h3>
        <p className="text-gray-700">{result.justification}</p>
      </div>

      {/* Reasoning Trace */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Reasoning Trace</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 px-4 font-semibold">Agent</th>
                <th className="text-left py-2 px-4 font-semibold">Summary</th>
                <th className="text-left py-2 px-4 font-semibold">Key Evidence</th>
                <th className="text-left py-2 px-4 font-semibold">Model</th>
              </tr>
            </thead>
            <tbody>
              {result.reasoning_trace.map((entry: AgentTraceEntry, idx) => (
                <tr key={idx} className="border-b hover:bg-gray-50">
                  <td className="py-2 px-4 font-medium capitalize">{entry.agent}</td>
                  <td className="py-2 px-4">{entry.summary}</td>
                  <td className="py-2 px-4 text-xs">{entry.key_evidence}</td>
                  <td className="py-2 px-4 text-xs">
                    {result.models_used[entry.agent] || 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {result.critic_revision_count > 0 && (
          <p className="text-sm text-gray-600 mt-4">
            Critic revisions: {result.critic_revision_count}
          </p>
        )}
      </div>
    </div>
  );
}
