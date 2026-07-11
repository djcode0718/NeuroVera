/**
 * Reusable history table component
 */

import type { HistoryItem } from '../types/api';

interface HistoryTableProps {
  items: HistoryItem[];
  onRowClick?: (item: HistoryItem) => void;
}

export default function HistoryTable({ items, onRowClick }: HistoryTableProps) {
  const getRoutingColor = (routing: string) => {
    switch (routing) {
      case 'auto-clear':
        return 'text-green-600 bg-green-50';
      case 'needs-review':
        return 'text-yellow-600 bg-yellow-50';
      case 'urgent':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  if (items.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No analysis history yet. Upload an MRI scan to get started.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b">
            <th className="text-left py-3 px-4 font-semibold text-gray-900">Run ID</th>
            <th className="text-left py-3 px-4 font-semibold text-gray-900">Classification</th>
            <th className="text-left py-3 px-4 font-semibold text-gray-900">Confidence</th>
            <th className="text-left py-3 px-4 font-semibold text-gray-900">Routing</th>
            <th className="text-left py-3 px-4 font-semibold text-gray-900">Created At</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, idx) => (
            <tr
              key={idx}
              className="border-b hover:bg-gray-50 cursor-pointer"
              onClick={() => onRowClick?.(item)}
            >
              <td className="py-3 px-4 text-sm font-mono text-gray-600 truncate">
                {item.run_id.substring(0, 8)}...
              </td>
              <td className="py-3 px-4 text-sm capitalize text-gray-900">
                {item.top_class}
              </td>
              <td className="py-3 px-4 text-sm text-gray-900">
                {(item.top_confidence * 100).toFixed(1)}%
              </td>
              <td className={`py-3 px-4 text-sm font-semibold rounded ${getRoutingColor(item.routing)}`}>
                {item.routing}
              </td>
              <td className="py-3 px-4 text-sm text-gray-600">
                {formatDate(item.created_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
