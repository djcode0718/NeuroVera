/**
 * Disclaimer component - displayed on all pages
 * Emphasizes research-only nature of NeuroTriage
 */

export default function Disclaimer() {
  return (
    <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-4">
      <p className="text-red-700 font-semibold">⚠️ Research Disclaimer</p>
      <p className="text-red-600 text-sm mt-1">
        NeuroTriage is a research proof-of-concept with no clinical validity. 
        This system should never be used for actual medical diagnosis or treatment decisions.
      </p>
    </div>
  );
}
