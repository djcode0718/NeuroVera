import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, ImageIcon, X, CheckCircle, AlertCircle, Loader2,
  Eye, Database, FileText, ShieldCheck, Zap,
} from 'lucide-react';
import { analyzeMri } from '../lib/api';
import type { AnalysisResult } from '../types/api';
import { PageTransition } from '../components/PageTransition';
import { Disclaimer } from '../components/Disclaimer';

const PIPELINE_STEPS = [
  { label: 'Vision Analysis', description: 'Running VGG16 classifier + Grad-CAM heatmap', icon: Eye },
  { label: 'Retrieval', description: 'Searching case bank for similar scans', icon: Database },
  { label: 'Report Drafting', description: 'Synthesizing structured clinical report via LLM', icon: FileText },
  { label: 'Critic Verification', description: 'Independent cross-check of report quality', icon: ShieldCheck },
  { label: 'Triage Routing', description: 'Deterministic urgency classification', icon: Zap },
];

type UploadState = 'idle' | 'uploading' | 'error';

function useProgressSimulation() {
  const [currentStep, setCurrentStep] = useState(-1);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = useCallback(() => {
    setCurrentStep(0);
    const delays = [2000, 4000, 7000, 11000]; // cumulative ms to mark each step done
    delays.forEach((delay, idx) => {
      setTimeout(() => setCurrentStep(idx + 1), delay);
    });
  }, []);

  const reset = useCallback(() => {
    setCurrentStep(-1);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  return { currentStep, start, reset };
}

interface StepIndicatorProps {
  step: typeof PIPELINE_STEPS[0];
  index: number;
  currentStep: number;
  total: number;
}

function StepIndicator({ step, index, currentStep, total }: StepIndicatorProps) {
  const Icon = step.icon;
  const isDone = currentStep > index;
  const isActive = currentStep === index;

  return (
    <div className="flex items-start gap-3 relative">
      {/* Connector line */}
      {index < total - 1 && (
        <div
          className="absolute left-5 top-10 w-0.5 h-6 transition-colors duration-500"
          style={{
            background: isDone ? 'var(--color-accent)' : 'var(--border)',
          }}
        />
      )}

      {/* Circle */}
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 transition-all duration-300"
        style={{
          background: isDone
            ? 'linear-gradient(135deg, #7C6CFF, #4F46E5)'
            : isActive
            ? 'linear-gradient(135deg, rgba(124,108,255,0.2), rgba(79,70,229,0.2))'
            : 'var(--surface-elevated)',
          border: `1px solid ${isDone ? 'transparent' : isActive ? 'rgba(99,91,255,0.5)' : 'var(--border)'}`,
          boxShadow: isActive ? '0 0 0 3px rgba(99,91,255,0.15)' : 'none',
        }}
      >
        {isDone ? (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', bounce: 0.4 }}
          >
            <CheckCircle size={18} className="text-white" />
          </motion.div>
        ) : isActive ? (
          <Loader2 size={18} className="animate-spin" style={{ color: 'var(--color-accent)' }} />
        ) : (
          <Icon size={16} style={{ color: 'var(--text-secondary)', opacity: 0.5 }} />
        )}
      </div>

      {/* Text */}
      <div className="pt-1.5 pb-6">
        <p
          className="text-sm font-semibold transition-colors duration-200"
          style={{ color: isDone || isActive ? 'var(--text-primary)' : 'var(--text-secondary)' }}
        >
          {step.label}
        </p>
        {(isDone || isActive) && (
          <motion.p
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-xs mt-0.5"
            style={{ color: 'var(--text-secondary)' }}
          >
            {step.description}
            {isDone && ' ✓'}
          </motion.p>
        )}
      </div>
    </div>
  );
}

export function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [errorMsg, setErrorMsg] = useState<string>('');
  const inputRef = useRef<HTMLInputElement>(null);
  const { currentStep, start: startProgress, reset: resetProgress } = useProgressSimulation();

  const MAX_SIZE_BYTES = 20 * 1024 * 1024; // 20MB

  function validateFile(f: File): string | null {
    if (!['image/jpeg', 'image/png'].includes(f.type)) {
      return 'Only JPEG or PNG files are accepted.';
    }
    if (f.size > MAX_SIZE_BYTES) {
      return 'File exceeds the 20MB size limit.';
    }
    return null;
  }

  function handleFile(f: File) {
    const err = validateFile(f);
    if (err) {
      setErrorMsg(err);
      setUploadState('error');
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setUploadState('idle');
    setErrorMsg('');
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragActive(true); };
  const onDragLeave = () => setDragActive(false);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setUploadState('idle');
    setErrorMsg('');
    resetProgress();
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleSubmit = async () => {
    if (!file) return;
    setUploadState('uploading');
    setErrorMsg('');
    startProgress();

    try {
      const result: AnalysisResult = await analyzeMri(file);
      navigate(`/results/${result.run_id}`, { state: { result } });
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string }; status?: number }; code?: string };
      let msg = 'Analysis failed. Please try again.';
      if (axiosErr.code === 'ECONNABORTED') {
        msg = 'Request timed out. The pipeline may be under load — please retry.';
      } else if (axiosErr.response?.status === 422) {
        msg = axiosErr.response.data?.detail ?? 'Invalid file or file too large.';
      } else if (axiosErr.response?.status === 500) {
        msg = axiosErr.response.data?.detail ?? 'Pipeline error on the backend.';
      } else if (!axiosErr.response) {
        msg = 'Cannot reach the backend. Is the server running on port 8000?';
      }
      setErrorMsg(msg);
      setUploadState('error');
      resetProgress();
    }
  };

  const isUploading = uploadState === 'uploading';

  return (
    <PageTransition>
      <div className="max-w-7xl mx-auto px-6 md:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1
            className="text-2xl md:text-3xl font-bold tracking-tight mb-1"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}
          >
            Upload MRI Scan
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            JPEG or PNG, max 20MB — runs the full 5-agent pipeline
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left: dropzone + file info — 3 cols */}
          <div className="lg:col-span-3 space-y-4">
            {/* Dropzone */}
            <motion.div
              onClick={() => !isUploading && !file && inputRef.current?.click()}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              animate={{
                scale: dragActive ? 1.01 : 1,
                borderColor: dragActive
                  ? 'var(--color-accent)'
                  : uploadState === 'error'
                  ? 'var(--color-danger)'
                  : 'var(--border)',
              }}
              transition={{ duration: 0.15 }}
              className="rounded-2xl border-2 border-dashed cursor-pointer transition-colors overflow-hidden"
              style={{
                backgroundColor: dragActive ? 'rgba(99,91,255,0.04)' : 'var(--surface)',
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png"
                className="hidden"
                onChange={onInputChange}
                disabled={isUploading}
                id="mri-file-input"
              />

              <AnimatePresence mode="wait">
                {preview ? (
                  <motion.div
                    key="preview"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="relative"
                  >
                    <img
                      src={preview}
                      alt="MRI preview"
                      className="w-full object-contain max-h-72"
                      style={{ background: '#000' }}
                    />
                    {!isUploading && (
                      <button
                        onClick={(e) => { e.stopPropagation(); clearFile(); }}
                        className="absolute top-3 right-3 w-8 h-8 rounded-lg flex items-center justify-center"
                        style={{ background: 'rgba(0,0,0,0.6)', color: '#fff' }}
                        aria-label="Remove file"
                      >
                        <X size={14} />
                      </button>
                    )}
                    {file && (
                      <div className="px-5 py-3 flex items-center gap-3" style={{ borderTop: '1px solid var(--border)' }}>
                        <ImageIcon size={15} style={{ color: 'var(--color-accent)' }} />
                        <div className="min-w-0">
                          <p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                            {file.name}
                          </p>
                          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                            {(file.size / 1024 / 1024).toFixed(2)} MB · {file.type.split('/')[1].toUpperCase()}
                          </p>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center py-16 px-8 text-center"
                  >
                    <div
                      className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
                      style={{
                        background: dragActive
                          ? 'linear-gradient(135deg, rgba(124,108,255,0.2), rgba(79,70,229,0.2))'
                          : 'var(--surface-elevated)',
                        border: '1px solid var(--border)',
                        color: dragActive ? 'var(--color-accent)' : 'var(--text-secondary)',
                      }}
                    >
                      <Upload size={24} />
                    </div>
                    <p
                      className="text-base font-semibold mb-1"
                      style={{ color: dragActive ? 'var(--color-accent)' : 'var(--text-primary)' }}
                    >
                      {dragActive ? 'Drop to upload' : 'Drag & drop your scan'}
                    </p>
                    <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      or{' '}
                      <span
                        className="font-medium underline underline-offset-2 cursor-pointer"
                        style={{ color: 'var(--color-accent)' }}
                      >
                        browse files
                      </span>
                    </p>
                    <p className="text-xs mt-3" style={{ color: 'var(--text-secondary)' }}>
                      JPEG or PNG · max 20 MB
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Error */}
            <AnimatePresence>
              {uploadState === 'error' && errorMsg && (
                <motion.div
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="flex items-start gap-3 rounded-xl px-4 py-3"
                  style={{
                    background: 'var(--color-danger-bg)',
                    border: '1px solid var(--color-danger-border)',
                  }}
                >
                  <AlertCircle size={15} className="shrink-0 mt-0.5" style={{ color: 'var(--color-danger)' }} />
                  <p className="text-sm" style={{ color: 'var(--color-danger)' }}>{errorMsg}</p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Submit button */}
            <button
              onClick={handleSubmit}
              disabled={!file || isUploading}
              id="analyze-button"
              className="w-full py-3 rounded-xl text-sm font-semibold text-white transition-all duration-150 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              style={{
                background: 'linear-gradient(135deg, #7C6CFF, #4F46E5)',
                boxShadow: file && !isUploading ? '0 4px 14px rgba(99,91,255,0.35)' : 'none',
              }}
            >
              {isUploading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Running pipeline…
                </>
              ) : (
                <>
                  <Zap size={16} />
                  Run Analysis
                </>
              )}
            </button>
          </div>

          {/* Right: pipeline progress — 2 cols */}
          <div className="lg:col-span-2">
            <div className="card p-5 sticky top-24">
              <div className="flex items-center gap-2 mb-5">
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center"
                  style={{ background: 'linear-gradient(135deg, rgba(124,108,255,0.15), rgba(79,70,229,0.15))' }}
                >
                  <Zap size={14} style={{ color: 'var(--color-accent)' }} />
                </div>
                <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                  Pipeline Progress
                </h2>
              </div>

              <div>
                {PIPELINE_STEPS.map((step, index) => (
                  <StepIndicator
                    key={step.label}
                    step={step}
                    index={index}
                    currentStep={isUploading || currentStep >= 0 ? currentStep : -1}
                    total={PIPELINE_STEPS.length}
                  />
                ))}
              </div>

              {!isUploading && currentStep === -1 && (
                <p className="text-xs text-center mt-2" style={{ color: 'var(--text-secondary)' }}>
                  Steps will animate as the pipeline runs
                </p>
              )}
            </div>
          </div>
        </div>

        <Disclaimer />
      </div>
    </PageTransition>
  );
}
