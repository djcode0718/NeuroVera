/**
 * Reusable drag-drop upload component
 */

import { useDropzone } from 'react-dropzone';
import { useCallback } from 'react';

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  isLoading?: boolean;
}

export default function UploadZone({ onFileSelected, isLoading }: UploadZoneProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelected(acceptedFiles[0]);
    }
  }, [onFileSelected]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
    },
    disabled: isLoading,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition
        ${isDragActive 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-300 bg-gray-50'
        }
        ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-400'}
      `}
    >
      <input {...getInputProps()} />
      {isDragActive ? (
        <p className="text-blue-600 font-semibold">Drop your MRI image here...</p>
      ) : (
        <div>
          <p className="text-gray-700 font-semibold mb-2">Drag and drop your MRI scan</p>
          <p className="text-gray-600 text-sm">or click to select a JPEG/PNG file (max 20MB)</p>
        </div>
      )}
    </div>
  );
}
