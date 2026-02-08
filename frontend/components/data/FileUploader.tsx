'use client';

import { useState, useCallback } from 'react';
import { uploadDataFile } from '@/lib/api';

interface FileUploaderProps {
  onUploadSuccess?: (data: any) => void;
}

// Icon Components
const UploadIcon = () => (
  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

const CheckIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

const SpinnerIcon = () => (
  <svg className="w-8 h-8 animate-spin" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const CsvIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const ExcelIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);

const JsonIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
  </svg>
);

const AlertIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  </svg>
);

export default function FileUploader({ onUploadSuccess }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      await handleFileUpload(files[0]);
    }
  }, []);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      await handleFileUpload(files[0]);
    }
  }, []);

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);

    try {
      const result = await uploadDataFile(file);
      setUploadedFile(file.name);

      if (onUploadSuccess) {
        onUploadSuccess(result);
      }

    } catch (error: any) {
      setUploadError(error.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative overflow-hidden
          border-2 border-dashed rounded-2xl p-8 text-center
          transition-all duration-300 ease-out
          ${isDragging
            ? 'border-cyan-500 bg-gradient-to-br from-blue-500/10 to-cyan-500/10 scale-[1.02]'
            : 'border-blue-300/30 bg-gradient-to-br from-slate-900/50 to-blue-900/20 hover:border-cyan-400 hover:shadow-lg hover:shadow-cyan-500/10'
          }
          ${isUploading ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        {/* Background decoration */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-cyan-500/5 pointer-events-none"></div>

        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".csv,.xlsx,.xls,.json"
          onChange={handleFileSelect}
          disabled={isUploading}
        />

        <label htmlFor="file-upload" className="cursor-pointer relative z-10">
          {isUploading ? (
            <div className="py-4">
              <div className="flex justify-center mb-4 text-cyan-500">
                <SpinnerIcon />
              </div>
              <p className="text-lg font-semibold text-cyan-400 mb-3">
                Uploading your file...
              </p>
              <div className="w-full bg-slate-800 rounded-full h-2 max-w-xs mx-auto overflow-hidden">
                <div className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full animate-pulse w-3/4"></div>
              </div>
            </div>
          ) : uploadedFile ? (
            <div className="py-4">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-white shadow-lg shadow-green-500/30">
                <CheckIcon />
              </div>
              <p className="text-lg font-semibold text-green-700 mb-1">
                Successfully uploaded!
              </p>
              <p className="text-sm font-medium text-green-600 mb-3">
                {uploadedFile}
              </p>
              <p className="text-sm text-gray-500">
                Click or drag to upload a different file
              </p>
            </div>
          ) : (
            <div className="py-4">
              <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white shadow-lg shadow-blue-500/30">
                <UploadIcon />
              </div>
              <p className="text-xl font-bold text-white mb-2">
                Drop your data file here
              </p>
              <p className="text-gray-400 mb-4">
                or <span className="text-cyan-400 font-medium hover:underline">browse files</span>
              </p>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full text-xs text-gray-400 border border-white/10">
                <span>Supported:</span>
                <span className="font-medium text-gray-700">CSV, Excel, JSON</span>
              </div>
            </div>
          )}
        </label>
      </div>

      {uploadError && (
        <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-center gap-3">
          <div className="flex-shrink-0 text-red-400">
            <AlertIcon />
          </div>
          <p className="text-red-300 text-sm font-medium">{uploadError}</p>
        </div>
      )}

      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="group text-center p-4 bg-white/5 rounded-xl border border-white/10 shadow-sm hover:shadow-md hover:border-cyan-500/30 transition-all">
          <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-gradient-to-br from-blue-500/80 to-blue-600/80 flex items-center justify-center text-white group-hover:scale-110 transition-transform shadow-lg shadow-blue-500/20">
            <CsvIcon />
          </div>
          <div className="text-sm font-medium text-gray-300">CSV</div>
          <div className="text-xs text-gray-500">Comma-separated</div>
        </div>
        <div className="group text-center p-4 bg-white/5 rounded-xl border border-white/10 shadow-sm hover:shadow-md hover:border-cyan-500/30 transition-all">
          <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-gradient-to-br from-emerald-500/80 to-emerald-600/80 flex items-center justify-center text-white group-hover:scale-110 transition-transform shadow-lg shadow-emerald-500/20">
            <ExcelIcon />
          </div>
          <div className="text-sm font-medium text-gray-300">Excel</div>
          <div className="text-xs text-gray-500">.xlsx, .xls</div>
        </div>
        <div className="group text-center p-4 bg-white/5 rounded-xl border border-white/10 shadow-sm hover:shadow-md hover:border-cyan-500/30 transition-all">
          <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-gradient-to-br from-amber-500/80 to-orange-600/80 flex items-center justify-center text-white group-hover:scale-110 transition-transform shadow-lg shadow-orange-500/20">
            <JsonIcon />
          </div>
          <div className="text-sm font-medium text-gray-300">JSON</div>
          <div className="text-xs text-gray-500">Data objects</div>
        </div>
      </div>
    </div>
  );
}