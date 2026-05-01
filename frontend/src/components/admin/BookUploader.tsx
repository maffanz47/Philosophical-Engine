import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, CheckCircle, XCircle } from 'lucide-react';
import api from '../../lib/api';

function BookUploader() {
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<{success: boolean, message: string} | null>(null);
  const [formData, setFormData] = useState({
    philosopher: '',
    category: ''
  });

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploading(true);
    setUploadResult(null);

    try {
      const uploadData = new FormData();
      uploadData.append('file', file);
      uploadData.append('philosopher', formData.philosopher);
      uploadData.append('category', formData.category);

      const response = await api.post('/admin/books/upload', uploadData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setUploadResult({ success: true, message: `Book "${response.data.filename}" uploaded successfully!` });
    } catch (error: any) {
      setUploadResult({
        success: false,
        message: error.response?.data?.detail || 'Upload failed'
      });
    } finally {
      setUploading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
    disabled: uploading
  });

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-cinzel font-bold text-ink-900 mb-6">Upload Philosophical Text</h2>

        <div className="space-y-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-ink-700 mb-2">Philosopher</label>
            <input
              type="text"
              value={formData.philosopher}
              onChange={(e) => setFormData(prev => ({ ...prev, philosopher: e.target.value }))}
              className="w-full px-3 py-2 border border-ink-300 rounded-md focus:outline-none focus:ring-2 focus:ring-ink-500"
              placeholder="e.g., Aristotle, Plato, Kant"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-ink-700 mb-2">Category</label>
            <select
              value={formData.category}
              onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value }))}
              className="w-full px-3 py-2 border border-ink-300 rounded-md focus:outline-none focus:ring-2 focus:ring-ink-500"
              required
            >
              <option value="">Select category</option>
              <option value="metaphysics">Metaphysics</option>
              <option value="epistemology">Epistemology</option>
              <option value="ethics">Ethics</option>
              <option value="politics">Politics</option>
              <option value="aesthetics">Aesthetics</option>
              <option value="logic">Logic</option>
              <option value="other">Other</option>
            </select>
          </div>
        </div>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-ink-500 bg-ink-50'
              : 'border-ink-300 hover:border-ink-400'
          } ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center">
            {uploading ? (
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-ink-600 mb-4"></div>
            ) : (
              <Upload className="h-12 w-12 text-ink-400 mb-4" />
            )}
            <p className="text-lg font-medium text-ink-700 mb-2">
              {uploading ? 'Uploading...' : 'Drop PDF here or click to browse'}
            </p>
            <p className="text-sm text-ink-500">
              Only PDF files are supported
            </p>
          </div>
        </div>

        {uploadResult && (
          <div className={`mt-6 p-4 rounded-lg flex items-center ${
            uploadResult.success
              ? 'bg-green-50 text-green-800 border border-green-200'
              : 'bg-red-50 text-red-800 border border-red-200'
          }`}>
            {uploadResult.success ? (
              <CheckCircle className="h-5 w-5 mr-3" />
            ) : (
              <XCircle className="h-5 w-5 mr-3" />
            )}
            <span>{uploadResult.message}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default BookUploader;