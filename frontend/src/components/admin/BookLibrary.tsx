import { useState, useEffect } from 'react';
import { FileText, Trash2, Play, CheckCircle, Clock, AlertCircle } from 'lucide-react';
import api from '../../lib/api';

interface Book {
  id: string;
  filename: string;
  philosopher: string;
  status: string;
  chunk_count: number;
  uploaded_at: string;
}

function BookLibrary() {
  const [books, setBooks] = useState<Book[]>([]);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState<string | null>(null);

  useEffect(() => {
    loadBooks();
  }, []);

  const loadBooks = async () => {
    try {
      const response = await api.get('/admin/books');
      setBooks(response.data);
    } catch (error) {
      console.error('Failed to load books:', error);
    } finally {
      setLoading(false);
    }
  };

  const processBook = async (bookId: string) => {
    setProcessing(bookId);
    try {
      await api.post(`/admin/books/${bookId}/process`);
      await loadBooks(); // Refresh list
    } catch (error) {
      console.error('Failed to process book:', error);
    } finally {
      setProcessing(null);
    }
  };

  const deleteBook = async (bookId: string) => {
    if (!confirm('Are you sure you want to delete this book?')) return;

    try {
      await api.delete(`/admin/books/${bookId}`);
      setBooks(books.filter(book => book.id !== bookId));
    } catch (error) {
      console.error('Failed to delete book:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'processed':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'processing':
        return <Clock className="h-5 w-5 text-blue-600 animate-spin" />;
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-600" />;
      default:
        return <FileText className="h-5 w-5 text-gray-600" />;
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-ink-600"></div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="px-6 py-4 bg-ink-900 text-parchment-50">
        <h2 className="text-xl font-cinzel font-bold">Book Library</h2>
        <p className="text-parchment-200 text-sm">Manage uploaded philosophical texts</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-ink-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">
                Book
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">
                Philosopher
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">
                Chunks
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">
                Uploaded
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-ink-200">
            {books.map((book) => (
              <tr key={book.id} className="hover:bg-ink-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <FileText className="h-5 w-5 text-ink-400 mr-3" />
                    <div>
                      <div className="text-sm font-medium text-ink-900">
                        {book.filename}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-ink-700">
                  {book.philosopher}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {getStatusIcon(book.status)}
                    <span className="ml-2 text-sm text-ink-700 capitalize">
                      {book.status}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-ink-700">
                  {book.chunk_count || 0}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-ink-700">
                  {new Date(book.uploaded_at).toLocaleDateString()}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                  <div className="flex space-x-2">
                    {book.status === 'uploaded' && (
                      <button
                        onClick={() => processBook(book.id)}
                        disabled={processing === book.id}
                        className="text-blue-600 hover:text-blue-900 disabled:opacity-50"
                        title="Process book"
                      >
                        {processing === book.id ? (
                          <Clock className="h-5 w-5 animate-spin" />
                        ) : (
                          <Play className="h-5 w-5" />
                        )}
                      </button>
                    )}
                    <button
                      onClick={() => deleteBook(book.id)}
                      className="text-red-600 hover:text-red-900"
                      title="Delete book"
                    >
                      <Trash2 className="h-5 w-5" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {books.length === 0 && (
        <div className="text-center py-12">
          <FileText className="mx-auto h-12 w-12 text-ink-400" />
          <h3 className="mt-2 text-sm font-medium text-ink-900">No books uploaded</h3>
          <p className="mt-1 text-sm text-ink-500">
            Upload your first philosophical text to get started.
          </p>
        </div>
      )}
    </div>
  );
}

export default BookLibrary;