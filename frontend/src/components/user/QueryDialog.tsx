interface Props {
  onClose: () => void;
}

const QueryDialog = ({ onClose }: Props) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div className="bg-deep-brown p-8 rounded w-full max-w-md">
        <h3 className="font-cinzel text-aged-gold">AI PHILOSOPHICAL TUTOR</h3>
        <p className="text-parchment">Ask questions about your analysis results.</p>
        <button onClick={onClose} className="mt-4 bg-faded-red text-parchment px-4 py-2 rounded">Close</button>
      </div>
    </div>
  );
};

export default QueryDialog;