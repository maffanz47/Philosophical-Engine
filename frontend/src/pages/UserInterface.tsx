import { useState } from 'react';
import { motion } from 'framer-motion';
import TextInputPanel from '../components/user/TextInputPanel';
import ResultsGrid from '../components/user/ResultsGrid';
import SocraticDialog from '../components/user/SocraticDialog';
import QueryDialog from '../components/user/QueryDialog';

const UserInterface = () => {
  const [results, setResults] = useState<any>(null);
  const [showQuery, setShowQuery] = useState(false);

  return (
    <div className="min-h-screen bg-ink text-parchment">
      <header className="p-4 border-b border-faded-gold">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-cinzel text-aged-gold">PHILOSOPHICAL ENGINE</h1>
          <p className="font-eb-garamond italic">Know thyself through the lens of reason</p>
          <div>
            <button className="mr-4 text-parchment hover:text-aged-gold font-eb-garamond">History</button>
            <button className="text-parchment hover:text-aged-gold font-eb-garamond">Logout</button>
          </div>
        </div>
      </header>
      <main className="p-8">
        <TextInputPanel onAnalyze={setResults} />
        {results && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mt-8">
            <ResultsGrid results={results} />
            <SocraticDialog />
          </motion.div>
        )}
      </main>
      <button
        onClick={() => setShowQuery(true)}
        className="fixed bottom-4 right-4 bg-aged-gold text-ink p-4 rounded-full shadow-lg hover:bg-candlelight font-cinzel"
      >
        AI Tutor
      </button>
      {showQuery && <QueryDialog onClose={() => setShowQuery(false)} />}
    </div>
  );
};

export default UserInterface;