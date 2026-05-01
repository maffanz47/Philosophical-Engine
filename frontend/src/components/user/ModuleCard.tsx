interface Props {
  module: string;
  result: any;
}

const ModuleCard = ({ module, result }: Props) => {
  if (!result || result.status === 'not_trained') {
    return (
      <div className="bg-vellum p-4 rounded border border-faded-gold">
        <h3 className="font-cinzel text-aged-gold">{module.replace('_', ' ').toUpperCase()}</h3>
        <p className="text-muted-slate">The oracle has not yet been awakened. Train the model first.</p>
      </div>
    );
  }

  return (
    <div className="bg-vellum p-4 rounded border border-faded-gold">
      <h3 className="font-cinzel text-aged-gold">{module.replace('_', ' ').toUpperCase()}</h3>
      <pre className="text-sm text-ink font-jetbrains">{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
};

export default ModuleCard;