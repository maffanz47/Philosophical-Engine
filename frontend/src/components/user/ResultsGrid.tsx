import ModuleCard from './ModuleCard';

interface Props {
  results: any;
}

const ResultsGrid = ({ results }: Props) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <ModuleCard module="argument_classifier" result={results.argument_classifier} />
      <ModuleCard module="sentiment_analyzer" result={results.sentiment_analyzer} />
      <ModuleCard module="complexity_scorer" result={results.complexity_scorer} />
      <ModuleCard module="philosopher_finder" result={results.philosopher_finder} />
      <ModuleCard module="fallacy_detector" result={results.fallacy_detector} />
      <ModuleCard module="topic_clusterer" result={results.topic_clusterer} />
      <div className="col-span-3">
        <ModuleCard module="concept_visualizer" result={results.concept_visualizer} />
      </div>
    </div>
  );
};

export default ResultsGrid;