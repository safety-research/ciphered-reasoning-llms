import React from 'react';
import { ChevronDown } from 'lucide-react';
import { Model } from '../types';
import '../styles/ModelSelector.css';

interface ModelSelectorProps {
  models: Model[];
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModel,
  onSelectModel
}) => {
  const [isOpen, setIsOpen] = React.useState(false);
  const selectedModelData = models.find(m => m.id === selectedModel);

  return (
    <div className="model-selector">
      <button
        className="model-selector-button"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div>
          <div className="model-name">{selectedModelData?.name}</div>
          <div className="model-provider">{selectedModelData?.provider}</div>
        </div>
        <ChevronDown size={20} className={`chevron ${isOpen ? 'open' : ''}`} />
      </button>

      {isOpen && (
        <div className="model-dropdown">
          {models.map((model) => (
            <div
              key={model.id}
              className={`model-option ${model.id === selectedModel ? 'selected' : ''}`}
              onClick={() => {
                onSelectModel(model.id);
                setIsOpen(false);
              }}
            >
              <div className="model-name">{model.name}</div>
              <div className="model-provider">{model.provider}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ModelSelector;