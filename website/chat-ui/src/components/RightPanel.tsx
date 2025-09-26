import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_CONFIG } from '../config';
import '../styles/RightPanel.css';

interface RightPanelProps {
  selectedModel: string;
  onPromptSelect: (prompt: string) => void;
  onInstructionSelect: (instruction: string, type: 'reasoning' | 'translation') => void;
}

const RightPanel: React.FC<RightPanelProps> = ({ selectedModel, onPromptSelect, onInstructionSelect }) => {
  const [prompt, setPrompt] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [instructionType, setInstructionType] = useState<'reasoning' | 'translation'>('reasoning');

  const instructions = {
    reasoning: 'Think step by step, making sure that your thinking is encoded according to the instructions. Then, provide your final answer in \\boxed{} without any encoding.',
    translation: 'Convert the following text, which has been encoded according to the provided scheme, back to English:'
  };

  useEffect(() => {
    if (selectedModel) {
      fetchPrompt();
    }
  }, [selectedModel]);

  useEffect(() => {
    if (prompt) {
      onPromptSelect(prompt);
    }
  }, [prompt]);

  useEffect(() => {
    onInstructionSelect(instructions[instructionType], instructionType);
  }, [instructionType]);

  const fetchPrompt = async () => {
    setLoading(true);
    try {
      const response = await axios.get(
        `${API_CONFIG.BASE_URL}/api/prompts/${selectedModel}`
      );
      // The API now returns a string directly
      let promptText = typeof response.data === 'string' ? response.data : response.data.prompt || 'You are a helpful assistant.';
      // Strip whitespace from both sides
      promptText = promptText.trim();
      setPrompt(promptText);
      onPromptSelect(promptText);
    } catch (error) {
      console.error('Error fetching prompt:', error);
      // Fallback prompt
      const fallbackPrompt = 'You are a helpful assistant.';
      setPrompt(fallbackPrompt);
      onPromptSelect(fallbackPrompt);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="right-panel">
      <div className="panel-header">
        <h2>System Prompt</h2>
      </div>

      <div className="prompt-content">
        <h3>Current System Prompt</h3>
        {loading ? (
          <div className="loading">Loading prompt...</div>
        ) : (
          <div className="prompt-text">
            {prompt || 'No prompt available'}
          </div>
        )}
      </div>

      <div className="instruction-section">
        <h3>Instruction Type</h3>
        <div className="instruction-selector">
          <button
            className={`instruction-btn ${instructionType === 'reasoning' ? 'active' : ''}`}
            onClick={() => setInstructionType('reasoning')}
          >
            Reasoning
          </button>
          <button
            className={`instruction-btn ${instructionType === 'translation' ? 'active' : ''}`}
            onClick={() => setInstructionType('translation')}
          >
            Translation
          </button>
        </div>
        <div className="instruction-text">
          <h4>{instructionType === 'reasoning' ? 'Suffix added to message:' : 'Prefix added to message:'}</h4>
          <div className="prompt-text">
            {instructions[instructionType]}
          </div>
        </div>
      </div>

      <div className="model-info">
        <h3>Model Info</h3>
        <div className="model-details">
          <span className="label">Model ID:</span>
          <span className="value">{selectedModel}</span>
        </div>
      </div>
    </div>
  );
};

export default RightPanel;