import React, { useState, useRef, useEffect } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import {
  Brain,
  Sparkles,
  Send,
  Loader2,
  AlertTriangle,
  Lightbulb,
  FlaskConical,
  BarChart3,
  GitBranch,
  MessageSquare,
  RefreshCw,
  ExternalLink,
  ShieldAlert,
} from 'lucide-react';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  type?: 'suggestion' | 'explanation' | 'warning' | 'info';
  linkedAction?: string;
}

interface QuickAction {
  id: string;
  label: string;
  icon: React.ReactNode;
  prompt: string;
  category: 'hypothesis' | 'experiment' | 'analysis' | 'next';
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'suggest_hypotheses',
    label: 'Suggest Hypotheses',
    icon: <Lightbulb size={14} />,
    prompt: 'Based on the current investigation, suggest potential hypotheses to test.',
    category: 'hypothesis',
  },
  {
    id: 'explain_pattern',
    label: 'Explain Activation Pattern',
    icon: <Brain size={14} />,
    prompt: 'Explain the activation pattern observed in the current experiment results.',
    category: 'analysis',
  },
  {
    id: 'control_importance',
    label: 'Why Controls Matter',
    icon: <ShieldAlert size={14} />,
    prompt: 'Explain why the chosen control components are important for causal validity.',
    category: 'experiment',
  },
  {
    id: 'compare_experiments',
    label: 'Compare Experiments',
    icon: <GitBranch size={14} />,
    prompt: 'Compare the recent experiment runs and identify key differences in their outcomes.',
    category: 'analysis',
  },
  {
    id: 'next_experiment',
    label: 'What to Test Next',
    icon: <FlaskConical size={14} />,
    prompt: 'Based on current evidence, what should be tested next to advance the investigation?',
    category: 'next',
  },
  {
    id: 'summarize_findings',
    label: 'Summarize Findings',
    icon: <BarChart3 size={14} />,
    prompt: 'Summarize all findings from the current investigation so far.',
    category: 'analysis',
  },
];

const DISCLAIMER = "I cannot fabricate experimental evidence. If an experiment is needed, I will call Agent 1's scientific engine to run it.";

export const AIResearchAssistant: React.FC = () => {
  const {
    activeInvestigation,
    hypotheses,
    runs,
    evidence,
    loading,
  } = useResearchStore();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = (role: 'user' | 'assistant', content: string, type?: ChatMessage['type'], linkedAction?: string) => {
    const newMessage: ChatMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      role,
      content,
      timestamp: Date.now() / 1000,
      type,
      linkedAction,
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  const processUserInput = async (input: string) => {
    addMessage('user', input);
    setIsProcessing(true);

    try {
      // Simulate AI processing (in real implementation, this would call an LLM API)
      await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 1000));

      // Generate contextual response based on input and current state
      const response = generateContextualResponse(input);
      addMessage('assistant', response.content, response.type, response.linkedAction);
    } catch (error) {
      addMessage('assistant', 'I encountered an error processing your request. Please try again.', 'warning');
    } finally {
      setIsProcessing(false);
    }
  };

  const generateContextualResponse = (input: string): { content: string; type?: ChatMessage['type']; linkedAction?: string } => {
    const lowerInput = input.toLowerCase();

    // Hypothesis suggestions
    if (lowerInput.includes('suggest') && lowerInput.includes('hypothesis')) {
      const existingTargets = hypotheses.map((h) => h.target_component);
      const suggestedTargets = ['L9H9', 'L7H9', 'L3H0', 'L8_MLP'].filter((t) => !existingTargets.includes(t));

      return {
        content: `Based on the current investigation, I suggest testing these components:\n\n${suggestedTargets.map((t) => `• **${t}**: Potential role in the circuit pathway`).join('\n')}\n\n${DISCLAIMER}`,
        type: 'suggestion',
        linkedAction: 'create_hypothesis',
      };
    }

    // Activation pattern explanation
    if (lowerInput.includes('explain') && lowerInput.includes('activation')) {
      return {
        content: `Activation patterns show how information flows through the model. Key observations:\n\n• **Logit Lens**: Tracks where target token prediction emerges\n• **Attention Heads**: Route information between token positions\n• **MLP Blocks**: Transform representations at each layer\n\n${DISCLAIMER}`,
        type: 'explanation',
      };
    }

    // Control importance
    if (lowerInput.includes('control') && lowerInput.includes('matter')) {
      return {
        content: `Controls are essential for causal validity:\n\n• **Same-layer control**: Tests if effect is specific to the target head\n• **Random control**: Establishes baseline effect level\n• **Adjacent-layer control**: Tests if effect is layer-specific\n\nWithout controls, we cannot distinguish causal effects from confounding factors.\n\n${DISCLAIMER}`,
        type: 'explanation',
      };
    }

    // Experiment comparison
    if (lowerInput.includes('compare') && lowerInput.includes('experiment')) {
      if (runs.length === 0) {
        return {
          content: 'No experiments have been run yet. Run an experiment to enable comparison.',
          type: 'info',
        };
      }
      return {
        content: `Comparing ${runs.length} experiment run(s):\n\n${runs.slice(0, 3).map((r, i) => `• Run ${i + 1}: ΔL = ${r.delta_logit?.toFixed(3) ?? 'N/A'}`).join('\n')}\n\nKey differences to examine: effect size, control specificity, and cross-prompt stability.\n\n${DISCLAIMER}`,
        type: 'info',
        linkedAction: 'compare_workspace',
      };
    }

    // Next experiment suggestion
    if (lowerInput.includes('next') || lowerInput.includes('test next')) {
      return {
        content: `Recommended next steps:\n\n1. **Test weaker hypothesis**: Try components with weaker evidence\n2. **Cross-prompt validation**: Test current findings on different prompts\n3. **Control refinement**: Add more control components\n\n${DISCLAIMER}`,
        type: 'suggestion',
        linkedAction: 'experiment_builder',
      };
    }

    // Findings summary
    if (lowerInput.includes('summarize') && lowerInput.includes('finding')) {
      const supportedCount = hypotheses.filter((h) => h.status === 'SUPPORTED').length;
      const evidenceCount = evidence.length;

      return {
        content: `**Investigation Summary:**\n\n• Hypotheses tested: ${hypotheses.length}\n• Hypotheses supported: ${supportedCount}\n• Evidence records: ${evidenceCount}\n• Experiment runs: ${runs.length}\n\n${supportedCount > 0 ? 'Some hypotheses have supporting evidence.' : 'No hypotheses have been supported yet.'}\n\n${DISCLAIMER}`,
        type: 'info',
      };
    }

    // Default response
    return {
      content: `I can help with:\n\n• **Suggest hypotheses** - Generate testable hypotheses\n• **Explain patterns** - Interpret activation/attention patterns\n• **Control importance** - Why controls matter for causality\n• **Compare experiments** - Analyze differences between runs\n• **Next steps** - What to test next\n• **Summarize findings** - Overview of investigation progress\n\nPlease ask a specific question or use a quick action button.\n\n${DISCLAIMER}`,
      type: 'info',
    };
  };

  const handleQuickAction = (action: QuickAction) => {
    processUserInput(action.prompt);
  };

  const handleSend = () => {
    if (!inputValue.trim() || isProcessing) return;
    processUserInput(inputValue);
    setInputValue('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            AI Research Assistant
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Research Guidance & Analysis
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Get help with hypotheses, analysis, and experimental design.
          </div>
        </div>
      </div>

      {/* Disclaimer Banner */}
      <div style={{
        padding: '10px 14px',
        borderRadius: 8,
        backgroundColor: colors.warningSoft,
        border: `1px solid ${colors.warningBorder}`,
        color: colors.warningText,
        fontSize: 12,
        marginBottom: 16,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}>
        <ShieldAlert size={16} />
        <span><b>Important:</b> {DISCLAIMER}</span>
      </div>

      {/* Quick Actions */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.id}
            onClick={() => handleQuickAction(action)}
            disabled={isProcessing}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '8px 14px',
              borderRadius: 6,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.bodyMuted,
              fontSize: 12,
              fontWeight: 600,
              cursor: isProcessing ? 'default' : 'pointer',
              opacity: isProcessing ? 0.5 : 1,
              transition: 'all 0.15s ease',
            }}
          >
            {action.icon}
            {action.label}
          </button>
        ))}
      </div>

      {/* Chat Messages */}
      <div style={{
        border: `1px solid ${colors.border}`,
        borderRadius: 10,
        backgroundColor: colors.surfaceTile1,
        display: 'flex',
        flexDirection: 'column',
        height: 'calc(100vh - 300px)',
        minHeight: 400,
      }}>
        <div style={{ flex: 1, overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
          {messages.length === 0 && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: colors.bodyMuted,
              textAlign: 'center',
            }}>
              <Brain size={48} style={{ opacity: 0.2, marginBottom: 12 }} />
              <div style={{ fontSize: 14, fontWeight: 600, color: colors.ink }}>AI Research Assistant</div>
              <div style={{ fontSize: 12, marginTop: 4, maxWidth: 400 }}>
                Ask questions about your research, get hypothesis suggestions, or request analysis of your experiments.
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              }}
            >
              <div
                style={{
                  maxWidth: '80%',
                  padding: '10px 14px',
                  borderRadius: 10,
                  backgroundColor: msg.role === 'user' ? colors.primary : colors.surfacePearl,
                  color: msg.role === 'user' ? colors.onPrimary : colors.ink,
                  fontSize: 12,
                  lineHeight: 1.5,
                  whiteSpace: 'pre-wrap',
                }}
              >
                {msg.role === 'assistant' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <Sparkles size={12} style={{ color: colors.primary }} />
                    <span style={{ fontSize: 10, fontWeight: 700, color: colors.primary }}>AI ASSISTANT</span>
                  </div>
                )}
                {msg.content}
                {msg.linkedAction && (
                  <div style={{ marginTop: 8 }}>
                    <button
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 4,
                        padding: '4px 8px',
                        borderRadius: 4,
                        border: `1px solid ${colors.primary}`,
                        backgroundColor: 'transparent',
                        color: colors.primary,
                        fontSize: 10,
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      <ExternalLink size={10} /> Open {msg.linkedAction.replace(/_/g, ' ')}
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isProcessing && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: colors.bodyMuted }}>
              <Loader2 size={14} className="animate-spin" />
              <span style={{ fontSize: 12 }}>Thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div style={{
          padding: 12,
          borderTop: `1px solid ${colors.border}`,
          display: 'flex',
          gap: 8,
        }}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your research..."
            disabled={isProcessing}
            style={{
              flex: 1,
              padding: '10px 14px',
              borderRadius: 6,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.ink,
              fontSize: 12,
              outline: 'none',
            }}
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || isProcessing}
            style={{
              padding: '10px 16px',
              borderRadius: 6,
              border: 'none',
              backgroundColor: colors.primary,
              color: colors.onPrimary,
              fontSize: 12,
              fontWeight: 600,
              cursor: !inputValue.trim() || isProcessing ? 'default' : 'pointer',
              opacity: !inputValue.trim() || isProcessing ? 0.5 : 1,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <Send size={14} /> Send
          </button>
        </div>
      </div>
    </div>
  );
};
