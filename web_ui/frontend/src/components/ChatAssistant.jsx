import React, { useState, useRef, useEffect } from 'react';

const ChatAssistant = ({ analysisState, predictionCtx }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Initial message when analysis completes
  useEffect(() => {
    if (analysisState === 'complete' && messages.length === 0 && predictionCtx) {
      const isFracture = predictionCtx.prediction.toLowerCase().includes('fracture') && !predictionCtx.prediction.toLowerCase().includes('not');
      const conditionSpan = isFracture 
        ? `<span class='text-red-400 font-bold underline'>${predictionCtx.prediction}</span>`
        : `<span class='text-green-400 font-bold'>${predictionCtx.prediction}</span>`;
      
      setMessages([
        {
          id: 1,
          role: 'assistant',
          content: `Analysis complete for the uploaded scan. I've detected a ${conditionSpan} with ${(predictionCtx.confidence * 100).toFixed(1)}% confidence. How can I help you interpret these results?`,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } else if (analysisState === 'idle' || analysisState === 'analyzing') {
      setMessages([]);
    }
  }, [analysisState, predictionCtx, messages.length]);

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || loading) return;

    const messageText = inputValue.trim();
    
    const newUserMessage = {
      id: Date.now(),
      role: 'user',
      content: messageText,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, newUserMessage]);
    setInputValue('');
    setLoading(true);

    try {
        const ctx = predictionCtx?.prediction ?? 'Unknown';
        const conf = predictionCtx?.confidence ?? 0;
        
        const res = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: messageText, prediction_context: ctx, confidence: conf })
        });
        
        if (!res.ok) throw new Error("Backend chat unavailable");
        
        const data = await res.json();
        
        setMessages(prev => [
          ...prev,
          {
            id: Date.now() + 1,
            role: 'assistant',
            content: data.response,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          }
        ]);
    } catch (err) {
        setMessages(prev => [
            ...prev,
            {
              id: Date.now() + 1,
              role: 'assistant',
              content: `<span class="text-red-400">Connection error: Could not reach the AI service. If you are running locally without an OpenAI key, check the backend server logs.</span>`,
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }
        ]);
    } finally {
        setLoading(false);
    }
  };

  const handleActionClick = (actionText) => {
    setInputValue(actionText);
    // Give state time to update before submitting
    setTimeout(() => {
     const formEvent = { preventDefault: () => {} };
     handleSendMessage(formEvent);
    }, 50);
  };

  return (
    <section className="col-span-12 lg:col-span-3 border-l border-slate-200 dark:border-brandBorder flex flex-col h-[calc(100vh-100px)] lg:h-auto bg-slate-50 dark:bg-brandSurface/30 transition-colors">
      <div className="p-4 border-b border-slate-200 dark:border-brandBorder flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse-green"></div>
          <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">AI Assistant</h2>
        </div>
        <button className="text-slate-500 hover:text-white transition-colors">
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"></path>
          </svg>
        </button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 flex flex-col">
        {messages.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-center px-4">
            <p className="text-xs text-slate-500">
              Upload a scan to begin analysis. The Assistant will appear once results are ready.
            </p>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <div key={msg.id} className={`flex flex-col gap-2 max-w-[90%] ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'}`}>
                <div 
                  className={`p-3 text-xs leading-relaxed border rounded-custom ${
                    msg.role === 'user' 
                      ? 'bg-primary/20 border-primary/30 text-slate-900 dark:text-white' 
                      : 'bg-slate-200 dark:bg-brandBorder/50 border-slate-300 dark:border-brandBorder text-slate-700 dark:text-slate-300'
                  }`}
                  dangerouslySetInnerHTML={{ __html: msg.content.replace(/\n/g, '<br/>') }}
                />
                <span className="text-[9px] text-slate-600 px-1">
                  {msg.role === 'assistant' ? 'AI Assistant' : 'You'} • {msg.time}
                </span>
              </div>
            ))}
            
            {loading && (
               <div className="flex gap-1 items-center self-start text-slate-500 ml-2">
                 <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                 <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                 <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
               </div>
            )}
            
            {/* User Options (Interactive style) - Only show if last msg is from AI */}
            {!loading && messages.length > 0 && messages[messages.length - 1].role === 'assistant' && (
              <div className="flex flex-wrap gap-2 mt-2">
                <button 
                  onClick={() => handleActionClick("What are the key findings?")}
                  className="text-[10px] px-3 py-1.5 rounded-full border border-primary/30 bg-primary/5 text-primary hover:bg-primary hover:text-white transition-all"
                >
                  Key Findings
                </button>
                <button 
                  onClick={() => handleActionClick("Generate a radiologist report summary")}
                  className="text-[10px] px-3 py-1.5 rounded-full border border-brandBorder bg-white/5 text-slate-400 hover:text-white transition-all"
                >
                  Generate Report
                </button>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Chat Input */}
      <div className="p-4 bg-white dark:bg-brandSurface border-t border-slate-200 dark:border-brandBorder shrink-0">
        <form onSubmit={handleSendMessage} className="relative">
          <input 
            type="text" 
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={analysisState !== 'complete' || loading}
            placeholder={analysisState === 'complete' ? "Ask assistant about this scan..." : "Waiting for scan..."}
            className="w-full bg-slate-100 dark:bg-brandDark border border-slate-300 dark:border-brandBorder rounded-custom text-xs focus:ring-primary focus:border-primary py-3 pl-4 pr-10 text-slate-800 dark:text-slate-300 placeholder:text-slate-400 dark:placeholder:text-slate-600 disabled:opacity-50"
          />
          <button 
            type="submit"
            disabled={!inputValue.trim() || analysisState !== 'complete' || loading}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-primary hover:text-white transition-colors disabled:opacity-50 disabled:hover:text-primary"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
            </svg>
          </button>
        </form>
        <p className="text-[9px] text-center text-slate-600 mt-2">
          AI-generated content should be verified by a medical professional.
        </p>
      </div>
    </section>
  );
};

export default ChatAssistant;
