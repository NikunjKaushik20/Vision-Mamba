import React, { useState, useEffect, useRef } from 'react';

const ChatAssistant = ({ predictionContext }) => {
    const [messages, setMessages] = useState([
        { role: 'assistant', text: "Hello! I'm the AI Radiologist Assistant. I've analyzed the scan below. Do you have any questions about the findings?" }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage = input.trim();
        setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
        setInput('');
        setLoading(true);

        try {
            // If we don't have prediction context, mock the request
            const contextStr = predictionContext ? predictionContext.prediction : "Unknown";
            const conf = predictionContext ? predictionContext.confidence : 0.0;

            const res = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMessage,
                    prediction_context: contextStr,
                    confidence: conf
                })
            });

            if (res.ok) {
                const data = await res.json();
                setMessages(prev => [...prev, { role: 'assistant', text: data.response }]);
            } else {
                throw new Error('Server error');
            }
        } catch (err) {
            // Provide a fallback if API key isn't set or backend is unreachable
            setTimeout(() => {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    text: "I'm currently running in 'Offline Mode' (No OpenAI API key provided or backend unreachable). Based on the model, this appears to be a " +
                        (predictionContext?.prediction || "scan") + ". Please review the Grad-CAM maps to see the focal points."
                }]);
            }, 1000);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', maxHeight: '800px' }}>
            <div style={{ padding: '1rem', borderBottom: '1px solid var(--border-color)', background: 'rgba(56, 189, 248, 0.05)', borderTopLeftRadius: '16px', borderTopRightRadius: '16px' }}>
                <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-blue)' }}>
                    <svg style={{ width: '20px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
                    AI Radiologist Assistant
                </h3>
                {predictionContext && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                        Context: {predictionContext.prediction} ({(predictionContext.confidence * 100).toFixed(1)}%)
                    </div>
                )}
            </div>

            <div style={{ flex: 1, padding: '1rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {messages.map((msg, i) => (
                    <div key={i} style={{
                        alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                        background: msg.role === 'user' ? 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))' : 'rgba(255,255,255,0.05)',
                        color: msg.role === 'user' ? 'white' : 'var(--text-primary)',
                        padding: '0.75rem 1rem',
                        borderRadius: '12px',
                        maxWidth: '85%',
                        fontSize: '0.95rem',
                        lineHeight: '1.4',
                        borderBottomRightRadius: msg.role === 'user' ? '2px' : '12px',
                        borderBottomLeftRadius: msg.role === 'assistant' ? '2px' : '12px',
                    }}>
                        {msg.text}
                    </div>
                ))}
                {loading && (
                    <div style={{ alignSelf: 'flex-start', background: 'rgba(255,255,255,0.05)', padding: '0.75rem 1rem', borderRadius: '12px', color: 'var(--text-secondary)' }}>
                        <span className="typing-dot" style={{ animation: 'blink 1.4s infinite both' }}>.</span>
                        <span className="typing-dot" style={{ animation: 'blink 1.4s infinite both', animationDelay: '0.2s' }}>.</span>
                        <span className="typing-dot" style={{ animation: 'blink 1.4s infinite both', animationDelay: '0.4s' }}>.</span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)' }}>
                <form onSubmit={handleSend} style={{ display: 'flex', gap: '0.5rem' }}>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about the X-ray..."
                        style={{
                            flex: 1,
                            padding: '0.75rem 1rem',
                            borderRadius: '8px',
                            border: '1px solid var(--border-color)',
                            background: 'rgba(0,0,0,0.2)',
                            color: 'white',
                            outline: 'none',
                            fontFamily: 'inherit'
                        }}
                    />
                    <button type="submit" className="btn-primary" disabled={loading || !input.trim()} style={{ padding: '0 1rem' }}>
                        <svg style={{ width: '20px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
                    </button>
                </form>
            </div>
            <style dangerouslySetInnerHTML={{
                __html: `
        @keyframes blink { 0% { opacity: .2; } 20% { opacity: 1; } 100% { opacity: .2; } }
      `}} />
        </div>
    );
};

export default ChatAssistant;
