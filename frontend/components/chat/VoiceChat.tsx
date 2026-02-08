'use client';

import { useState, useRef, useEffect } from 'react';
import { sendChatMessage, getVisualizationSuggestions, clearChatHistory, VisualizationSuggestion, transcribeAudio } from '@/lib/api';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    code?: string | null;
    output?: string | null;
    visualization?: string | null;
    error?: boolean;
    timestamp: Date;
}

// Icon Components
const MicIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
    </svg>
);

const StopIcon = () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
);

const SendIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
    </svg>
);

const SparklesIcon = () => (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
        <path d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
    </svg>
);

const CopyIcon = () => (
    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
    </svg>
);

const UserIcon = () => (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
    </svg>
);

const BotIcon = () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H6a2 2 0 01-2-2v-1H3a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2zM9 16a1 1 0 100-2 1 1 0 000 2zm6 0a1 1 0 100-2 1 1 0 000 2z" />
    </svg>
);

const SpinnerIcon = () => (
    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);

// Python syntax highlighter
const highlightPython = (code: string): React.ReactNode[] => {
    const tokens: { type: string; value: string }[] = [];
    let remaining = code;

    const patterns: [string, RegExp][] = [
        ['comment', /^#.*/],
        ['string', /^('''[\s\S]*?'''|"""[\s\S]*?"""|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")/],
        ['decorator', /^@\w+/],
        ['keyword', /^(def|class|if|elif|else|for|while|try|except|finally|with|return|yield|import|from|as|in|is|not|and|or|True|False|None|lambda|pass|break|continue|raise|assert|global|nonlocal|async|await)\b/],
        ['builtin', /^(print|len|range|str|int|float|list|dict|set|tuple|bool|type|isinstance|hasattr|getattr|setattr|open|input|sum|min|max|abs|round|sorted|reversed|enumerate|zip|map|filter|any|all|format)\b/],
        ['function', /^([a-zA-Z_]\w*)\s*(?=\()/],
        ['number', /^(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)/],
        ['operator', /^(==|!=|<=|>=|<|>|\+|-|\*\*|\*|\/\/|\/|%|=|\+=|-=|\*=|\/=|&|\||\^|~|<<|>>)/],
        ['punctuation', /^[()[\]{},.:;]/],
        ['variable', /^[a-zA-Z_]\w*/],
        ['whitespace', /^\s+/],
    ];

    while (remaining.length > 0) {
        let matched = false;
        for (const [type, pattern] of patterns) {
            const match = remaining.match(pattern);
            if (match) {
                tokens.push({ type, value: match[0] });
                remaining = remaining.slice(match[0].length);
                matched = true;
                break;
            }
        }
        if (!matched) {
            tokens.push({ type: 'text', value: remaining[0] });
            remaining = remaining.slice(1);
        }
    }

    const colors: Record<string, string> = {
        comment: 'text-gray-500 italic',
        string: 'text-amber-300',
        decorator: 'text-yellow-400',
        keyword: 'text-cyan-400 font-medium', // Updated from purple
        builtin: 'text-blue-400',
        function: 'text-sky-300',
        number: 'text-orange-400',
        operator: 'text-cyan-300', // Updated from pink
        punctuation: 'text-gray-400',
        variable: 'text-slate-200',
        whitespace: '',
        text: 'text-slate-200',
    };

    return tokens.map((token, i) => (
        <span key={i} className={colors[token.type] || 'text-slate-200'}>
            {token.value}
        </span>
    ));
};

// Collapsible code block component
const CollapsibleCode = ({ lang, code, isPython }: { lang: string; code: string; isPython: boolean }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className="my-3">
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 text-xs text-cyan-400 hover:text-white transition-colors py-2 px-3 bg-cyan-900/20 hover:bg-cyan-900/40 rounded-lg border border-cyan-500/10 group"
            >
                <svg
                    className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                    fill="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z" />
                </svg>
                <svg className="w-3.5 h-3.5 text-cyan-500/70 group-hover:text-cyan-400" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z" />
                </svg>
                {isExpanded ? 'Hide Code' : 'Show Code'}
                <span className="text-cyan-500/50 ml-1 font-mono">({lang || 'Python'})</span>
            </button>

            {isExpanded && (
                <div className="mt-2 animate-fadeIn">
                    <div className="flex items-center justify-between text-xs mb-1 px-2">
                        <span className="uppercase font-medium text-cyan-400/80 flex items-center gap-1.5 font-mono text-[10px] tracking-wider">
                            GENERATED CODE
                        </span>
                        <button
                            onClick={() => navigator.clipboard.writeText(code)}
                            className="hover:text-white text-cyan-500/70 transition-colors flex items-center gap-1 text-xs"
                        >
                            <CopyIcon /> Copy
                        </button>
                    </div>
                    <div className="bg-slate-950/80 rounded-lg p-4 overflow-x-auto border border-blue-500/10 shadow-inner">
                        <pre className="text-xs font-mono leading-relaxed">
                            {isPython ? highlightPython(code) : <span className="text-gray-300">{code}</span>}
                        </pre>
                    </div>
                </div>
            )}
        </div>
    );
};

// Markdown renderer for chat messages
const renderMarkdown = (text: string) => {
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let key = 0;

    while (remaining.length > 0) {
        const codeBlockMatch = remaining.match(/^```(\w*)\n?([\s\S]*?)```/);
        if (codeBlockMatch) {
            const lang = codeBlockMatch[1] || 'python';
            const code = codeBlockMatch[2].trim();
            const isPython = lang.toLowerCase() === 'python' || lang === '' || lang.toLowerCase() === 'py';

            parts.push(
                <CollapsibleCode key={key++} lang={lang} code={code} isPython={isPython} />
            );
            remaining = remaining.slice(codeBlockMatch[0].length);
            continue;
        }

        const inlineCodeMatch = remaining.match(/^`([^`]+)`/);
        if (inlineCodeMatch) {
            parts.push(
                <code key={key++} className="bg-blue-900/30 px-1.5 py-0.5 rounded text-cyan-300 text-xs font-mono border border-blue-500/20">
                    {inlineCodeMatch[1]}
                </code>
            );
            remaining = remaining.slice(inlineCodeMatch[0].length);
            continue;
        }

        const boldMatch = remaining.match(/^\*\*([^*]+)\*\*/) || remaining.match(/^__([^_]+)__/);
        if (boldMatch) {
            parts.push(<strong key={key++} className="font-semibold text-white">{boldMatch[1]}</strong>);
            remaining = remaining.slice(boldMatch[0].length);
            continue;
        }

        const nextSpecial = remaining.search(/```|`|\*\*|__/);
        if (nextSpecial === -1) {
            parts.push(<span key={key++}>{remaining}</span>);
            break;
        } else if (nextSpecial === 0) {
            parts.push(<span key={key++}>{remaining[0]}</span>);
            remaining = remaining.slice(1);
        } else {
            parts.push(<span key={key++}>{remaining.slice(0, nextSpecial)}</span>);
            remaining = remaining.slice(nextSpecial);
        }
    }

    return <>{parts}</>;
};

export default function VoiceChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [suggestions, setSuggestions] = useState<VisualizationSuggestion[]>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Voice recording state
    const [isRecording, setIsRecording] = useState(false);
    const [isPreparing, setIsPreparing] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [voiceError, setVoiceError] = useState<string | null>(null);

    // Recording refs
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        loadSuggestions();
    }, []);

    const loadSuggestions = async () => {
        try {
            const result = await getVisualizationSuggestions();
            setSuggestions(result.suggestions);
        } catch (error) {
            console.error('Failed to load suggestions:', error);
        }
    };

    const handleSend = async (text?: string) => {
        const messageText = text || input.trim();
        if (!messageText || isLoading) return;

        setInput('');

        setMessages(prev => [...prev, {
            role: 'user',
            content: messageText,
            timestamp: new Date()
        }]);
        setIsLoading(true);

        try {
            const response = await sendChatMessage(messageText);

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.text,
                code: response.code,
                output: response.output,
                visualization: response.visualization,
                error: response.error,
                timestamp: new Date()
            }]);
        } catch (error: any) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${error.message || 'Unable to process your request'}`,
                error: true,
                timestamp: new Date()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const startRecording = async () => {
        try {
            setIsPreparing(true);
            setVoiceError(null);
            audioChunksRef.current = [];

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            });

            streamRef.current = stream;

            let mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'audio/webm';
            if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'audio/mp4';

            const mediaRecorder = new MediaRecorder(stream, { mimeType });

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onerror = () => {
                setVoiceError('Recording error');
                setIsRecording(false);
            };

            mediaRecorder.start(200);
            mediaRecorderRef.current = mediaRecorder;

            setIsRecording(true);
            setIsPreparing(false);

        } catch (err: any) {
            setVoiceError('Microphone access denied or not found');
            setIsPreparing(false);
        }
    };

    const stopRecording = async () => {
        const mediaRecorder = mediaRecorderRef.current;
        if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

        return new Promise<void>((resolve) => {
            mediaRecorder.onstop = async () => {
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop());
                    streamRef.current = null;
                }

                setIsRecording(false);
                mediaRecorderRef.current = null;

                if (audioChunksRef.current.length === 0) {
                    setVoiceError('No audio recorded');
                    resolve();
                    return;
                }

                const audioBlob = new Blob(audioChunksRef.current, {
                    type: mediaRecorder.mimeType || 'audio/webm'
                });

                if (audioBlob.size < 1000) {
                    setVoiceError('Recording too short');
                    resolve();
                    return;
                }

                setIsTranscribing(true);
                try {
                    const result = await transcribeAudio(audioBlob);
                    if (result.success && result.text) {
                        await handleSend(result.text);
                    } else {
                        setVoiceError('Could not transcribe audio');
                    }
                } catch (err: any) {
                    setVoiceError('Transcription failed');
                } finally {
                    setIsTranscribing(false);
                }
                resolve();
            };
            mediaRecorder.stop();
        });
    };

    const handleVoiceClick = async () => {
        setVoiceError(null);
        if (isRecording) {
            await stopRecording();
        } else {
            await startRecording();
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleClear = async () => {
        try {
            await clearChatHistory();
            setMessages([]);
        } catch (error) {
            console.error('Failed to clear');
        }
    };

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    const quickQuestions = [
        "Show data summary",
        "Which columns have missing values?",
        "What is the average of numeric columns?",
        "Create a correlation heatmap"
    ];

    return (
        <div className="h-[750px] flex flex-col bg-slate-950 rounded-2xl shadow-2xl overflow-hidden border border-blue-500/10 relative">
            {/* Ambient Background Glow */}
            <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600/5 rounded-full blur-[100px] pointer-events-none" />
            <div className="absolute bottom-0 left-0 w-96 h-96 bg-cyan-600/5 rounded-full blur-[100px] pointer-events-none" />

            {/* Header */}
            <div className="px-6 py-5 bg-slate-900/50 backdrop-blur-md border-b border-white/5 z-10 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="relative">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-blue-500/20 text-white">
                            <BotIcon />
                        </div>
                        <div className="absolute -bottom-1 -right-1 w-3.5 h-3.5 bg-green-500 rounded-full border-2 border-slate-900"></div>
                    </div>
                    <div>
                        <h3 className="text-lg font-bold text-white tracking-wide">AI Data Assistant</h3>
                        <p className="text-cyan-200/60 text-xs font-medium uppercase tracking-wider">
                            IntelliML Core Active
                        </p>
                    </div>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => setShowSuggestions(!showSuggestions)}
                        className={`px-3 py-2 text-xs font-medium rounded-lg transition-all flex items-center gap-1.5 border ${showSuggestions
                                ? 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                                : 'bg-white/5 text-gray-400 border-white/5 hover:bg-white/10 hover:text-white'
                            }`}
                    >
                        <SparklesIcon /> Ideas
                    </button>
                    <button
                        onClick={handleClear}
                        className="px-3 py-2 text-xs font-medium bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white rounded-lg transition-all border border-white/5"
                    >
                        Clear
                    </button>
                </div>
            </div>

            {/* Suggestions Panel */}
            {showSuggestions && (
                <div className="px-4 py-3 bg-slate-900/80 backdrop-blur-md border-b border-white/5 animate-fadeIn z-10">
                    <p className="text-[10px] font-bold text-cyan-400/80 uppercase tracking-widest mb-3 px-1">Suggested Visualizations</p>
                    <div className="flex flex-wrap gap-2">
                        {suggestions.map((s, idx) => (
                            <button
                                key={idx}
                                onClick={() => {
                                    handleSend(`Create a ${s.type} chart: ${s.description}`);
                                    setShowSuggestions(false);
                                }}
                                className="px-3 py-1.5 text-xs bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-200 border border-cyan-500/10 rounded-full transition-all hover:border-cyan-500/30"
                            >
                                {s.title}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6 z-0 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center p-8">
                        <div className="w-24 h-24 rounded-full bg-slate-900 border border-blue-500/20 flex items-center justify-center mb-8 relative">
                            <div className="absolute inset-0 rounded-full bg-blue-500/5 animate-pulse"></div>
                            <BotIcon />
                        </div>
                        <h4 className="text-2xl font-bold text-white mb-3">How can I help you?</h4>
                        <p className="text-gray-400 mb-8 max-w-sm text-sm leading-relaxed">
                            I can analyze your dataset, create visualizations, and answer questions. Just ask or use a preset below.
                        </p>
                        <div className="grid grid-cols-2 gap-3 max-w-md w-full">
                            {quickQuestions.map((q, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => handleSend(q)}
                                    className="px-4 py-3 text-xs font-medium bg-slate-800/50 hover:bg-slate-800 hover:border-blue-500/30 rounded-xl text-gray-300 hover:text-white transition-all border border-white/5 text-left"
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn group`}
                        >
                            <div className={`max-w-[85%] flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-white mt-1 ${msg.role === 'user'
                                    ? 'bg-blue-600'
                                    : 'bg-slate-800 border border-white/10'
                                    }`}>
                                    {msg.role === 'user' ? <UserIcon /> : <BotIcon />}
                                </div>

                                <div className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                    <div className={`rounded-2xl px-5 py-3.5 ${msg.role === 'user'
                                        ? 'bg-gradient-to-br from-blue-600 to-cyan-600 text-white rounded-tr-sm shadow-lg shadow-blue-500/10'
                                        : msg.error
                                            ? 'bg-red-950/30 text-red-200 border border-red-500/20 rounded-tl-sm'
                                            : 'bg-slate-800/80 text-gray-200 border border-white/5 rounded-tl-sm backdrop-blur-sm'
                                        }`}>
                                        <div className="text-sm leading-relaxed">
                                            {msg.role === 'user' ? msg.content : renderMarkdown(msg.content)}
                                        </div>

                                        {msg.output && (
                                            <div className="mt-4 border-t border-white/10 pt-3">
                                                <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">System Output</div>
                                                <div className="bg-black/40 rounded-lg p-3 overflow-x-auto border border-white/5">
                                                    <pre className="text-cyan-300/90 text-xs font-mono whitespace-pre-wrap">{msg.output}</pre>
                                                </div>
                                            </div>
                                        )}

                                        {msg.visualization && (
                                            <div className="mt-4">
                                                <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                                                    <span className="font-bold uppercase tracking-widest text-[10px]">Generated Plot</span>
                                                    <button
                                                        onClick={() => {
                                                            const link = document.createElement('a');
                                                            link.href = msg.visualization!;
                                                            link.download = `viz-${Date.now()}.png`;
                                                            link.click();
                                                        }}
                                                        className="hover:text-white transition-colors flex items-center gap-1.5 bg-white/5 hover:bg-white/10 px-2 py-1 rounded"
                                                    >
                                                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                                                        Download
                                                    </button>
                                                </div>
                                                <img
                                                    src={msg.visualization}
                                                    alt="Data visualization"
                                                    className="rounded-lg border border-white/10 max-w-full bg-slate-900/50"
                                                />
                                            </div>
                                        )}
                                    </div>
                                    <span className="text-[10px] text-gray-600 mt-1.5 px-1 font-medium opactiy-0 group-hover:opacity-100 transition-opacity">
                                        {formatTime(msg.timestamp)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))
                )}

                {/* Loading State */}
                {(isLoading || isTranscribing) && (
                    <div className="flex justify-start animate-fadeIn">
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-lg bg-slate-800 border border-white/10 flex items-center justify-center text-white mt-1">
                                <BotIcon />
                            </div>
                            <div className="bg-slate-800/50 rounded-2xl rounded-tl-sm px-4 py-3 border border-white/5 flex items-center gap-3">
                                <div className="flex gap-1">
                                    <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                    <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                    <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                </div>
                                <span className="text-xs font-medium text-cyan-200/70">
                                    {isTranscribing ? 'Listening & Transcribing...' : 'Processing...'}
                                </span>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Recording Indicator */}
            {isRecording && (
                <div className="absolute bottom-24 left-0 right-0 flex justify-center pointer-events-none z-20">
                    <div className="bg-slate-900/90 backdrop-blur-md border border-red-500/30 rounded-full px-6 py-2 shadow-2xl shadow-red-500/10 flex items-center gap-4 animate-in slide-in-from-bottom-5">
                        <div className="flex items-center gap-1 h-4">
                            {[...Array(8)].map((_, i) => (
                                <div
                                    key={i}
                                    className="w-1 bg-red-500 rounded-full animate-pulse"
                                    style={{
                                        height: `${8 + Math.random() * 12}px`,
                                        animationDelay: `${i * 0.1}s`,
                                        animationDuration: '0.6s',
                                    }}
                                />
                            ))}
                        </div>
                        <span className="text-red-400 text-xs font-bold uppercase tracking-wider animate-pulse">
                            Listening
                        </span>
                    </div>
                </div>
            )}

            {/* Input Area */}
            <div className="p-4 z-20 relative">
                <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-slate-950/90 to-transparent pointer-events-none" />
                <div className="relative flex gap-2 items-end max-w-3xl mx-auto bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-2xl p-2 shadow-2xl shadow-black/50">
                    <button
                        onClick={handleVoiceClick}
                        disabled={isPreparing || isLoading || isTranscribing}
                        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all flex-shrink-0 text-white ${isRecording
                            ? 'bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/20'
                            : isPreparing || isTranscribing
                                ? 'bg-slate-700 cursor-wait'
                                : 'bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-white border border-white/5 hover:border-white/10'
                            }`}
                        title={isRecording ? 'Stop Recording' : 'Start Voice Input'}
                    >
                        {isPreparing || isTranscribing ? <SpinnerIcon /> : isRecording ? <StopIcon /> : <MicIcon />}
                    </button>

                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder={isRecording ? "Listening..." : "Ask something..."}
                        className="flex-1 bg-transparent border-none rounded-none focus:ring-0 text-white placeholder-gray-500 text-sm py-2.5 px-2"
                        disabled={isLoading || isRecording || isTranscribing}
                    />

                    <button
                        onClick={() => handleSend()}
                        disabled={!input.trim() || isLoading || isRecording}
                        className="w-10 h-10 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-gray-600 flex items-center justify-center transition-all shadow-lg shadow-blue-500/20 disabled:shadow-none flex-shrink-0 text-white"
                        title="Send Message"
                    >
                        <SendIcon />
                    </button>
                </div>
                {voiceError && (
                    <p className="text-red-400 text-[10px] mt-2 text-center absolute bottom-1 left-0 right-0">
                        {voiceError}
                    </p>
                )}
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(5px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fadeIn {
                    animation: fadeIn 0.3s ease-out;
                }
            `}</style>
        </div>
    );
}
