import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useEffect, useRef, useState } from "react";
import * as reviewApi from "../api/review";
export default function ReviewEditor({ videoId, onDone }) {
    const [entries, setEntries] = useState([]);
    const [loading, setLoading] = useState(true);
    const [playingFile, setPlayingFile] = useState(null);
    const [editingFile, setEditingFile] = useState(null);
    const [editText, setEditText] = useState("");
    const [saving, setSaving] = useState(false);
    const [activeSpeaker, setActiveSpeaker] = useState(null);
    const audioRef = useRef(null);
    const load = useCallback(async () => {
        const data = await reviewApi.getTranscriptions(videoId);
        setEntries(data.entries);
        setLoading(false);
    }, [videoId]);
    useEffect(() => {
        load();
    }, [load]);
    // 화자 목록
    const speakerList = [...new Set(entries.map((e) => e.speaker))];
    // 첫 화자 자동 선택
    useEffect(() => {
        if (activeSpeaker === null && speakerList.length > 0) {
            setActiveSpeaker(speakerList[0]);
        }
    }, [speakerList, activeSpeaker]);
    // 현재 화자의 엔트리
    const filtered = activeSpeaker
        ? entries.filter((e) => e.speaker === activeSpeaker)
        : entries;
    // 화자별 통계
    const speakerCounts = new Map();
    for (const e of entries) {
        speakerCounts.set(e.speaker, (speakerCounts.get(e.speaker) || 0) + 1);
    }
    const play = (entry) => {
        if (!audioRef.current)
            return;
        const url = `/api/media/videos/${videoId}/segments/${entry.speaker}/${entry.file}`;
        audioRef.current.src = url;
        audioRef.current.play().catch(() => { });
        setPlayingFile(entry.file);
    };
    const handleAudioEnded = () => setPlayingFile(null);
    const startEdit = (entry) => {
        setEditingFile(entry.file);
        setEditText(entry.text);
    };
    const cancelEdit = () => {
        setEditingFile(null);
        setEditText("");
    };
    const saveEdit = async () => {
        if (editingFile === null)
            return;
        setSaving(true);
        await reviewApi.editTranscription(videoId, editingFile, editText);
        setEntries((prev) => prev.map((e) => (e.file === editingFile ? { ...e, text: editText } : e)));
        setEditingFile(null);
        setEditText("");
        setSaving(false);
    };
    const handleDelete = async (entry) => {
        if (!confirm(`"${entry.file}" 전사를 삭제하시겠습니까?`))
            return;
        await reviewApi.deleteTranscription(videoId, entry.file);
        setEntries((prev) => prev.filter((e) => e.file !== entry.file));
        if (editingFile === entry.file)
            cancelEdit();
    };
    const handleDone = async () => {
        if (!confirm("검토를 완료하시겠습니까?"))
            return;
        await reviewApi.markReviewDone(videoId);
        onDone();
    };
    if (loading)
        return _jsx("div", { className: "p-6 text-fg-muted", children: "\uB85C\uB529 \uC911..." });
    return (_jsxs("div", { className: "bg-surface border border-line rounded-xl p-6", children: [_jsx("audio", { ref: audioRef, onEnded: handleAudioEnded }), _jsxs("div", { className: "bg-primary/[0.08] border border-primary/25 rounded-lg p-4 mb-5", children: [_jsx("p", { className: "text-primary text-[11px] font-bold uppercase tracking-[1.5px] mb-1.5 font-display", children: "\uC804\uC0AC \uAC80\uD1A0" }), _jsx("p", { className: "text-fg-muted text-xs leading-relaxed mb-2", children: "\uC790\uB3D9 \uC804\uC0AC(STT) \uACB0\uACFC\uB97C \uD655\uC778\uD558\uACE0 \uC624\uB958\uB97C \uC218\uC815\uD558\uC138\uC694. \uC815\uD655\uD55C \uD14D\uC2A4\uD2B8\uAC00 TTS \uD559\uC2B5 \uD488\uC9C8\uC5D0 \uC9C1\uC811 \uC601\uD5A5\uC744 \uBBF8\uCE69\uB2C8\uB2E4." }), _jsxs("ul", { className: "text-xs text-fg-dim space-y-0.5 list-disc list-inside marker:text-primary", children: [_jsx("li", { children: "\uC7AC\uC0DD \uBC84\uD2BC\uC744 \uB20C\uB7EC \uC2E4\uC81C \uBC1C\uD654\uB97C \uB4E4\uC73C\uBA70 \uD14D\uC2A4\uD2B8\uC640 \uBE44\uAD50\uD558\uC138\uC694" }), _jsx("li", { children: "\uD14D\uC2A4\uD2B8\uB97C \uD074\uB9AD\uD558\uBA74 \uBC14\uB85C \uC218\uC815\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4 (Enter\uB85C \uC800\uC7A5, Esc\uB85C \uCDE8\uC18C)" }), _jsx("li", { children: "\uC758\uBBF8 \uC5C6\uB294 \uAD6C\uAC04\uC774\uB098 \uC798\uBABB\uB41C \uD56D\uBAA9\uC740 \u00D7 \uBC84\uD2BC\uC73C\uB85C \uC0AD\uC81C\uD558\uC138\uC694" })] })] }), _jsxs("div", { className: "flex justify-between items-center mb-4", children: [_jsxs("p", { className: "text-sm text-fg-muted", children: ["\uC804\uCCB4 ", _jsx("span", { className: "font-mono text-primary font-semibold", children: entries.length }), "\uAC1C \uC138\uADF8\uBA3C\uD2B8"] }), _jsx("button", { className: "bg-primary hover:bg-primary-hover text-canvas px-5 py-2 rounded-lg text-sm font-semibold transition-colors", onClick: handleDone, children: "\uAC80\uD1A0 \uC644\uB8CC" })] }), _jsx("div", { className: "flex gap-0 mb-4 overflow-x-auto border-b border-line", children: speakerList.map((speaker) => (_jsxs("button", { className: `px-5 py-2.5 text-sm font-semibold transition-colors shrink-0 border-b-2 -mb-px ${activeSpeaker === speaker
                        ? "text-primary border-primary"
                        : "text-fg-dim border-transparent hover:text-fg-muted"}`, onClick: () => setActiveSpeaker(speaker), children: [speaker, _jsxs("span", { className: "ml-1.5 text-xs opacity-70 font-mono", children: ["(", speakerCounts.get(speaker) || 0, ")"] })] }, speaker))) }), _jsx("div", { className: "space-y-2 max-h-[60vh] overflow-y-auto pr-1", children: filtered.length === 0 ? (_jsx("div", { className: "text-fg-dim text-sm text-center py-8", children: "\uC774 \uD654\uC790\uC758 \uC138\uADF8\uBA3C\uD2B8\uAC00 \uC5C6\uC2B5\uB2C8\uB2E4." })) : (filtered.map((entry) => (_jsxs("div", { className: `flex items-start gap-3 p-3 rounded-lg border transition-colors ${playingFile === entry.file
                        ? "bg-primary/[0.08] border-primary/25"
                        : "bg-canvas border-line hover:border-line-strong"}`, children: [_jsx("button", { className: `mt-0.5 shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs transition-colors border ${playingFile === entry.file
                                ? "bg-primary border-primary text-canvas"
                                : "bg-surface border-line text-fg-muted hover:border-primary/50 hover:text-primary"}`, onClick: () => play(entry), title: "\uC7AC\uC0DD", children: playingFile === entry.file ? "\u23F9" : "\u25B6" }), _jsxs("div", { className: "flex-1 min-w-0", children: [_jsx("p", { className: "text-[10px] text-fg-dim mb-1 font-mono", children: entry.file }), editingFile === entry.file ? (_jsxs("div", { className: "flex gap-2", children: [_jsx("input", { type: "text", className: "flex-1 bg-surface border border-line rounded-lg px-3 py-1.5 text-sm text-fg focus:outline-none focus:border-primary/50 transition-colors", value: editText, onChange: (e) => setEditText(e.target.value), onKeyDown: (e) => {
                                                if (e.key === "Enter")
                                                    saveEdit();
                                                if (e.key === "Escape")
                                                    cancelEdit();
                                            }, autoFocus: true }), _jsx("button", { className: "bg-primary hover:bg-primary-hover text-canvas px-3 py-1 rounded-md text-xs font-semibold transition-colors disabled:opacity-40", onClick: saveEdit, disabled: saving, children: "\uC800\uC7A5" }), _jsx("button", { className: "bg-transparent border border-line hover:border-line-strong text-fg-muted hover:text-fg px-3 py-1 rounded-md text-xs font-semibold transition-colors", onClick: cancelEdit, children: "\uCDE8\uC18C" })] })) : (_jsx("p", { className: "text-sm text-fg cursor-pointer hover:text-primary transition-colors", onClick: () => startEdit(entry), title: "\uD074\uB9AD\uD558\uC5EC \uC218\uC815", children: entry.text || _jsx("span", { className: "text-fg-dim italic", children: "(\uBE48 \uD14D\uC2A4\uD2B8 - \uD074\uB9AD\uD558\uC5EC \uC785\uB825\uD558\uAC70\uB098 \uC0AD\uC81C\uD558\uC138\uC694)" }) }))] }), _jsx("button", { className: "text-fg-dim hover:text-error text-sm shrink-0 mt-0.5 transition-colors", onClick: () => handleDelete(entry), title: "\uC0AD\uC81C", children: "\u00D7" })] }, entry.file)))) })] }));
}
