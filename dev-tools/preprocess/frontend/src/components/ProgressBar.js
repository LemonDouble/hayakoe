import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function ProgressBar({ progress, message }) {
    const hasProgress = progress > 0;
    const pct = Math.round(progress * 100);
    return (_jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-sm mb-1.5", children: [_jsx("span", { className: "text-fg-muted", children: message || "처리 중..." }), hasProgress && _jsxs("span", { className: "text-primary font-mono font-semibold", children: [pct, "%"] })] }), hasProgress ? (_jsx("div", { className: "w-full bg-line rounded-full h-1.5 overflow-hidden", children: _jsx("div", { className: "progress-fill h-1.5 rounded-full transition-all duration-300", style: { width: `${pct}%` } }) })) : (_jsx("div", { className: "w-full bg-line rounded-full h-1.5 overflow-hidden", children: _jsx("div", { className: "progress-fill h-1.5 rounded-full animate-indeterminate" }) }))] }));
}
