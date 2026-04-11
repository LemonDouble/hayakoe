import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function ProgressBar({ progress, message }) {
    const hasProgress = progress > 0;
    const pct = Math.round(progress * 100);
    return (_jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-sm mb-1", children: [_jsx("span", { children: message || "처리 중..." }), hasProgress && _jsxs("span", { children: [pct, "%"] })] }), hasProgress ? (_jsx("div", { className: "w-full bg-slate-700 rounded h-3", children: _jsx("div", { className: "bg-blue-500 h-3 rounded transition-all duration-300", style: { width: `${pct}%` } }) })) : (_jsx("div", { className: "w-full bg-slate-700 rounded h-3 overflow-hidden", children: _jsx("div", { className: "h-3 rounded bg-blue-500/60 animate-indeterminate" }) }))] }));
}
