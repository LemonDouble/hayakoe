import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { Fragment, useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router";
import api from "../api/client";
import * as speakersApi from "../api/speakers";
import * as videosApi from "../api/videos";
import * as datasetApi from "../api/dataset";
const STAGE_LABELS = {
    extract: "추출 대기",
    separate: "분리 대기",
    vad: "VAD 대기",
    classify: "분류 대기",
    classifying: "분류 중",
    transcribe: "전사 대기",
    review: "검토 대기",
    done: "완료",
    empty: "소스 없음",
};
function stageLabel(stage) {
    if (stage.startsWith("processing:")) {
        const s = stage.split(":")[1];
        return `처리 중 (${s})`;
    }
    return STAGE_LABELS[stage] || stage;
}
function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return m > 0 ? `${m}분 ${s}초` : `${s}초`;
}
const WORKFLOW_STEPS = [
    { n: "1", label: "화자 등록", desc: "목소리 주인 등록" },
    { n: "2", label: "영상·오디오 업로드", desc: "학습 소스 준비" },
    { n: "3", label: "영상별 전처리", desc: "6단계 파이프라인" },
    { n: "4", label: "데이터셋 생성", desc: "학습 데이터 출력" },
    { n: "5", label: "CLI 학습", desc: "남은 학습은 CLI에서" },
];
export default function Dashboard() {
    const navigate = useNavigate();
    const [speakers, setSpeakers] = useState([]);
    const [videos, setVideos] = useState([]);
    const [speakerSummary, setSpeakerSummary] = useState([]);
    const [newSpeaker, setNewSpeaker] = useState("");
    const [error, setError] = useState("");
    const [valRatio, setValRatio] = useState(0.1);
    const [datasetResult, setDatasetResult] = useState(null);
    const [building, setBuilding] = useState(false);
    const [dataDir, setDataDir] = useState("");
    const fileRef = useRef(null);
    const refresh = async () => {
        const [s, v] = await Promise.all([
            speakersApi.listSpeakers(),
            videosApi.listVideos(),
        ]);
        setSpeakers(s);
        setVideos(v);
        // 화자 요약은 별도 (에러 무시 — 세그먼트 없을 수 있음)
        speakersApi.getSummary().then(setSpeakerSummary).catch(() => { });
    };
    useEffect(() => {
        refresh();
        api.get("/info").then(({ data }) => setDataDir(data.data_dir));
    }, []);
    const handleAddSpeaker = async () => {
        if (!newSpeaker.trim())
            return;
        try {
            const s = await speakersApi.addSpeaker(newSpeaker.trim());
            setSpeakers(s);
            setNewSpeaker("");
            setError("");
        }
        catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            setError(msg);
        }
    };
    const handleDeleteSpeaker = async (name) => {
        if (!confirm(`화자 "${name}"을(를) 삭제하시겠습니까?`))
            return;
        const s = await speakersApi.deleteSpeaker(name);
        setSpeakers(s);
    };
    const handleUpload = async (e) => {
        const file = e.target.files?.[0];
        if (!file)
            return;
        if (file.name.toLowerCase() === "extracted.wav") {
            alert("파일명이 'extracted.wav' 인 파일은 업로드할 수 없습니다.\n" +
                "전처리 파이프라인 내부에서 사용하는 이름과 충돌하므로, 다른 이름으로 바꿔 다시 업로드해주세요.");
            if (fileRef.current)
                fileRef.current.value = "";
            return;
        }
        await videosApi.uploadVideo(file);
        await refresh();
        if (fileRef.current)
            fileRef.current.value = "";
    };
    const handleDeleteVideo = async (id) => {
        if (!confirm(`영상 ${id}를 삭제하시겠습니까?`))
            return;
        await videosApi.deleteVideo(id);
        await refresh();
    };
    const handleBuildDataset = async () => {
        setBuilding(true);
        setDatasetResult(null);
        try {
            const result = await datasetApi.buildDataset(valRatio);
            setDatasetResult(result);
            setError("");
        }
        catch (e) {
            const detail = e?.response?.data
                ?.detail || String(e);
            setError(detail);
        }
        finally {
            setBuilding(false);
        }
    };
    const hasSpeakers = speakers.length > 0;
    const hasDoneVideos = videos.some((v) => v.stage === "done");
    return (_jsxs("div", { className: "max-w-4xl mx-auto p-6 pb-16", children: [_jsxs("h1", { className: "font-display text-3xl font-bold mb-1 text-fg tracking-tight", children: [_jsx("span", { className: "text-primary", children: "HayaKoe" }), " \uC804\uCC98\uB9AC"] }), dataDir && (_jsxs("p", { className: "text-fg-dim text-xs mb-8 font-mono", children: ["\uB370\uC774\uD130 \uACBD\uB85C: ", dataDir] })), _jsxs("div", { className: "bg-primary/[0.08] border border-primary/25 rounded-xl p-5 mb-8", children: [_jsx("p", { className: "text-primary text-[11px] font-semibold uppercase tracking-[1.5px] mb-1.5 font-display", children: "WORKFLOW" }), _jsx("p", { className: "text-fg-muted text-xs mb-4", children: "\uC74C\uC131 \uB370\uC774\uD130\uB97C \uC218\uC9D1\uD558\uC5EC TTS \uD559\uC2B5\uC6A9 \uB370\uC774\uD130\uC14B\uC744 \uB9CC\uB4DC\uB294 \uB3C4\uAD6C\uC785\uB2C8\uB2E4. \uC544\uB798 \uC21C\uC11C\uB300\uB85C \uC9C4\uD589\uD558\uC138\uC694." }), _jsx("div", { className: "flex items-center gap-3", children: WORKFLOW_STEPS.map((step, i) => (_jsxs(Fragment, { children: [i > 0 && (_jsx("span", { className: "text-primary/60 text-base shrink-0 font-bold select-none", "aria-hidden": "true", children: "\u2192" })), _jsxs("div", { className: "flex items-center justify-center gap-2 flex-1 min-w-0", children: [_jsx("span", { className: "w-6 h-6 rounded-full bg-primary text-canvas flex items-center justify-center text-xs font-bold shrink-0", children: step.n }), _jsxs("div", { className: "min-w-0", children: [_jsx("p", { className: "text-fg text-xs font-semibold truncate", children: step.label }), _jsx("p", { className: "text-fg-dim text-[10px] truncate", children: step.desc })] })] })] }, step.n))) })] }), error && (_jsx("div", { className: "bg-error/10 border border-error/25 rounded-lg p-3 mb-4 text-error text-sm", children: error })), _jsxs("section", { className: "mb-5 bg-surface border border-line rounded-xl p-6", children: [_jsxs("div", { className: "flex items-center gap-2.5 mb-1", children: [_jsx("span", { className: "w-7 h-7 rounded-full bg-primary text-canvas flex items-center justify-center text-xs font-bold shrink-0", children: "1" }), _jsx("h2", { className: "font-display text-xl font-bold text-fg", children: "\uD654\uC790 \uB4F1\uB85D" })] }), _jsx("p", { className: "text-fg-muted text-sm mb-4 ml-10 leading-relaxed", children: "TTS \uBAA8\uB378\uC744 \uD559\uC2B5\uD560 \uD654\uC790(\uBAA9\uC18C\uB9AC\uC758 \uC8FC\uC778)\uB97C \uB4F1\uB85D\uD569\uB2C8\uB2E4. \uC601\uC0C1\uC5D0 \uC5EC\uB7EC \uC0AC\uB78C\uC774 \uB4F1\uC7A5\uD558\uBA74, \uB098\uC911\uC5D0 \uAC01 \uC74C\uC131 \uAD6C\uAC04\uC744 \uD654\uC790\uBCC4\uB85C \uBD84\uB958\uD558\uAC8C \uB429\uB2C8\uB2E4." }), _jsxs("div", { className: "ml-10", children: [_jsxs("div", { className: "flex gap-2 mb-3", children: [_jsx("input", { className: "bg-surface border border-line rounded-lg px-4 py-2 flex-1 text-fg placeholder:text-fg-dim focus:border-primary/50 focus:outline-none transition-colors", placeholder: "\uD654\uC790 \uC774\uB984 (\uC608: \uD64D\uAE38\uB3D9)", value: newSpeaker, onChange: (e) => setNewSpeaker(e.target.value), onKeyDown: (e) => e.key === "Enter" && handleAddSpeaker() }), _jsx("button", { className: "bg-primary hover:bg-primary-hover text-canvas px-5 py-2 rounded-lg font-semibold text-sm transition-colors", onClick: handleAddSpeaker, children: "\uCD94\uAC00" })] }), speakers.length === 0 ? (_jsx("div", { className: "text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center", children: "\uB4F1\uB85D\uB41C \uD654\uC790\uAC00 \uC5C6\uC2B5\uB2C8\uB2E4. \uC704 \uC785\uB825\uB780\uC5D0 \uD654\uC790 \uC774\uB984\uC744 \uC785\uB825\uD558\uACE0 \uCD94\uAC00 \uBC84\uD2BC\uC744 \uB20C\uB7EC\uC8FC\uC138\uC694." })) : (_jsx("div", { className: "flex flex-wrap gap-2", children: speakers.map((s) => (_jsxs("span", { className: "bg-primary/[0.12] border border-primary/25 text-primary px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-2", children: [s, _jsx("button", { className: "text-primary/60 hover:text-error transition-colors", onClick: () => handleDeleteSpeaker(s), children: "\u00D7" })] }, s))) }))] })] }), _jsxs("section", { className: `mb-5 bg-surface border border-line rounded-xl p-6 transition-opacity ${!hasSpeakers ? "opacity-40" : ""}`, children: [_jsxs("div", { className: "flex items-center gap-2.5 mb-1", children: [_jsx("span", { className: `w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${hasSpeakers ? "bg-primary text-canvas" : "bg-line text-fg-dim"}`, children: "2" }), _jsx("h2", { className: "font-display text-xl font-bold text-fg", children: "\uC601\uC0C1\u00B7\uC624\uB514\uC624 \uC5C5\uB85C\uB4DC" })] }), _jsxs("p", { className: "text-fg-muted text-sm mb-4 ml-10 leading-relaxed", children: ["\uD559\uC2B5\uC5D0 \uC0AC\uC6A9\uD560 \uC601\uC0C1 \uB610\uB294 \uC624\uB514\uC624 \uD30C\uC77C\uC744 \uC5C5\uB85C\uB4DC\uD569\uB2C8\uB2E4.", _jsx("br", {}), "\uC5EC\uB7EC \uC601\uC0C1\uC744 \uCD94\uAC00\uD558\uC5EC \uB370\uC774\uD130\uB97C \uB354 \uB9CE\uC774 \uC218\uC9D1\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4. \uC5C5\uB85C\uB4DC \uD6C4 \uC601\uC0C1\uC744 \uD074\uB9AD\uD558\uBA74 \uC804\uCC98\uB9AC\uB97C \uC2DC\uC791\uD569\uB2C8\uB2E4."] }), _jsx("div", { className: "ml-10", children: !hasSpeakers ? (_jsx("div", { className: "text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center", children: "\uBA3C\uC800 \uC704\uC5D0\uC11C \uD654\uC790\uB97C \uB4F1\uB85D\uD574\uC8FC\uC138\uC694." })) : (_jsxs(_Fragment, { children: [_jsxs("div", { className: "mb-4", children: [_jsxs("label", { className: "inline-flex items-center gap-2 bg-secondary/[0.12] border border-secondary/25 rounded-lg px-4 py-2.5 cursor-pointer hover:bg-secondary/20 transition-colors", children: [_jsx("span", { className: "text-secondary text-sm font-semibold", children: "\uD30C\uC77C \uC120\uD0DD" }), _jsx("input", { ref: fileRef, type: "file", accept: "video/*,audio/*", onChange: handleUpload, className: "hidden" })] }), _jsx("span", { className: "text-fg-dim text-xs ml-3", children: "\uB3D9\uC601\uC0C1 \uB610\uB294 \uC624\uB514\uC624 \uD30C\uC77C" })] }), videos.length === 0 ? (_jsx("div", { className: "text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center", children: "\uC5C5\uB85C\uB4DC\uB41C \uC601\uC0C1\uC774 \uC5C6\uC2B5\uB2C8\uB2E4. \uC704 \uBC84\uD2BC\uC73C\uB85C \uD30C\uC77C\uC744 \uC5C5\uB85C\uB4DC\uD574\uC8FC\uC138\uC694." })) : (_jsx("div", { className: "space-y-2", children: videos.map((v) => (_jsxs("div", { className: "flex items-center gap-4 bg-canvas border border-line hover:border-primary/25 rounded-lg px-4 py-3 cursor-pointer transition-colors group", onClick: () => navigate(`/video/${v.id}`), children: [_jsx("span", { className: "font-mono text-xs text-fg-dim w-20 shrink-0", children: v.id }), _jsx("span", { className: "flex-1 text-sm text-fg truncate", children: v.filename }), _jsx("span", { className: `px-2.5 py-0.5 rounded-full text-[11px] font-semibold shrink-0 border ${v.stage === "done"
                                                    ? "bg-success/10 border-success/25 text-success"
                                                    : v.stage.startsWith("processing")
                                                        ? "bg-warning/10 border-warning/25 text-warning animate-pulse"
                                                        : "bg-surface-2 border-line text-fg-muted"}`, children: stageLabel(v.stage) }), _jsx("button", { className: "text-fg-dim hover:text-error text-xs shrink-0 transition-colors", onClick: (e) => {
                                                    e.stopPropagation();
                                                    handleDeleteVideo(v.id);
                                                }, children: "\uC0AD\uC81C" })] }, v.id))) }))] })) })] }), speakerSummary.some((s) => s.count > 0) && (_jsxs("section", { className: "mb-5 bg-surface border border-line rounded-xl p-6", children: [_jsx("h2", { className: "font-display text-xl font-bold text-fg mb-1", children: "\uC218\uC9D1 \uD604\uD669" }), _jsx("p", { className: "text-fg-muted text-sm mb-1", children: "\uC804\uCC98\uB9AC\uAC00 \uC644\uB8CC\uB41C \uC601\uC0C1\uC5D0\uC11C \uC218\uC9D1\uB41C \uD654\uC790\uBCC4 \uC74C\uC131 \uB370\uC774\uD130\uC785\uB2C8\uB2E4." }), _jsxs("p", { className: "text-fg-dim text-xs mb-4", children: ["\uBE60\uB974\uAC8C \uD14C\uC2A4\uD2B8\uD558\uB824\uBA74 ", _jsx("span", { className: "text-fg", children: "\uC57D 10\uBD84" }), ", \uC548\uC815\uC801\uC778 \uD488\uC9C8\uC744 \uC6D0\uD558\uBA74 ", _jsx("span", { className: "text-fg", children: "20~45\uBD84" }), " \uC815\uB3C4\uC758 \uAE68\uB057\uD55C \uC74C\uC131 \uB370\uC774\uD130\uB97C \uC900\uBE44\uD558\uBA74 \uC88B\uC2B5\uB2C8\uB2E4."] }), _jsx("div", { className: "grid grid-cols-2 gap-3", children: speakerSummary
                            .filter((s) => s.count > 0)
                            .map((s) => (_jsxs("div", { className: "bg-canvas border border-line rounded-lg px-4 py-3", children: [_jsxs("div", { className: "flex justify-between items-center mb-1", children: [_jsx("span", { className: "text-fg font-semibold text-sm", children: s.name }), _jsxs("span", { className: "text-primary text-sm font-mono font-semibold", children: [s.count, "\uAC1C"] })] }), _jsxs("p", { className: "text-xs text-fg-dim", children: ["\uCD1D ", formatDuration(s.total_duration)] })] }, s.name))) })] })), _jsxs("section", { className: `bg-surface border border-line rounded-xl p-6 transition-opacity ${!hasDoneVideos ? "opacity-40" : ""}`, children: [_jsxs("div", { className: "flex items-center gap-2.5 mb-1", children: [_jsx("span", { className: `w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${hasDoneVideos ? "bg-primary text-canvas" : "bg-line text-fg-dim"}`, children: "4" }), _jsx("h2", { className: "font-display text-xl font-bold text-fg", children: "\uB370\uC774\uD130\uC14B \uC0DD\uC131" })] }), _jsx("p", { className: "text-fg-muted text-sm mb-4 ml-10 leading-relaxed", children: "\uC218\uC9D1\uB41C \uC74C\uC131 \uB370\uC774\uD130\uB97C TTS \uD559\uC2B5\uC5D0 \uC0AC\uC6A9\uD560 \uC218 \uC788\uB294 \uB370\uC774\uD130\uC14B\uC73C\uB85C \uBCC0\uD658\uD569\uB2C8\uB2E4. \uBAA8\uB4E0 \uC601\uC0C1\uC758 \uC804\uCC98\uB9AC\uB97C \uC644\uB8CC\uD55C \uD6C4 \uC2E4\uD589\uD558\uC138\uC694." }), _jsx("div", { className: "ml-10", children: !hasDoneVideos ? (_jsx("div", { className: "text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center", children: "\uC804\uCC98\uB9AC\uAC00 \uC644\uB8CC\uB41C \uC601\uC0C1\uC774 \uC5C6\uC2B5\uB2C8\uB2E4. \uBA3C\uC800 \uC601\uC0C1\uC744 \uC5C5\uB85C\uB4DC\uD558\uACE0 \uC804\uCC98\uB9AC\uB97C \uC9C4\uD589\uD574\uC8FC\uC138\uC694." })) : (_jsxs(_Fragment, { children: [_jsxs("div", { className: "bg-canvas border border-line rounded-lg p-4 mb-4", children: [_jsx("label", { className: "text-sm text-fg font-semibold block mb-1", children: "\uAC80\uC99D(Validation) \uB370\uC774\uD130 \uBE44\uC728" }), _jsx("p", { className: "text-xs text-fg-dim mb-3 leading-relaxed", children: "\uC804\uCCB4 \uB370\uC774\uD130 \uC911 \uD559\uC2B5 \uD488\uC9C8 \uAC80\uC99D\uC5D0 \uC0AC\uC6A9\uD560 \uBE44\uC728\uC785\uB2C8\uB2E4. \uC77C\uBC18\uC801\uC73C\uB85C 10~20%\uAC00 \uC801\uC808\uD569\uB2C8\uB2E4. \uB098\uBA38\uC9C0\uB294 \uD559\uC2B5(Training)\uC5D0 \uC0AC\uC6A9\uB429\uB2C8\uB2E4." }), _jsxs("div", { className: "flex items-center gap-3", children: [_jsx("input", { type: "range", min: "0.01", max: "0.5", step: "0.01", className: "flex-1 accent-primary", value: valRatio, onChange: (e) => setValRatio(+e.target.value) }), _jsxs("span", { className: "text-primary font-mono font-semibold text-sm w-12 text-right", children: [Math.round(valRatio * 100), "%"] })] })] }), _jsx("button", { className: "bg-primary hover:bg-primary-hover text-canvas px-6 py-2.5 rounded-lg font-semibold text-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors", onClick: handleBuildDataset, disabled: building, children: building ? "생성 중..." : "데이터셋 생성" }), _jsx("p", { className: "text-fg-dim text-xs mt-2", children: "\uC774\uBBF8 \uB370\uC774\uD130\uC14B\uC774 \uC788\uB294 \uACBD\uC6B0, \uAE30\uC874 \uB370\uC774\uD130\uC14B\uC744 \uC0AD\uC81C\uD558\uACE0 \uC0C8\uB85C \uC0DD\uC131\uD569\uB2C8\uB2E4." }), datasetResult && (_jsxs("div", { className: "bg-success/[0.06] border border-success/25 rounded-lg p-5 mt-4", children: [_jsx("p", { className: "text-success font-semibold mb-1", children: "\uB370\uC774\uD130\uC14B \uC0DD\uC131 \uC644\uB8CC" }), _jsx("p", { className: "text-fg-dim text-xs mb-4 font-mono", children: datasetResult.dataset_dir }), Object.entries(datasetResult.speakers).map(([name, s]) => (_jsxs("div", { className: "mb-3 last:mb-0", children: [_jsx("p", { className: "text-fg font-semibold mb-1", children: name }), _jsxs("div", { className: "grid grid-cols-3 gap-2 text-xs text-fg-muted bg-canvas border border-line rounded-lg p-3", children: [_jsxs("div", { children: [_jsx("p", { className: "text-fg-dim mb-0.5", children: "\uC804\uCCB4" }), _jsxs("p", { children: [s.count, "\uAC1C / ", formatDuration(s.duration)] })] }), _jsxs("div", { children: [_jsx("p", { className: "text-fg-dim mb-0.5", children: "Train" }), _jsxs("p", { children: [s.train_count, "\uAC1C / ", formatDuration(s.train_duration)] })] }), _jsxs("div", { children: [_jsx("p", { className: "text-fg-dim mb-0.5", children: "Val" }), _jsxs("p", { children: [s.val_count, "\uAC1C / ", formatDuration(s.val_duration)] })] })] })] }, name)))] }))] })) })] })] }));
}
