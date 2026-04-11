import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState, useRef } from "react";
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
    { n: "2", label: "영상 업로드", desc: "학습 소스 준비" },
    { n: "3", label: "영상별 전처리", desc: "6단계 파이프라인" },
    { n: "4", label: "데이터셋 생성", desc: "학습 데이터 출력" },
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
    return (_jsxs("div", { className: "max-w-4xl mx-auto p-6 pb-16", children: [_jsx("h1", { className: "text-2xl font-bold mb-1", children: "HayaKoe \uC804\uCC98\uB9AC" }), dataDir && (_jsxs("p", { className: "text-slate-500 text-sm mb-6 font-mono", children: ["\uB370\uC774\uD130 \uACBD\uB85C: ", dataDir] })), _jsxs("div", { className: "bg-gradient-to-r from-blue-950/50 to-slate-800/50 border border-slate-700 rounded-xl p-5 mb-8", children: [_jsx("p", { className: "text-slate-200 text-sm font-semibold mb-1", children: "\uC791\uC5C5 \uC21C\uC11C" }), _jsx("p", { className: "text-slate-400 text-xs mb-2", children: "\uC74C\uC131 \uB370\uC774\uD130\uB97C \uC218\uC9D1\uD558\uC5EC TTS \uD559\uC2B5\uC6A9 \uB370\uC774\uD130\uC14B\uC744 \uB9CC\uB4DC\uB294 \uB3C4\uAD6C\uC785\uB2C8\uB2E4. \uC544\uB798 \uC21C\uC11C\uB300\uB85C \uC9C4\uD589\uD558\uC138\uC694." }), _jsx("div", { className: "flex items-center gap-1", children: WORKFLOW_STEPS.map((step, i) => (_jsxs("div", { className: "flex items-center gap-1 flex-1", children: [i > 0 && _jsx("div", { className: "w-6 h-px bg-slate-600 shrink-0" }), _jsxs("div", { className: "flex items-center gap-2 min-w-0", children: [_jsx("span", { className: "w-6 h-6 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold shrink-0", children: step.n }), _jsxs("div", { className: "min-w-0", children: [_jsx("p", { className: "text-slate-200 text-xs font-medium truncate", children: step.label }), _jsx("p", { className: "text-slate-500 text-[10px] truncate", children: step.desc })] })] })] }, step.n))) })] }), error && (_jsx("div", { className: "bg-red-900/50 border border-red-500 rounded-lg p-3 mb-4 text-red-200", children: error })), _jsxs("section", { className: "mb-5 bg-slate-800/60 border border-slate-700 rounded-xl p-5", children: [_jsxs("div", { className: "flex items-center gap-2.5 mb-1", children: [_jsx("span", { className: "w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold shrink-0", children: "1" }), _jsx("h2", { className: "text-lg font-semibold", children: "\uD654\uC790 \uB4F1\uB85D" })] }), _jsx("p", { className: "text-slate-400 text-sm mb-4 ml-10", children: "TTS \uBAA8\uB378\uC744 \uD559\uC2B5\uD560 \uD654\uC790(\uBAA9\uC18C\uB9AC\uC758 \uC8FC\uC778)\uB97C \uB4F1\uB85D\uD569\uB2C8\uB2E4. \uC601\uC0C1\uC5D0 \uC5EC\uB7EC \uC0AC\uB78C\uC774 \uB4F1\uC7A5\uD558\uBA74, \uB098\uC911\uC5D0 \uAC01 \uC74C\uC131 \uAD6C\uAC04\uC744 \uD654\uC790\uBCC4\uB85C \uBD84\uB958\uD558\uAC8C \uB429\uB2C8\uB2E4." }), _jsxs("div", { className: "ml-10", children: [_jsxs("div", { className: "flex gap-2 mb-3", children: [_jsx("input", { className: "bg-slate-800 border border-slate-600 rounded-lg px-3 py-1.5 flex-1 focus:border-blue-500 focus:outline-none transition-colors", placeholder: "\uD654\uC790 \uC774\uB984 (\uC608: \uD64D\uAE38\uB3D9)", value: newSpeaker, onChange: (e) => setNewSpeaker(e.target.value), onKeyDown: (e) => e.key === "Enter" && handleAddSpeaker() }), _jsx("button", { className: "bg-blue-600 hover:bg-blue-700 px-4 py-1.5 rounded-lg font-medium transition-colors", onClick: handleAddSpeaker, children: "\uCD94\uAC00" })] }), speakers.length === 0 ? (_jsx("div", { className: "text-slate-500 text-sm bg-slate-700/30 rounded-lg p-4 text-center", children: "\uB4F1\uB85D\uB41C \uD654\uC790\uAC00 \uC5C6\uC2B5\uB2C8\uB2E4. \uC704 \uC785\uB825\uB780\uC5D0 \uD654\uC790 \uC774\uB984\uC744 \uC785\uB825\uD558\uACE0 \uCD94\uAC00 \uBC84\uD2BC\uC744 \uB20C\uB7EC\uC8FC\uC138\uC694." })) : (_jsx("div", { className: "flex flex-wrap gap-2", children: speakers.map((s) => (_jsxs("span", { className: "bg-slate-700 px-3 py-1.5 rounded-full text-sm flex items-center gap-2", children: [s, _jsx("button", { className: "text-slate-400 hover:text-red-400 transition-colors", onClick: () => handleDeleteSpeaker(s), children: "\u00D7" })] }, s))) }))] })] }), _jsxs("section", { className: `mb-5 bg-slate-800/60 border border-slate-700 rounded-xl p-5 transition-opacity ${!hasSpeakers ? "opacity-50" : ""}`, children: [_jsxs("div", { className: "flex items-center gap-2.5 mb-1", children: [_jsx("span", { className: `w-7 h-7 rounded-full text-white flex items-center justify-center text-xs font-bold shrink-0 ${hasSpeakers ? "bg-blue-600" : "bg-slate-600"}`, children: "2" }), _jsx("h2", { className: "text-lg font-semibold", children: "\uC601\uC0C1 \uC5C5\uB85C\uB4DC" })] }), _jsxs("p", { className: "text-slate-400 text-sm mb-4 ml-10", children: ["\uD559\uC2B5\uC5D0 \uC0AC\uC6A9\uD560 \uC601\uC0C1 \uB610\uB294 \uC624\uB514\uC624 \uD30C\uC77C\uC744 \uC5C5\uB85C\uB4DC\uD569\uB2C8\uB2E4.", _jsx("br", {}), "\uC5EC\uB7EC \uC601\uC0C1\uC744 \uCD94\uAC00\uD558\uC5EC \uB370\uC774\uD130\uB97C \uB354 \uB9CE\uC774 \uC218\uC9D1\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4. \uC5C5\uB85C\uB4DC \uD6C4 \uC601\uC0C1\uC744 \uD074\uB9AD\uD558\uBA74 \uC804\uCC98\uB9AC\uB97C \uC2DC\uC791\uD569\uB2C8\uB2E4."] }), _jsx("div", { className: "ml-10", children: !hasSpeakers ? (_jsx("div", { className: "text-slate-500 text-sm bg-slate-700/30 rounded-lg p-4 text-center", children: "\uBA3C\uC800 \uC704\uC5D0\uC11C \uD654\uC790\uB97C \uB4F1\uB85D\uD574\uC8FC\uC138\uC694." })) : (_jsxs(_Fragment, { children: [_jsxs("div", { className: "mb-4", children: [_jsxs("label", { className: "inline-flex items-center gap-2 bg-blue-600/20 border border-blue-500/30 rounded-lg px-4 py-2.5 cursor-pointer hover:bg-blue-600/30 transition-colors", children: [_jsx("span", { className: "text-blue-300 text-sm font-medium", children: "\uD30C\uC77C \uC120\uD0DD" }), _jsx("input", { ref: fileRef, type: "file", accept: "video/*,audio/*", onChange: handleUpload, className: "hidden" })] }), _jsx("span", { className: "text-slate-500 text-xs ml-3", children: "\uB3D9\uC601\uC0C1 \uB610\uB294 \uC624\uB514\uC624 \uD30C\uC77C" })] }), videos.length === 0 ? (_jsx("div", { className: "text-slate-500 text-sm bg-slate-700/30 rounded-lg p-4 text-center", children: "\uC5C5\uB85C\uB4DC\uB41C \uC601\uC0C1\uC774 \uC5C6\uC2B5\uB2C8\uB2E4. \uC704 \uBC84\uD2BC\uC73C\uB85C \uD30C\uC77C\uC744 \uC5C5\uB85C\uB4DC\uD574\uC8FC\uC138\uC694." })) : (_jsx("div", { className: "space-y-2", children: videos.map((v) => (_jsxs("div", { className: "flex items-center gap-4 bg-slate-700/40 hover:bg-slate-700/70 rounded-lg px-4 py-3 cursor-pointer transition-colors group", onClick: () => navigate(`/video/${v.id}`), children: [_jsx("span", { className: "font-mono text-xs text-slate-500 w-20 shrink-0", children: v.id }), _jsx("span", { className: "flex-1 text-sm truncate group-hover:text-white transition-colors", children: v.filename }), _jsx("span", { className: `px-2.5 py-0.5 rounded-full text-xs font-medium shrink-0 ${v.stage === "done"
                                                    ? "bg-green-900/60 text-green-300"
                                                    : v.stage.startsWith("processing")
                                                        ? "bg-yellow-900/60 text-yellow-300 animate-pulse"
                                                        : "bg-slate-600 text-slate-300"}`, children: stageLabel(v.stage) }), _jsx("button", { className: "text-slate-500 hover:text-red-400 text-xs shrink-0 transition-colors", onClick: (e) => {
                                                    e.stopPropagation();
                                                    handleDeleteVideo(v.id);
                                                }, children: "\uC0AD\uC81C" })] }, v.id))) }))] })) })] }), speakerSummary.some((s) => s.count > 0) && (_jsxs("section", { className: "mb-5 bg-slate-800/60 border border-slate-700 rounded-xl p-5", children: [_jsx("h2", { className: "text-lg font-semibold mb-1", children: "\uC218\uC9D1 \uD604\uD669" }), _jsx("p", { className: "text-slate-400 text-sm mb-1", children: "\uC804\uCC98\uB9AC\uAC00 \uC644\uB8CC\uB41C \uC601\uC0C1\uC5D0\uC11C \uC218\uC9D1\uB41C \uD654\uC790\uBCC4 \uC74C\uC131 \uB370\uC774\uD130\uC785\uB2C8\uB2E4." }), _jsxs("p", { className: "text-slate-500 text-xs mb-4", children: ["\uBE60\uB974\uAC8C \uD14C\uC2A4\uD2B8\uD558\uB824\uBA74 ", _jsx("span", { className: "text-slate-300", children: "\uC57D 10\uBD84" }), ", \uC548\uC815\uC801\uC778 \uD488\uC9C8\uC744 \uC6D0\uD558\uBA74 ", _jsx("span", { className: "text-slate-300", children: "20~45\uBD84" }), " \uC815\uB3C4\uC758 \uAE68\uB057\uD55C \uC74C\uC131 \uB370\uC774\uD130\uB97C \uC900\uBE44\uD558\uBA74 \uC88B\uC2B5\uB2C8\uB2E4."] }), _jsx("div", { className: "grid grid-cols-2 gap-3", children: speakerSummary
                            .filter((s) => s.count > 0)
                            .map((s) => (_jsxs("div", { className: "bg-slate-700/50 rounded-lg px-4 py-3", children: [_jsxs("div", { className: "flex justify-between items-center mb-1", children: [_jsx("span", { className: "text-slate-200 font-medium", children: s.name }), _jsxs("span", { className: "text-slate-400 text-sm font-mono", children: [s.count, "\uAC1C"] })] }), _jsxs("p", { className: "text-xs text-slate-500", children: ["\uCD1D ", formatDuration(s.total_duration)] })] }, s.name))) })] })), _jsxs("section", { className: `bg-slate-800/60 border border-slate-700 rounded-xl p-5 transition-opacity ${!hasDoneVideos ? "opacity-50" : ""}`, children: [_jsxs("div", { className: "flex items-center gap-2.5 mb-1", children: [_jsx("span", { className: `w-7 h-7 rounded-full text-white flex items-center justify-center text-xs font-bold shrink-0 ${hasDoneVideos ? "bg-blue-600" : "bg-slate-600"}`, children: "4" }), _jsx("h2", { className: "text-lg font-semibold", children: "\uB370\uC774\uD130\uC14B \uC0DD\uC131" })] }), _jsx("p", { className: "text-slate-400 text-sm mb-4 ml-10", children: "\uC218\uC9D1\uB41C \uC74C\uC131 \uB370\uC774\uD130\uB97C TTS \uD559\uC2B5\uC5D0 \uC0AC\uC6A9\uD560 \uC218 \uC788\uB294 \uB370\uC774\uD130\uC14B\uC73C\uB85C \uBCC0\uD658\uD569\uB2C8\uB2E4. \uBAA8\uB4E0 \uC601\uC0C1\uC758 \uC804\uCC98\uB9AC\uB97C \uC644\uB8CC\uD55C \uD6C4 \uC2E4\uD589\uD558\uC138\uC694." }), _jsx("div", { className: "ml-10", children: !hasDoneVideos ? (_jsx("div", { className: "text-slate-500 text-sm bg-slate-700/30 rounded-lg p-4 text-center", children: "\uC804\uCC98\uB9AC\uAC00 \uC644\uB8CC\uB41C \uC601\uC0C1\uC774 \uC5C6\uC2B5\uB2C8\uB2E4. \uBA3C\uC800 \uC601\uC0C1\uC744 \uC5C5\uB85C\uB4DC\uD558\uACE0 \uC804\uCC98\uB9AC\uB97C \uC9C4\uD589\uD574\uC8FC\uC138\uC694." })) : (_jsxs(_Fragment, { children: [_jsxs("div", { className: "bg-slate-700/30 rounded-lg p-4 mb-4", children: [_jsx("label", { className: "text-sm text-slate-300 block mb-1", children: "\uAC80\uC99D(Validation) \uB370\uC774\uD130 \uBE44\uC728" }), _jsx("p", { className: "text-xs text-slate-500 mb-3", children: "\uC804\uCCB4 \uB370\uC774\uD130 \uC911 \uD559\uC2B5 \uD488\uC9C8 \uAC80\uC99D\uC5D0 \uC0AC\uC6A9\uD560 \uBE44\uC728\uC785\uB2C8\uB2E4. \uC77C\uBC18\uC801\uC73C\uB85C 10~20%\uAC00 \uC801\uC808\uD569\uB2C8\uB2E4. \uB098\uBA38\uC9C0\uB294 \uD559\uC2B5(Training)\uC5D0 \uC0AC\uC6A9\uB429\uB2C8\uB2E4." }), _jsxs("div", { className: "flex items-center gap-3", children: [_jsx("input", { type: "range", min: "0.01", max: "0.5", step: "0.01", className: "flex-1 accent-blue-500", value: valRatio, onChange: (e) => setValRatio(+e.target.value) }), _jsxs("span", { className: "text-white font-mono text-sm w-12 text-right", children: [Math.round(valRatio * 100), "%"] })] })] }), _jsx("button", { className: "bg-green-600 hover:bg-green-700 px-6 py-2.5 rounded-lg font-medium disabled:opacity-50 transition-colors", onClick: handleBuildDataset, disabled: building, children: building ? "생성 중..." : "데이터셋 생성" }), _jsx("p", { className: "text-slate-500 text-xs mt-2", children: "\uC774\uBBF8 \uB370\uC774\uD130\uC14B\uC774 \uC788\uB294 \uACBD\uC6B0, \uAE30\uC874 \uB370\uC774\uD130\uC14B\uC744 \uC0AD\uC81C\uD558\uACE0 \uC0C8\uB85C \uC0DD\uC131\uD569\uB2C8\uB2E4." }), datasetResult && (_jsxs("div", { className: "bg-green-900/20 border border-green-800/50 rounded-lg p-5 mt-4", children: [_jsx("p", { className: "text-green-300 font-semibold mb-1", children: "\uB370\uC774\uD130\uC14B \uC0DD\uC131 \uC644\uB8CC" }), _jsx("p", { className: "text-slate-500 text-xs mb-4 font-mono", children: datasetResult.dataset_dir }), Object.entries(datasetResult.speakers).map(([name, s]) => (_jsxs("div", { className: "mb-3 last:mb-0", children: [_jsx("p", { className: "text-slate-200 font-semibold mb-1", children: name }), _jsxs("div", { className: "grid grid-cols-3 gap-2 text-xs text-slate-400 bg-slate-700/50 rounded-lg p-3", children: [_jsxs("div", { children: [_jsx("p", { className: "text-slate-500 mb-0.5", children: "\uC804\uCCB4" }), _jsxs("p", { children: [s.count, "\uAC1C / ", formatDuration(s.duration)] })] }), _jsxs("div", { children: [_jsx("p", { className: "text-slate-500 mb-0.5", children: "Train" }), _jsxs("p", { children: [s.train_count, "\uAC1C / ", formatDuration(s.train_duration)] })] }), _jsxs("div", { children: [_jsx("p", { className: "text-slate-500 mb-0.5", children: "Val" }), _jsxs("p", { children: [s.val_count, "\uAC1C / ", formatDuration(s.val_duration)] })] })] })] }, name)))] }))] })) })] })] }));
}
