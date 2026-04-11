import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Fragment, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router";
import * as videosApi from "../api/videos";
import { usePolling } from "../hooks/usePolling";
import ProgressBar from "../components/ProgressBar";
import CardClassifier from "../components/CardClassifier";
import ReviewEditor from "../components/ReviewEditor";
const STAGE_ORDER = ["extract", "separate", "vad", "classify", "transcribe", "review"];
const STAGE_LABELS = {
    extract: "추출",
    separate: "배경음 제거",
    vad: "VAD 세그먼팅",
    classify: "분류",
    transcribe: "전사",
    review: "검토",
};
const STAGE_DESCRIPTIONS = {
    extract: {
        title: "오디오 추출",
        desc: "영상 파일에서 오디오 트랙을 분리합니다. 오디오 파일인 경우 포맷을 변환합니다.",
    },
    separate: {
        title: "배경음 제거",
        desc: "음악, 효과음 등 배경 소리를 제거하고 사람 목소리(보컬)만 추출합니다. 파일 길이에 따라 수 분이 소요될 수 있습니다.",
    },
    vad: {
        title: "음성 구간 분할 (VAD)",
        desc: "음성 활동을 자동으로 감지하여 개별 대사 단위로 분할합니다. 아래 파라미터를 소스에 맞게 조정하면 분할 품질이 높아집니다.",
    },
    classify: {
        title: "화자 분류",
        desc: "분할된 각 음성 구간을 듣고 해당 화자에게 배정합니다. 잡음이나 불필요한 구간은 버릴 수 있습니다.",
    },
    transcribe: {
        title: "음성 전사 (STT)",
        desc: "각 음성 구간의 내용을 텍스트로 자동 변환합니다. 변환 결과는 다음 단계에서 직접 검토/수정할 수 있습니다.",
    },
    review: {
        title: "전사 검토",
        desc: "자동 전사 결과를 확인하고 오류를 수정합니다. 정확한 텍스트가 TTS 학습 품질에 직접 영향을 미칩니다.",
    },
};
const VAD_PRESETS = [
    {
        label: "기본값",
        desc: "일반적인 대화 영상",
        params: { min_segment_sec: 1.0, max_segment_sec: 8.0, threshold: 0.3, min_silence_ms: 50 },
    },
    {
        label: "시끄러운 환경",
        desc: "배경 음악/잡음이 많은 경우",
        params: { min_segment_sec: 1.5, max_segment_sec: 8.0, threshold: 0.65, min_silence_ms: 40 },
    },
    {
        label: "긴 독백",
        desc: "나레이션/강의 영상",
        params: { min_segment_sec: 1.0, max_segment_sec: 12.0, threshold: 0.3, min_silence_ms: 80 },
    },
];
// 각 단계에서 "다음 실행" 버튼에 쓸 API 함수 (vad는 파라미터 필요하므로 별도 처리)
const STAGE_ACTIONS = {
    extract: videosApi.startExtract,
    separate: videosApi.startSeparate,
    transcribe: videosApi.startTranscription,
};
const DEFAULT_VAD_PARAMS = {
    min_segment_sec: 1.0,
    max_segment_sec: 8.0,
    threshold: 0.3,
    min_silence_ms: 50,
};
function stageIndex(stage) {
    if (stage.startsWith("processing:")) {
        return STAGE_ORDER.indexOf(stage.split(":")[1]);
    }
    if (stage === "classifying")
        return STAGE_ORDER.indexOf("classify");
    if (stage === "done")
        return STAGE_ORDER.length;
    return STAGE_ORDER.indexOf(stage);
}
export default function VideoDetail() {
    const { videoId } = useParams();
    const navigate = useNavigate();
    const [status, setStatus] = useState(null);
    const [error, setError] = useState("");
    const [vadParams, setVadParams] = useState({ ...DEFAULT_VAD_PARAMS });
    const isProcessing = status?.stage.startsWith("processing:") ?? false;
    usePolling(async () => {
        if (!videoId)
            return;
        try {
            setStatus(await videosApi.getStatus(videoId));
        }
        catch {
            setError("상태 조회 실패");
        }
    }, 1500, isProcessing);
    useEffect(() => {
        if (!videoId)
            return;
        videosApi
            .getStatus(videoId)
            .then(setStatus)
            .catch(() => setError("영상을 찾을 수 없습니다."));
    }, [videoId]);
    const refreshStatus = async () => {
        if (!videoId)
            return;
        setStatus(await videosApi.getStatus(videoId));
    };
    const handleRollback = async (stage) => {
        if (!videoId)
            return;
        if (!confirm(`"${STAGE_LABELS[stage]}" 단계부터 재처리하시겠습니까?\n이후 단계 데이터가 모두 삭제됩니다.`))
            return;
        await videosApi.rollbackVideo(videoId, stage);
        await refreshStatus();
    };
    const [stageError, setStageError] = useState("");
    const [pendingStage, setPendingStage] = useState(null);
    const handleRunStage = async (stage) => {
        if (!videoId)
            return;
        const action = STAGE_ACTIONS[stage];
        if (!action)
            return;
        setStageError("");
        setPendingStage(null);
        try {
            await action(videoId);
            setTimeout(refreshStatus, 500);
        }
        catch (e) {
            const resp = e?.response;
            if (resp?.status === 409) {
                setPendingStage(stage);
            }
            else {
                setStageError(resp?.data?.detail || "실행에 실패했습니다.");
            }
        }
    };
    // 409 자동 재시도
    useEffect(() => {
        if (!pendingStage || !videoId)
            return;
        const action = STAGE_ACTIONS[pendingStage];
        if (!action)
            return;
        const interval = setInterval(async () => {
            try {
                await action(videoId);
                setPendingStage(null);
                videosApi.getStatus(videoId).then(setStatus);
            }
            catch {
                // 아직 실행 중 — 계속 대기
            }
        }, 5000);
        return () => clearInterval(interval);
    }, [pendingStage, videoId]);
    if (error) {
        return (_jsxs("div", { className: "max-w-4xl mx-auto p-6", children: [_jsx("p", { className: "text-red-400", children: error }), _jsx("button", { className: "text-blue-400 mt-2", onClick: () => navigate("/"), children: "\uB3CC\uC544\uAC00\uAE30" })] }));
    }
    if (!status) {
        return _jsx("div", { className: "max-w-4xl mx-auto p-6", children: "\uB85C\uB529 \uC911..." });
    }
    const stage = status.stage;
    const currentIdx = stageIndex(stage);
    const allStages = [...STAGE_ORDER, "done"];
    return (_jsxs("div", { className: "max-w-4xl mx-auto p-6 pb-16", children: [_jsxs("div", { className: "flex items-center gap-4 mb-8", children: [_jsx("button", { className: "bg-slate-700 hover:bg-slate-600 text-slate-300 hover:text-white px-3 py-1.5 rounded-lg text-sm transition-colors", onClick: () => navigate("/"), children: "\u2190 \uBAA9\uB85D" }), _jsxs("div", { children: [_jsx("h1", { className: "text-xl font-bold", children: status.filename }), _jsxs("span", { className: "text-slate-500 text-xs font-mono", children: ["#", videoId] })] })] }), _jsx("div", { className: "mb-2", children: _jsx("div", { className: "flex items-start", children: allStages.map((s, i) => {
                        const isDone = s === "done";
                        const stageIdx = isDone ? STAGE_ORDER.length : i;
                        const isCompleted = isDone ? stage === "done" : stageIdx < currentIdx;
                        const isCurrent = stageIdx === currentIdx && stage !== "done";
                        const isClickable = !isDone && isCompleted;
                        return (_jsxs(Fragment, { children: [i > 0 && (_jsx("div", { className: `flex-1 h-0.5 mt-4 transition-colors ${stageIdx <= currentIdx || (isDone && stage === "done")
                                        ? "bg-green-600"
                                        : "bg-slate-700"}` })), _jsxs("div", { className: `flex flex-col items-center ${isClickable ? "cursor-pointer group" : ""}`, onClick: () => isClickable && handleRollback(s), title: isClickable ? `"${STAGE_LABELS[s]}" 단계부터 재처리` : "", children: [_jsx("div", { className: `w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 transition-colors ${isCompleted
                                                ? "bg-green-600 text-white group-hover:bg-green-500"
                                                : isCurrent
                                                    ? "bg-blue-600 text-white ring-2 ring-blue-400/50"
                                                    : "bg-slate-700 text-slate-500"}`, children: isCompleted ? "\u2713" : isDone ? "" : i + 1 }), _jsx("span", { className: `text-[10px] mt-1.5 whitespace-nowrap transition-colors ${isCompleted
                                                ? "text-green-400 group-hover:text-green-300"
                                                : isCurrent
                                                    ? "text-blue-300 font-semibold"
                                                    : "text-slate-500"}`, children: isDone ? "완료" : STAGE_LABELS[s] })] })] }, s));
                    }) }) }), currentIdx > 0 && stage !== "done" && (_jsx("p", { className: "text-slate-600 text-[11px] text-center mb-6", children: "\uCD08\uB85D\uC0C9 \uB2E8\uACC4\uB97C \uD074\uB9AD\uD558\uBA74 \uD574\uB2F9 \uB2E8\uACC4\uBD80\uD130 \uC7AC\uCC98\uB9AC\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4" })), (currentIdx === 0 || stage === "done") && _jsx("div", { className: "mb-6" }), status.error && (_jsxs("div", { className: "bg-red-900/30 border border-red-700 rounded-xl p-6 mb-6", children: [_jsxs("p", { className: "text-red-300 font-semibold mb-2", children: [STAGE_LABELS[status.error.stage] || status.error.stage, " \uB2E8\uACC4\uC5D0\uC11C \uC624\uB958 \uBC1C\uC0DD"] }), _jsx("pre", { className: "text-red-400 text-sm whitespace-pre-wrap break-words bg-red-950/50 rounded-lg p-3", children: status.error.message }), _jsx("p", { className: "text-slate-500 text-xs mt-3", children: "\uC544\uB798 \uBC84\uD2BC\uC73C\uB85C \uC7AC\uC2E4\uD589\uD558\uAC70\uB098, \uC704 \uC2A4\uD14C\uD37C\uC5D0\uC11C \uC774\uC804 \uB2E8\uACC4\uB97C \uD074\uB9AD\uD558\uC5EC \uB864\uBC31\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4." })] })), isProcessing && status.processing && (_jsxs("div", { className: "bg-slate-800 rounded-xl p-6 mb-6", children: [_jsx("p", { className: "text-slate-300 text-sm font-medium mb-3", children: STAGE_DESCRIPTIONS[status.processing.stage]?.title || STAGE_LABELS[status.processing.stage] || "처리 중" }), _jsx(ProgressBar, { progress: status.processing.progress, message: status.processing.message }), status.processing.stage === "separate" && (_jsx("p", { className: "text-slate-500 text-xs mt-3", children: "\uBC30\uACBD\uC74C \uC81C\uAC70\uB294 \uC74C\uC6D0 \uAE38\uC774\uC5D0 \uB530\uB77C \uC218 \uBD84~\uC218\uC2ED \uBD84 \uC18C\uC694\uB420 \uC218 \uC788\uC2B5\uB2C8\uB2E4. \uC774 \uD398\uC774\uC9C0\uB97C \uBC97\uC5B4\uB098\uB3C4 \uCC98\uB9AC\uB294 \uACC4\uC18D\uB429\uB2C8\uB2E4." }))] })), !isProcessing &&
                stage in STAGE_ACTIONS &&
                stage !== "classify" &&
                stage !== "classifying" && (_jsxs("div", { className: "bg-slate-800 rounded-xl p-6", children: [_jsxs("div", { className: "text-center mb-5", children: [_jsx("p", { className: "text-blue-300 font-semibold text-lg mb-2", children: STAGE_DESCRIPTIONS[stage]?.title || STAGE_LABELS[stage] }), _jsx("p", { className: "text-slate-400 text-sm max-w-md mx-auto", children: STAGE_DESCRIPTIONS[stage]?.desc })] }), pendingStage && (_jsxs("div", { className: "bg-yellow-900/30 border border-yellow-700/50 rounded-lg p-4 mb-4 text-center", children: [_jsx("p", { className: "text-yellow-300 text-sm mb-1", children: "\uB2E4\uB978 \uC601\uC0C1\uC758 \uBC30\uACBD\uC74C \uC81C\uAC70\uAC00 \uC9C4\uD589 \uC911\uC785\uB2C8\uB2E4" }), _jsx("p", { className: "text-slate-400 text-xs", children: "\uC644\uB8CC\uB418\uBA74 \uC790\uB3D9\uC73C\uB85C \uC2DC\uC791\uB429\uB2C8\uB2E4. \uC774 \uD398\uC774\uC9C0\uC5D0\uC11C \uAE30\uB2E4\uB824\uC8FC\uC138\uC694." })] })), stageError && !pendingStage && (_jsx("div", { className: "bg-red-900/30 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm text-center", children: stageError })), _jsx("div", { className: "text-center", children: _jsx("button", { className: `px-8 py-2.5 rounded-lg font-medium transition-colors ${pendingStage
                                ? "bg-yellow-600/50 text-yellow-200 cursor-wait animate-pulse"
                                : "bg-blue-600 hover:bg-blue-700"}`, onClick: () => !pendingStage && handleRunStage(stage), disabled: !!pendingStage, children: pendingStage ? "대기 중..." : `${STAGE_LABELS[stage]} 실행` }) })] })), !isProcessing && stage === "vad" && (_jsxs("div", { className: "bg-slate-800 rounded-xl p-6", children: [_jsxs("div", { className: "text-center mb-6", children: [_jsx("p", { className: "text-blue-300 font-semibold text-lg mb-2", children: STAGE_DESCRIPTIONS.vad.title }), _jsx("p", { className: "text-slate-400 text-sm max-w-lg mx-auto", children: STAGE_DESCRIPTIONS.vad.desc })] }), _jsxs("div", { className: "max-w-lg mx-auto mb-6", children: [_jsx("p", { className: "text-slate-300 text-sm font-medium mb-2", children: "\uBE60\uB978 \uC124\uC815" }), _jsx("div", { className: "grid grid-cols-3 gap-2", children: VAD_PRESETS.map((preset) => (_jsxs("button", { className: "bg-slate-700/60 hover:bg-slate-700 border border-slate-600 hover:border-blue-500/50 rounded-lg px-3 py-2.5 text-left transition-colors", onClick: () => setVadParams({ ...preset.params }), children: [_jsx("p", { className: "text-slate-200 text-xs font-medium", children: preset.label }), _jsx("p", { className: "text-slate-500 text-[10px] mt-0.5", children: preset.desc })] }, preset.label))) })] }), _jsxs("div", { className: "space-y-4 mb-6 max-w-lg mx-auto text-sm", children: [_jsxs("label", { className: "block text-slate-400", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("span", { children: "\uC138\uADF8\uBA3C\uD2B8 \uCD5C\uC18C \uAE38\uC774 (\uCD08)" }), _jsx("input", { type: "number", step: "0.1", className: "w-24 bg-slate-700 rounded-lg px-3 py-1.5 text-white text-right focus:outline-none focus:ring-1 focus:ring-blue-500", value: vadParams.min_segment_sec, onChange: (e) => setVadParams({ ...vadParams, min_segment_sec: +e.target.value }) })] }), _jsx("p", { className: "text-slate-500 text-xs mt-1", children: "\uC774\uBCF4\uB2E4 \uC9E7\uC740 \uB300\uC0AC\uB294 \uBC84\uB9BD\uB2C8\uB2E4. \uB108\uBB34 \uC9E7\uC740 \uC74C\uC131\uC740 \uD559\uC2B5\uC5D0 \uBD80\uC801\uD569\uD558\uBBC0\uB85C 1~2\uCD08\uB97C \uAD8C\uC7A5\uD569\uB2C8\uB2E4." })] }), _jsxs("label", { className: "block text-slate-400", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("span", { children: "\uC138\uADF8\uBA3C\uD2B8 \uCD5C\uB300 \uAE38\uC774 (\uCD08)" }), _jsx("input", { type: "number", step: "0.5", className: "w-24 bg-slate-700 rounded-lg px-3 py-1.5 text-white text-right focus:outline-none focus:ring-1 focus:ring-blue-500", value: vadParams.max_segment_sec, onChange: (e) => setVadParams({ ...vadParams, max_segment_sec: +e.target.value }) })] }), _jsx("p", { className: "text-slate-500 text-xs mt-1", children: "\uC774\uBCF4\uB2E4 \uAE34 \uB300\uC0AC\uB294 \uC790\uB3D9\uC73C\uB85C \uBD84\uD560\uB429\uB2C8\uB2E4. TTS \uD559\uC2B5\uC5D0\uB294 5~15\uCD08\uAC00 \uC801\uD569\uD569\uB2C8\uB2E4." })] }), _jsxs("label", { className: "block text-slate-400", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("span", { children: "\uC74C\uC131 \uAC10\uC9C0 \uC784\uACC4\uAC12" }), _jsx("input", { type: "number", step: "0.05", min: "0", max: "1", className: "w-24 bg-slate-700 rounded-lg px-3 py-1.5 text-white text-right focus:outline-none focus:ring-1 focus:ring-blue-500", value: vadParams.threshold, onChange: (e) => setVadParams({ ...vadParams, threshold: +e.target.value }) })] }), _jsx("p", { className: "text-slate-500 text-xs mt-1", children: "\uB0AE\uCD94\uBA74 \uB354 \uB9CE\uC740 \uB300\uC0AC\uB97C \uC7A1\uC544\uB0B4\uACE0, \uC62C\uB9AC\uBA74 \uD655\uC2E4\uD55C \uC74C\uC131\uB9CC \uB0A8\uAE41\uB2C8\uB2E4. \uBC30\uACBD \uC7A1\uC74C\uC774 \uB9CE\uC73C\uBA74 0.6~0.7\uB85C \uC62C\uB9AC\uACE0, \uC870\uC6A9\uD55C \uD658\uACBD\uC774\uBA74 0.3~0.5\uB85C \uB0AE\uCD94\uC138\uC694." })] }), _jsxs("label", { className: "block text-slate-400", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("span", { children: "\uB300\uC0AC \uAC04 \uCD5C\uC18C \uBB34\uC74C (ms)" }), _jsx("input", { type: "number", step: "10", className: "w-24 bg-slate-700 rounded-lg px-3 py-1.5 text-white text-right focus:outline-none focus:ring-1 focus:ring-blue-500", value: vadParams.min_silence_ms, onChange: (e) => setVadParams({ ...vadParams, min_silence_ms: +e.target.value }) })] }), _jsx("p", { className: "text-slate-500 text-xs mt-1", children: "\uB300\uC0AC\uC640 \uB300\uC0AC \uC0AC\uC774\uC5D0 \uC774 \uAE38\uC774 \uC774\uC0C1\uC758 \uBB34\uC74C\uC774 \uC788\uC5B4\uC57C \uBCC4\uB3C4 \uC138\uADF8\uBA3C\uD2B8\uB85C \uBD84\uB9AC\uD569\uB2C8\uB2E4. \uAC12\uC774 \uB108\uBB34 \uC791\uC73C\uBA74 \uD55C \uBB38\uC7A5\uC774 \uC5EC\uB7EC \uC870\uAC01\uC73C\uB85C \uCABC\uAC1C\uC9C0\uACE0, \uB108\uBB34 \uD06C\uBA74 \uC11C\uB85C \uB2E4\uB978 \uB300\uC0AC\uAC00 \uD558\uB098\uB85C \uD569\uCCD0\uC9D1\uB2C8\uB2E4." })] })] }), _jsxs("details", { className: "max-w-lg mx-auto mb-6", children: [_jsx("summary", { className: "text-slate-400 text-xs cursor-pointer hover:text-slate-300 transition-colors", children: "\uC0C1\uD669\uBCC4 \uC870\uC815 \uAC00\uC774\uB4DC \uBCF4\uAE30" }), _jsxs("div", { className: "bg-slate-700/40 rounded-lg p-4 mt-2 text-xs text-slate-400 space-y-1.5", children: [_jsxs("p", { children: [_jsx("span", { className: "text-blue-400", children: "\uB450 \uD654\uC790\uAC00 \uBE60\uB974\uAC8C \uB300\uD654\uD558\uB294 \uC601\uC0C1" }), " \u2192 최소 무음을 30~50ms로 낮추면 대사가 더 잘 분리됩니다."] }), _jsxs("p", { children: [_jsx("span", { className: "text-blue-400", children: "\uBC30\uACBD \uC74C\uC545/\uC7A1\uC74C\uC774 \uB0A8\uC544\uC788\uB294 \uACBD\uC6B0" }), " \u2192 감지 임계값을 0.6~0.7로 올려 잡음을 걸러내세요."] }), _jsxs("p", { children: [_jsx("span", { className: "text-blue-400", children: "\uAE34 \uB3C5\uBC31\uC774 \uB9CE\uC740 \uC601\uC0C1" }), " \u2192 최대 길이를 10~15초로 설정하면 자연스러운 분할이 됩니다."] }), _jsxs("p", { children: [_jsx("span", { className: "text-blue-400", children: "\uC9E7\uC740 \uAC10\uD0C4\uC0AC/\uCD94\uC784\uC0C8\uAC00 \uB9CE\uC740 \uACBD\uC6B0" }), " \u2192 최소 길이를 1.5~2초로 올려 불필요한 세그먼트를 줄이세요."] })] })] }), _jsxs("div", { className: "text-center", children: [_jsx("button", { className: "bg-blue-600 hover:bg-blue-700 px-8 py-2.5 rounded-lg font-medium transition-colors", onClick: async () => {
                                    if (!videoId)
                                        return;
                                    await videosApi.startVad(videoId, vadParams);
                                    setTimeout(refreshStatus, 500);
                                }, children: "VAD \uC138\uADF8\uBA3C\uD305 \uC2E4\uD589" }), _jsx("p", { className: "text-slate-500 text-xs mt-3", children: "\uC2E4\uD589 \uD6C4 \uC5D0\uB7EC \uBA54\uC2DC\uC9C0\uAC00 \uB098\uD0C0\uB098\uC9C0 \uC54A\uC73C\uBA74 \uC815\uC0C1\uC801\uC73C\uB85C \uC9C4\uD589 \uC911\uC785\uB2C8\uB2E4. \uC7A0\uC2DC \uAE30\uB2E4\uB824\uC8FC\uC138\uC694." })] })] })), (stage === "classify" || stage === "classifying") && videoId && (_jsx(CardClassifier, { videoId: videoId, sourceFile: status.source_file, onDone: refreshStatus })), stage === "review" && videoId && (_jsx(ReviewEditor, { videoId: videoId, onDone: refreshStatus })), stage === "done" && (_jsxs("div", { className: "bg-gradient-to-br from-green-900/30 to-slate-800/30 border border-green-700/50 rounded-xl p-8 text-center", children: [_jsx("div", { className: "w-14 h-14 rounded-full bg-green-600 flex items-center justify-center text-xl mx-auto mb-4", children: "\u2713" }), _jsx("p", { className: "text-green-300 text-xl font-semibold mb-2", children: "\uC804\uCC98\uB9AC \uC644\uB8CC" }), _jsx("p", { className: "text-slate-400 text-sm mb-6", children: "\uC774 \uC601\uC0C1\uC758 \uBAA8\uB4E0 \uC804\uCC98\uB9AC \uB2E8\uACC4\uAC00 \uC644\uB8CC\uB418\uC5C8\uC2B5\uB2C8\uB2E4." }), status.summary && status.summary.length > 0 && (_jsx("div", { className: "max-w-sm mx-auto space-y-2 mb-6", children: status.summary.map((s) => (_jsxs("div", { className: "flex justify-between items-center bg-green-900/20 rounded-lg px-4 py-2.5 text-sm", children: [_jsx("span", { className: s.name === "discarded" ? "text-slate-500" : "text-slate-200", children: s.name === "discarded" ? "버림" : s.name }), _jsxs("span", { className: "text-slate-400 font-mono text-xs", children: [s.count, "\uAC1C / ", Math.floor(s.total_duration / 60), "\uBD84 ", Math.round(s.total_duration % 60), "\uCD08"] })] }, s.name))) })), _jsx("p", { className: "text-slate-500 text-sm", children: "\uB300\uC2DC\uBCF4\uB4DC\uB85C \uB3CC\uC544\uAC00 \uB370\uC774\uD130\uC14B\uC744 \uC0DD\uC131\uD558\uAC70\uB098, \uCD94\uAC00 \uC601\uC0C1\uC744 \uC5C5\uB85C\uB4DC\uD558\uC5EC \uB370\uC774\uD130\uB97C \uB354 \uC218\uC9D1\uD558\uC138\uC694." })] }))] }));
}
