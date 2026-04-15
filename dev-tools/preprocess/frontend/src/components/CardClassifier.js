import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback, useEffect, useRef, useState } from "react";
import * as clsApi from "../api/classification";
import * as speakersApi from "../api/speakers";
const BUFFER_SIZE = 10;
export default function CardClassifier({ videoId, sourceFile, onDone }) {
    const [speakers, setSpeakers] = useState([]);
    const [segments, setSegments] = useState([]);
    const [currentIdx, setCurrentIdx] = useState(0);
    const [totalUnclassified, setTotalUnclassified] = useState(0);
    const [classified, setClassified] = useState(0);
    const [totalAll, setTotalAll] = useState(0);
    const [loading, setLoading] = useState(true);
    const [bucketCounts, setBucketCounts] = useState([]);
    const audioRef = useRef(null);
    const videoRef = useRef(null);
    // 버킷 카운트 새로고침
    const refreshBuckets = useCallback(async () => {
        const state = await clsApi.getClassification(videoId);
        setBucketCounts(state.speakers);
    }, [videoId]);
    // 초기 로드
    useEffect(() => {
        (async () => {
            const [spk, data] = await Promise.all([
                speakersApi.listSpeakers(),
                clsApi.getUnclassified(videoId, 0, BUFFER_SIZE),
            ]);
            setSpeakers(spk);
            setSegments(data.segments);
            setTotalUnclassified(data.total);
            setClassified(data.classified);
            setTotalAll(data.total_all);
            setLoading(false);
            await refreshBuckets();
        })();
    }, [videoId, refreshBuckets]);
    // 버퍼 보충
    const refillBuffer = useCallback(async () => {
        const data = await clsApi.getUnclassified(videoId, 0, BUFFER_SIZE);
        setSegments(data.segments);
        setCurrentIdx(0);
        setTotalUnclassified(data.total);
        setClassified(data.classified);
        setTotalAll(data.total_all);
        await refreshBuckets();
    }, [videoId, refreshBuckets]);
    const current = segments[currentIdx] || null;
    // 오디오 URL
    const audioUrl = current
        ? `/api/media/videos/${videoId}/segments/unclassified/${current.file}`
        : "";
    // 비디오 URL (실제 소스 파일명 사용)
    const videoUrl = sourceFile
        ? `/api/media/videos/${videoId}/${sourceFile}`
        : "";
    // 영상+오디오 동기 재생
    const playSegment = useCallback(() => {
        if (!audioRef.current || !videoRef.current || !current)
            return;
        videoRef.current.currentTime = current.start;
        videoRef.current.play().catch(() => { });
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => { });
    }, [current]);
    // 오디오 끝나면 영상도 정지
    useEffect(() => {
        const audio = audioRef.current;
        const video = videoRef.current;
        if (!audio || !video)
            return;
        const handleEnded = () => {
            video.pause();
        };
        audio.addEventListener("ended", handleEnded);
        return () => audio.removeEventListener("ended", handleEnded);
    }, [audioUrl]);
    // 분류 처리
    const handleClassify = useCallback(async (speaker) => {
        if (!current)
            return;
        await clsApi.classifySegment(videoId, current.file, speaker);
        // 다음 세그먼트로
        const nextIdx = currentIdx + 1;
        if (nextIdx >= segments.length) {
            await refillBuffer();
        }
        else {
            setCurrentIdx(nextIdx);
            setClassified((c) => c + 1);
            setTotalUnclassified((t) => t - 1);
            // 버킷 카운트 비동기 갱신
            refreshBuckets();
        }
    }, [current, currentIdx, segments.length, videoId, refillBuffer, refreshBuckets]);
    // Undo
    const handleUndo = useCallback(async () => {
        await clsApi.undoClassification(videoId);
        await refillBuffer();
    }, [videoId, refillBuffer]);
    // 분류 완료
    const handleDone = async () => {
        if (!confirm("분류를 완료하시겠습니까? 미분류 세그먼트가 남아있어도 진행됩니다."))
            return;
        await clsApi.markDone(videoId);
        onDone();
    };
    // 키보드 단축키
    useEffect(() => {
        const handler = (e) => {
            if (e.target instanceof HTMLInputElement)
                return;
            // 1~9: 화자 배정
            if (e.key >= "1" && e.key <= "9") {
                const idx = parseInt(e.key) - 1;
                if (idx < speakers.length)
                    handleClassify(speakers[idx]);
                return;
            }
            switch (e.key.toLowerCase()) {
                case "d":
                    handleClassify("discarded");
                    break;
                case "z":
                    handleUndo();
                    break;
                case "r":
                    playSegment();
                    break;
            }
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [speakers, handleClassify, handleUndo, playSegment]);
    // 새 세그먼트 로드 시 자동 재생
    useEffect(() => {
        if (audioUrl && current) {
            // 약간의 딜레이로 오디오/비디오 로드 대기
            const timer = setTimeout(playSegment, 100);
            return () => clearTimeout(timer);
        }
    }, [audioUrl, current, playSegment]);
    if (loading)
        return _jsx("div", { className: "p-6 text-fg-muted", children: "\uB85C\uB529 \uC911..." });
    if (totalUnclassified === 0 && segments.length === 0) {
        return (_jsxs("div", { className: "bg-surface border border-line rounded-xl p-8 text-center", children: [_jsx("div", { className: "w-12 h-12 rounded-full bg-success/15 border border-success/40 text-success flex items-center justify-center text-lg mx-auto mb-3", children: "\u2713" }), _jsx("p", { className: "font-display text-xl font-bold text-success mb-2", children: "\uBAA8\uB4E0 \uC138\uADF8\uBA3C\uD2B8 \uBD84\uB958 \uC644\uB8CC" }), _jsx("p", { className: "text-fg-muted text-sm mb-5", children: "\uBAA8\uB4E0 \uC74C\uC131 \uAD6C\uAC04\uC774 \uD654\uC790\uC5D0\uAC8C \uBC30\uC815\uB418\uC5C8\uC2B5\uB2C8\uB2E4." }), _jsx("button", { className: "bg-primary hover:bg-primary-hover text-canvas px-6 py-2.5 rounded-lg font-semibold text-sm transition-colors", onClick: handleDone, children: "\uBD84\uB958 \uC644\uB8CC \u2192 \uB2E4\uC74C \uB2E8\uACC4" })] }));
    }
    // 버킷별 카운트 맵
    const countMap = new Map(bucketCounts.map((b) => [b.name, b.count]));
    const progressPct = totalAll > 0 ? Math.round((classified / totalAll) * 100) : 0;
    return (_jsxs("div", { className: "bg-surface border border-line rounded-xl p-6", children: [_jsxs("div", { className: "bg-primary/[0.08] border border-primary/25 rounded-lg p-4 mb-5", children: [_jsx("p", { className: "text-primary text-[11px] font-bold uppercase tracking-[1.5px] mb-1.5 font-display", children: "\uD654\uC790 \uBD84\uB958" }), _jsx("p", { className: "text-fg-muted text-xs leading-relaxed mb-3", children: "\uC624\uB514\uC624\uAC00 \uC790\uB3D9 \uC7AC\uC0DD\uB429\uB2C8\uB2E4. \uB4E4\uB9AC\uB294 \uBAA9\uC18C\uB9AC\uC5D0 \uD574\uB2F9\uD558\uB294 \uD654\uC790 \uBC84\uD2BC\uC744 \uD074\uB9AD\uD558\uAC70\uB098 \uD0A4\uBCF4\uB4DC \uB2E8\uCD95\uD0A4\uB97C \uC0AC\uC6A9\uD558\uC138\uC694. \uC7A1\uC74C, \uC74C\uC545, \uB610\uB294 \uB4F1\uB85D\uD558\uC9C0 \uC54A\uC740 \uC0AC\uB78C\uC758 \uBAA9\uC18C\uB9AC\uB294 \u201C\uBC84\uB9AC\uAE30\u201D\uB85C \uC81C\uC678\uD560 \uC218 \uC788\uC2B5\uB2C8\uB2E4." }), _jsxs("div", { className: "flex flex-wrap gap-x-5 gap-y-1 text-xs text-fg-dim", children: [_jsxs("span", { children: [_jsx("kbd", { className: "bg-canvas border border-line px-1.5 py-0.5 rounded text-fg font-mono text-[11px]", children: "1-9" }), " \uD654\uC790 \uBC30\uC815"] }), _jsxs("span", { children: [_jsx("kbd", { className: "bg-canvas border border-line px-1.5 py-0.5 rounded text-fg font-mono text-[11px]", children: "D" }), " \uBC84\uB9AC\uAE30"] }), _jsxs("span", { children: [_jsx("kbd", { className: "bg-canvas border border-line px-1.5 py-0.5 rounded text-fg font-mono text-[11px]", children: "R" }), " \uB2E4\uC2DC \uB4E3\uAE30"] }), _jsxs("span", { children: [_jsx("kbd", { className: "bg-canvas border border-line px-1.5 py-0.5 rounded text-fg font-mono text-[11px]", children: "Z" }), " \uB418\uB3CC\uB9AC\uAE30"] })] })] }), _jsxs("div", { className: "mb-5", children: [_jsxs("div", { className: "flex justify-between text-sm mb-1.5", children: [_jsx("span", { className: "text-fg-muted", children: "\uBD84\uB958 \uC9C4\uD589\uB960" }), _jsxs("span", { className: "text-primary font-mono font-semibold", children: [classified, " / ", totalAll, " (", progressPct, "%)"] })] }), _jsx("div", { className: "w-full bg-line rounded-full h-1.5 overflow-hidden", children: _jsx("div", { className: "progress-fill h-1.5 rounded-full transition-all duration-300", style: { width: `${progressPct}%` } }) }), _jsxs("p", { className: "text-xs text-fg-dim mt-1.5", children: ["\uB0A8\uC740 \uC138\uADF8\uBA3C\uD2B8: ", totalUnclassified, "\uAC1C"] })] }), videoUrl && (_jsx("div", { className: "mb-4", children: _jsx("video", { ref: videoRef, src: videoUrl, className: "w-full max-h-96 bg-canvas border border-line rounded-lg", muted: true }) })), current && (_jsxs("div", { className: "mb-5", children: [_jsxs("div", { className: "flex justify-between text-sm text-fg-muted mb-2 bg-canvas border border-line rounded-lg px-3 py-2", children: [_jsx("span", { className: "font-mono text-xs text-fg-dim", children: current.file }), _jsxs("span", { className: "text-xs text-fg", children: [current.start.toFixed(1), "s ~ ", current.end.toFixed(1), "s", _jsxs("span", { className: "text-fg-dim ml-1", children: ["(", current.duration.toFixed(1), "s)"] })] })] }), _jsx("audio", { ref: audioRef, src: audioUrl })] })), _jsxs("div", { className: "mb-4", children: [_jsx("p", { className: "text-xs text-fg-dim mb-2", children: "\uD654\uC790 \uBC30\uC815 (\uC22B\uC790 \uD0A4\uB85C\uB3C4 \uC120\uD0DD \uAC00\uB2A5)" }), _jsx("div", { className: "grid grid-cols-3 gap-2", children: speakers.map((s, i) => (_jsxs("button", { className: "bg-canvas border border-line hover:border-primary/50 hover:bg-primary/[0.08] px-3 py-2.5 rounded-lg text-sm transition-colors text-left group", onClick: () => handleClassify(s), children: [_jsxs("span", { className: "text-fg-dim group-hover:text-primary mr-1.5 font-mono text-xs", children: [i + 1, "."] }), _jsx("span", { className: "text-fg group-hover:text-primary font-semibold", children: s }), countMap.has(s) && (_jsxs("span", { className: "text-fg-dim ml-1.5 text-xs", children: ["(", countMap.get(s), ")"] }))] }, s))) })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("button", { className: "bg-canvas border border-line hover:border-error/40 hover:text-error text-fg-muted px-3 py-2 rounded-lg text-sm transition-colors", onClick: () => handleClassify("discarded"), title: "\uC7A1\uC74C, \uC74C\uC545, \uBD88\uD544\uC694\uD55C \uAD6C\uAC04\uC744 \uC81C\uC678\uD569\uB2C8\uB2E4", children: [_jsx("span", { className: "text-fg-dim mr-1 font-mono text-xs", children: "D." }), " \uBC84\uB9AC\uAE30", countMap.has("discarded") && (_jsxs("span", { className: "text-fg-dim ml-1 text-xs", children: ["(", countMap.get("discarded"), ")"] }))] }), _jsxs("button", { className: "bg-canvas border border-line hover:border-line-strong text-fg-muted hover:text-fg px-3 py-2 rounded-lg text-sm transition-colors", onClick: playSegment, children: [_jsx("span", { className: "text-fg-dim mr-1 font-mono text-xs", children: "R." }), " \uB2E4\uC2DC\uB4E3\uAE30"] }), _jsxs("button", { className: "bg-canvas border border-line hover:border-line-strong text-fg-muted hover:text-fg px-3 py-2 rounded-lg text-sm transition-colors", onClick: handleUndo, children: [_jsx("span", { className: "text-fg-dim mr-1 font-mono text-xs", children: "Z." }), " \uB418\uB3CC\uB9AC\uAE30"] }), _jsx("div", { className: "flex-1" }), _jsx("button", { className: "bg-primary hover:bg-primary-hover text-canvas px-5 py-2 rounded-lg text-sm font-semibold transition-colors", onClick: handleDone, children: "\uBD84\uB958 \uC644\uB8CC" })] })] }));
}
