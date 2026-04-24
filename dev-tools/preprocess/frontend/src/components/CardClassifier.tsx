import { useCallback, useEffect, useRef, useState } from "react";
import * as clsApi from "../api/classification";
import * as speakersApi from "../api/speakers";
import type { SegmentInfo, ClassificationState } from "../api/classification";
import { t } from "../i18n";

interface Props {
  videoId: string;
  sourceFile: string | null;
  onDone: () => void;
}

const BUFFER_SIZE = 10;

export default function CardClassifier({ videoId, sourceFile, onDone }: Props) {
  const [speakers, setSpeakers] = useState<string[]>([]);
  const [segments, setSegments] = useState<SegmentInfo[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [totalUnclassified, setTotalUnclassified] = useState(0);
  const [classified, setClassified] = useState(0);
  const [totalAll, setTotalAll] = useState(0);
  const [loading, setLoading] = useState(true);
  const [bucketCounts, setBucketCounts] = useState<ClassificationState["speakers"]>([]);

  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

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
    if (!audioRef.current || !videoRef.current || !current) return;
    videoRef.current.currentTime = current.start;
    videoRef.current.play().catch(() => {});
    audioRef.current.currentTime = 0;
    audioRef.current.play().catch(() => {});
  }, [current]);

  // 오디오 끝나면 영상도 정지
  useEffect(() => {
    const audio = audioRef.current;
    const video = videoRef.current;
    if (!audio || !video) return;

    const handleEnded = () => {
      video.pause();
    };
    audio.addEventListener("ended", handleEnded);
    return () => audio.removeEventListener("ended", handleEnded);
  }, [audioUrl]);

  // 분류 처리
  const handleClassify = useCallback(
    async (speaker: string) => {
      if (!current) return;
      await clsApi.classifySegment(videoId, current.file, speaker);

      // 다음 세그먼트로
      const nextIdx = currentIdx + 1;
      if (nextIdx >= segments.length) {
        await refillBuffer();
      } else {
        setCurrentIdx(nextIdx);
        setClassified((c) => c + 1);
        setTotalUnclassified((t) => t - 1);
        // 버킷 카운트 비동기 갱신
        refreshBuckets();
      }
    },
    [current, currentIdx, segments.length, videoId, refillBuffer, refreshBuckets]
  );

  // Undo
  const handleUndo = useCallback(async () => {
    await clsApi.undoClassification(videoId);
    await refillBuffer();
  }, [videoId, refillBuffer]);

  // 분류 완료
  const handleDone = async () => {
    if (!confirm(t("classifier.confirm_done"))) return;
    await clsApi.markDone(videoId);
    onDone();
  };

  // 키보드 단축키
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      // 1~9: 화자 배정
      if (e.key >= "1" && e.key <= "9") {
        const idx = parseInt(e.key) - 1;
        if (idx < speakers.length) handleClassify(speakers[idx]);
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

  if (loading) return <div className="p-6 text-fg-muted">{t("classifier.loading")}</div>;

  if (totalUnclassified === 0 && segments.length === 0) {
    return (
      <div className="bg-surface border border-line rounded-xl p-8 text-center">
        <div className="w-12 h-12 rounded-full bg-success/15 border border-success/40 text-success flex items-center justify-center text-lg mx-auto mb-3">
          {"✓"}
        </div>
        <p className="font-display text-xl font-bold text-success mb-2">{t("classifier.all_done.title")}</p>
        <p className="text-fg-muted text-sm mb-5">{t("classifier.all_done.description")}</p>
        <button
          className="bg-primary hover:bg-primary-hover text-canvas px-6 py-2.5 rounded-lg font-semibold text-sm transition-colors"
          onClick={handleDone}
        >
          {t("classifier.all_done.next")}
        </button>
      </div>
    );
  }

  // 버킷별 카운트 맵
  const countMap = new Map(bucketCounts.map((b) => [b.name, b.count]));
  const progressPct = totalAll > 0 ? Math.round((classified / totalAll) * 100) : 0;

  return (
    <div className="bg-surface border border-line rounded-xl p-6">
      {/* 안내 배너 */}
      <div className="bg-primary/[0.08] border border-primary/25 rounded-lg p-4 mb-5">
        <p className="text-primary text-[11px] font-bold uppercase tracking-[1.5px] mb-1.5 font-display">{t("classifier.title")}</p>
        <p className="text-fg-muted text-xs leading-relaxed mb-3">
          {t("classifier.description")}
        </p>
        <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-fg-dim">
          <span><kbd className="bg-canvas border border-line px-1.5 py-0.5 rounded text-fg font-mono text-[11px]">1-9</kbd> {t("classifier.shortcut.assign")}</span>
          <span><kbd className="bg-canvas border border-line px-1.5 py-0.5 rounded text-fg font-mono text-[11px]">D</kbd> {t("classifier.shortcut.discard")}</span>
          <span><kbd className="bg-canvas border border-line px-1.5 py-0.5 rounded text-fg font-mono text-[11px]">R</kbd> {t("classifier.shortcut.replay")}</span>
          <span><kbd className="bg-canvas border border-line px-1.5 py-0.5 rounded text-fg font-mono text-[11px]">Z</kbd> {t("classifier.shortcut.undo")}</span>
        </div>
      </div>

      {/* 진행률 바 */}
      <div className="mb-5">
        <div className="flex justify-between text-sm mb-1.5">
          <span className="text-fg-muted">{t("classifier.progress")}</span>
          <span className="text-primary font-mono font-semibold">
            {classified} / {totalAll} ({progressPct}%)
          </span>
        </div>
        <div className="w-full bg-line rounded-full h-1.5 overflow-hidden">
          <div
            className="progress-fill h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>
        <p className="text-xs text-fg-dim mt-1.5">{t("classifier.remaining", { count: totalUnclassified })}</p>
      </div>

      {/* 원본 영상 플레이어 (뮤트) */}
      {videoUrl && (
        <div className="mb-4">
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full max-h-96 bg-canvas border border-line rounded-lg"
            muted
          />
        </div>
      )}

      {/* 세그먼트 오디오 (숨김 - 영상과 동기 재생) */}
      {current && (
        <div className="mb-5">
          <div className="flex justify-between text-sm text-fg-muted mb-2 bg-canvas border border-line rounded-lg px-3 py-2">
            <span className="font-mono text-xs text-fg-dim">{current.file}</span>
            <span className="text-xs text-fg">
              {current.start.toFixed(1)}s ~ {current.end.toFixed(1)}s
              <span className="text-fg-dim ml-1">({current.duration.toFixed(1)}s)</span>
            </span>
          </div>
          <audio ref={audioRef} src={audioUrl} />
        </div>
      )}

      {/* 화자 배정 */}
      <div className="mb-4">
        <p className="text-xs text-fg-dim mb-2">{t("classifier.assign_hint")}</p>
        <div className="grid grid-cols-3 gap-2">
          {speakers.map((s, i) => (
            <button
              key={s}
              className="bg-canvas border border-line hover:border-primary/50 hover:bg-primary/[0.08] px-3 py-2.5 rounded-lg text-sm transition-colors text-left group"
              onClick={() => handleClassify(s)}
            >
              <span className="text-fg-dim group-hover:text-primary mr-1.5 font-mono text-xs">{i + 1}.</span>
              <span className="text-fg group-hover:text-primary font-semibold">{s}</span>
              {countMap.has(s) && (
                <span className="text-fg-dim ml-1.5 text-xs">({countMap.get(s)})</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* 조작 */}
      <div className="flex items-center gap-2">
        <button
          className="bg-canvas border border-line hover:border-error/40 hover:text-error text-fg-muted px-3 py-2 rounded-lg text-sm transition-colors"
          onClick={() => handleClassify("discarded")}
          title={t("classifier.shortcut.discard")}
        >
          <span className="text-fg-dim mr-1 font-mono text-xs">D.</span> {t("classifier.discard")}
          {countMap.has("discarded") && (
            <span className="text-fg-dim ml-1 text-xs">({countMap.get("discarded")})</span>
          )}
        </button>
        <button
          className="bg-canvas border border-line hover:border-line-strong text-fg-muted hover:text-fg px-3 py-2 rounded-lg text-sm transition-colors"
          onClick={playSegment}
        >
          <span className="text-fg-dim mr-1 font-mono text-xs">R.</span> {t("classifier.replay")}
        </button>
        <button
          className="bg-canvas border border-line hover:border-line-strong text-fg-muted hover:text-fg px-3 py-2 rounded-lg text-sm transition-colors"
          onClick={handleUndo}
        >
          <span className="text-fg-dim mr-1 font-mono text-xs">Z.</span> {t("classifier.undo")}
        </button>
        <div className="flex-1" />
        <button
          className="bg-primary hover:bg-primary-hover text-canvas px-5 py-2 rounded-lg text-sm font-semibold transition-colors"
          onClick={handleDone}
        >
          {t("classifier.done")}
        </button>
      </div>
    </div>
  );
}
