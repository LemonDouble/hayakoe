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

function errorDetail(e: unknown): string {
  const resp = (e as { response?: { data?: { detail?: string } } })?.response;
  return resp?.data?.detail || t("detail.error.run_failed");
}

export default function CardClassifier({ videoId, sourceFile, onDone }: Props) {
  const [speakers, setSpeakers] = useState<string[]>([]);
  const [segments, setSegments] = useState<SegmentInfo[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [totalUnclassified, setTotalUnclassified] = useState(0);
  const [classified, setClassified] = useState(0);
  const [totalAll, setTotalAll] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [bucketCounts, setBucketCounts] = useState<ClassificationState["speakers"]>([]);

  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const queueRef = useRef<Promise<void>>(Promise.resolve());

  const enqueue = useCallback((task: () => Promise<void>) => {
    const run = queueRef.current.then(task);
    // 한 POST가 실패해도 뒤의 항목이 계속 흐르도록 체인은 rejection을 삼킨다 (에러는 run으로 전파)
    queueRef.current = run.then(
      () => undefined,
      () => undefined
    );
    return run;
  }, []);

  const drainQueue = useCallback(async () => {
    // 대기 중 새 POST가 enqueue될 수 있으므로 tail이 안정될 때까지 반복
    let tail: Promise<void>;
    do {
      tail = queueRef.current;
      await tail;
    } while (tail !== queueRef.current);
  }, []);

  const refreshBuckets = useCallback(async () => {
    const state = await clsApi.getClassification(videoId);
    setBucketCounts(state.speakers);
  }, [videoId]);

  const refillBuffer = useCallback(async () => {
    // in-flight POST가 서버에 반영되기 전에 조회하면 방금 분류한 세그먼트가 목록에 재등장함
    await drainQueue();
    const data = await clsApi.getUnclassified(videoId, 0, BUFFER_SIZE);
    setSegments(data.segments);
    setCurrentIdx(0);
    setTotalUnclassified(data.total);
    setClassified(data.classified);
    setTotalAll(data.total_all);
    await refreshBuckets();
  }, [videoId, drainQueue, refreshBuckets]);

  const resync = useCallback(() => {
    refillBuffer().catch((e: unknown) => setError(errorDetail(e)));
  }, [refillBuffer]);

  useEffect(() => {
    (async () => {
      const [spk] = await Promise.all([speakersApi.listSpeakers(), refillBuffer()]);
      setSpeakers(spk);
      setLoading(false);
    })();
  }, [videoId, refillBuffer]);

  const current = segments[currentIdx] || null;

  const audioUrl = current
    ? `/api/media/videos/${videoId}/segments/unclassified/${current.file}`
    : "";

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

  const handleClassify = useCallback(
    (speaker: string) => {
      if (!current) return;
      const { file, duration } = current;
      setError("");

      // fire-and-forget이 아닌 직렬 큐: 서버 도착 순서를 입력 순서와 일치시킨다
      enqueue(() => clsApi.classifySegment(videoId, file, speaker)).catch((e: unknown) => {
        // 실패한 세그먼트는 서버에 미분류로 남아있으므로 재조회로 낙관 갱신을 되돌린다
        setError(errorDetail(e));
        resync();
      });

      setClassified((c) => c + 1);
      setTotalUnclassified((t) => t - 1);
      setBucketCounts((prev) =>
        prev.some((b) => b.name === speaker)
          ? prev.map((b) =>
              b.name === speaker
                ? { ...b, count: b.count + 1, total_duration: b.total_duration + duration }
                : b
            )
          : // count 0인 화자(예: 첫 discarded)는 서버 응답에서 생략되므로 새로 추가
            [...prev, { name: speaker, count: 1, total_duration: duration }]
      );

      // 버퍼 소진 시에도 idx를 범위 밖으로 밀어, refill 완료 전 같은 카드에 대한 재입력을 차단
      const nextIdx = currentIdx + 1;
      setCurrentIdx(nextIdx);
      if (nextIdx >= segments.length) resync();
    },
    [current, currentIdx, segments.length, videoId, enqueue, resync]
  );

  const handleUndo = useCallback(async () => {
    setError("");
    // in-flight 분류 POST가 남은 채로 undo하면 서버 history의 다른 항목이 되돌아감
    await drainQueue();
    try {
      await clsApi.undoClassification(videoId);
    } catch (e: unknown) {
      setError(errorDetail(e));
    }
    resync();
  }, [videoId, drainQueue, resync]);

  const handleDone = async () => {
    if (!confirm(t("classifier.confirm_done"))) return;
    await clsApi.markDone(videoId);
    onDone();
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

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

  const countMap = new Map(bucketCounts.map((b) => [b.name, b.count]));
  const progressPct = totalAll > 0 ? Math.round((classified / totalAll) * 100) : 0;

  return (
    <div className="bg-surface border border-line rounded-xl p-6">
      {error && (
        <div className="bg-error/[0.06] border border-error/25 rounded-lg p-3 mb-4 text-error text-sm">
          {error}
        </div>
      )}
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
