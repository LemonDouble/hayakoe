import { Fragment, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router";
import * as videosApi from "../api/videos";
import type { VideoStatus, VadParams } from "../api/videos";
import { usePolling } from "../hooks/usePolling";
import ProgressBar from "../components/ProgressBar";
import CardClassifier from "../components/CardClassifier";
import ReviewEditor from "../components/ReviewEditor";
import { t } from "../i18n";

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? t("dashboard.format_duration.min_sec", { m, s }) : t("dashboard.format_duration.sec", { s });
}

const STAGE_ORDER = ["extract", "separate", "vad", "classify", "transcribe", "review"];

const STAGE_LABEL_KEYS: Record<string, string> = {
  extract: "detail.stages.extract",
  separate: "detail.stages.separate",
  vad: "detail.stages.vad",
  classify: "detail.stages.classify",
  transcribe: "detail.stages.transcribe",
  review: "detail.stages.review",
};

const STAGE_DESC_KEYS: Record<string, { titleKey: string; descKey: string }> = {
  extract: {
    titleKey: "detail.stage_desc.extract.title",
    descKey: "detail.stage_desc.extract.desc",
  },
  separate: {
    titleKey: "detail.stage_desc.separate.title",
    descKey: "detail.stage_desc.separate.desc",
  },
  vad: {
    titleKey: "detail.stage_desc.vad.title",
    descKey: "detail.stage_desc.vad.desc",
  },
  classify: {
    titleKey: "detail.stage_desc.classify.title",
    descKey: "detail.stage_desc.classify.desc",
  },
  transcribe: {
    titleKey: "detail.stage_desc.transcribe.title",
    descKey: "detail.stage_desc.transcribe.desc",
  },
  review: {
    titleKey: "detail.stage_desc.review.title",
    descKey: "detail.stage_desc.review.desc",
  },
};

const VAD_PRESETS: { labelKey: string; descKey: string; params: VadParams }[] = [
  {
    labelKey: "detail.vad.preset.default.label",
    descKey: "detail.vad.preset.default.desc",
    params: { min_segment_sec: 1.0, max_segment_sec: 8.0, threshold: 0.3, min_silence_ms: 50 },
  },
  {
    labelKey: "detail.vad.preset.noisy.label",
    descKey: "detail.vad.preset.noisy.desc",
    params: { min_segment_sec: 1.5, max_segment_sec: 8.0, threshold: 0.65, min_silence_ms: 40 },
  },
  {
    labelKey: "detail.vad.preset.monologue.label",
    descKey: "detail.vad.preset.monologue.desc",
    params: { min_segment_sec: 1.0, max_segment_sec: 12.0, threshold: 0.3, min_silence_ms: 80 },
  },
];

// 각 단계에서 "다음 실행" 버튼에 쓸 API 함수 (vad는 파라미터 필요하므로 별도 처리)
const STAGE_ACTIONS: Record<string, (id: string) => Promise<void>> = {
  extract: videosApi.startExtract,
  separate: videosApi.startSeparate,
  transcribe: videosApi.startTranscription,
};

const DEFAULT_VAD_PARAMS: VadParams = {
  min_segment_sec: 1.0,
  max_segment_sec: 8.0,
  threshold: 0.3,
  min_silence_ms: 50,
};

function stageIndex(stage: string): number {
  if (stage.startsWith("processing:")) {
    return STAGE_ORDER.indexOf(stage.split(":")[1]);
  }
  if (stage === "classifying") return STAGE_ORDER.indexOf("classify");
  if (stage === "done") return STAGE_ORDER.length;
  return STAGE_ORDER.indexOf(stage);
}

export default function VideoDetail() {
  const { videoId } = useParams<{ videoId: string }>();
  const navigate = useNavigate();
  const [status, setStatus] = useState<VideoStatus | null>(null);
  const [error, setError] = useState("");
  const [vadParams, setVadParams] = useState<VadParams>({ ...DEFAULT_VAD_PARAMS });

  const isProcessing = status?.stage.startsWith("processing:") ?? false;

  usePolling(
    async () => {
      if (!videoId) return;
      try {
        setStatus(await videosApi.getStatus(videoId));
      } catch {
        setError(t("detail.error.status_fetch"));
      }
    },
    1500,
    isProcessing
  );

  useEffect(() => {
    if (!videoId) return;
    videosApi
      .getStatus(videoId)
      .then(setStatus)
      .catch(() => setError(t("detail.error.not_found")));
  }, [videoId]);

  const refreshStatus = async () => {
    if (!videoId) return;
    setStatus(await videosApi.getStatus(videoId));
  };

  const handleRollback = async (stage: string) => {
    if (!videoId) return;
    const stageLabel = STAGE_LABEL_KEYS[stage] ? t(STAGE_LABEL_KEYS[stage]) : stage;
    if (
      !confirm(t("detail.rollback_confirm", { stage: stageLabel }))
    )
      return;
    await videosApi.rollbackVideo(videoId, stage);
    await refreshStatus();
  };

  const [stageError, setStageError] = useState("");
  const [pendingStage, setPendingStage] = useState<string | null>(null);

  const handleRunStage = async (stage: string) => {
    if (!videoId) return;
    const action = STAGE_ACTIONS[stage];
    if (!action) return;
    setStageError("");
    setPendingStage(null);
    try {
      await action(videoId);
      setTimeout(refreshStatus, 500);
    } catch (e: unknown) {
      const resp = (e as { response?: { status?: number; data?: { detail?: string } } })?.response;
      if (resp?.status === 409) {
        setPendingStage(stage);
      } else {
        setStageError(resp?.data?.detail || t("detail.error.run_failed"));
      }
    }
  };

  // 409 자동 재시도
  useEffect(() => {
    if (!pendingStage || !videoId) return;
    const action = STAGE_ACTIONS[pendingStage];
    if (!action) return;

    const interval = setInterval(async () => {
      try {
        await action(videoId);
        setPendingStage(null);
        videosApi.getStatus(videoId).then(setStatus);
      } catch {
        // 아직 실행 중 — 계속 대기
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [pendingStage, videoId]);

  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <p className="text-error">{error}</p>
        <button className="text-primary mt-2 hover:text-primary-hover" onClick={() => navigate("/")}>
          {t("detail.back")}
        </button>
      </div>
    );
  }

  if (!status) {
    return <div className="max-w-4xl mx-auto p-6 text-fg-muted">{t("detail.loading")}</div>;
  }

  const stage = status.stage;
  const currentIdx = stageIndex(stage);
  const allStages = [...STAGE_ORDER, "done"];

  return (
    <div className="max-w-4xl mx-auto p-6 pb-16">
      {/* 헤더 */}
      <div className="flex items-center gap-4 mb-8">
        <button
          className="bg-transparent border border-line hover:border-line-strong text-fg-muted hover:text-fg px-3.5 py-1.5 rounded-md text-[13px] font-semibold transition-colors"
          onClick={() => navigate("/")}
        >
          {t("detail.back")}
        </button>
        <div>
          <h1 className="font-display text-xl font-bold text-fg">{status.filename}</h1>
          <span className="text-fg-dim text-xs font-mono">#{videoId}</span>
        </div>
      </div>

      {/* 파이프라인 스테퍼 */}
      <div className="mb-2">
        <div className="flex items-start">
          {allStages.map((s, i) => {
            const isDone = s === "done";
            const stageIdx = isDone ? STAGE_ORDER.length : i;
            const isCompleted = isDone ? stage === "done" : stageIdx < currentIdx;
            const isCurrent = stageIdx === currentIdx && stage !== "done";
            const isClickable = !isDone && isCompleted;
            const stageLabel = isDone
              ? t("detail.stages.done")
              : STAGE_LABEL_KEYS[s]
                ? t(STAGE_LABEL_KEYS[s])
                : s;

            return (
              <Fragment key={s}>
                {i > 0 && (
                  <div
                    className={`flex-1 h-0.5 mt-4 transition-colors ${
                      stageIdx <= currentIdx || (isDone && stage === "done")
                        ? "bg-primary"
                        : "bg-line"
                    }`}
                  />
                )}
                <div
                  className={`flex flex-col items-center ${isClickable ? "cursor-pointer group" : ""}`}
                  onClick={() => isClickable && handleRollback(s)}
                  title={isClickable ? t("detail.rollback_title", { stage: stageLabel }) : ""}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 transition-colors ${
                      isCompleted
                        ? "bg-primary text-canvas group-hover:bg-primary-hover"
                        : isCurrent
                          ? "bg-surface border-2 border-primary text-primary"
                          : "bg-surface-2 border border-line text-fg-dim"
                    }`}
                  >
                    {isCompleted ? "✓" : isDone ? "" : i + 1}
                  </div>
                  <span
                    className={`text-[10px] mt-1.5 whitespace-nowrap transition-colors font-semibold ${
                      isCompleted
                        ? "text-primary group-hover:text-primary-hover"
                        : isCurrent
                          ? "text-primary"
                          : "text-fg-dim"
                    }`}
                  >
                    {stageLabel}
                  </span>
                </div>
              </Fragment>
            );
          })}
        </div>
      </div>

      {/* 롤백 힌트 */}
      {currentIdx > 0 && stage !== "done" && (
        <p className="text-fg-dim text-[11px] text-center mb-6">
          {t("detail.rollback_hint")}
        </p>
      )}
      {(currentIdx === 0 || stage === "done") && <div className="mb-6" />}

      {/* 에러 표시 */}
      {status.error && (
        <div className="bg-error/[0.06] border border-error/25 rounded-xl p-5 mb-6">
          <p className="text-error font-semibold mb-2">
            {t("detail.error.stage_error", {
              stage: STAGE_LABEL_KEYS[status.error.stage]
                ? t(STAGE_LABEL_KEYS[status.error.stage])
                : status.error.stage,
            })}
          </p>
          <pre className="text-error/90 text-sm whitespace-pre-wrap break-words bg-canvas border border-line rounded-lg p-3 font-mono">
            {status.error.message}
          </pre>
          <p className="text-fg-dim text-xs mt-3">
            {t("detail.error.retry_hint")}
          </p>
        </div>
      )}

      {/* 처리 중 (폴링으로 진행률 표시) */}
      {isProcessing && status.processing && (
        <div className="bg-surface border border-line rounded-xl p-6 mb-6">
          <p className="text-fg text-sm font-semibold mb-3">
            {STAGE_DESC_KEYS[status.processing.stage]
              ? t(STAGE_DESC_KEYS[status.processing.stage].titleKey)
              : STAGE_LABEL_KEYS[status.processing.stage]
                ? t(STAGE_LABEL_KEYS[status.processing.stage])
                : t("detail.processing")}
          </p>
          <ProgressBar
            progress={status.processing.progress}
            message={status.processing.message}
          />
          {status.processing.stage === "separate" && (
            <p className="text-fg-dim text-xs mt-3">
              {t("detail.separate_hint")}
            </p>
          )}
        </div>
      )}

      {/* 수동 실행 버튼: extract, separate, transcribe */}
      {!isProcessing &&
        stage in STAGE_ACTIONS &&
        stage !== "classify" &&
        stage !== "classifying" && (
          <div className="bg-surface border border-line rounded-xl p-6">
            <div className="text-center mb-5">
              <p className="text-primary text-[11px] font-bold uppercase tracking-[1.5px] mb-2 font-display">NEXT STEP</p>
              <p className="font-display text-xl font-bold text-fg mb-2">
                {STAGE_DESC_KEYS[stage] ? t(STAGE_DESC_KEYS[stage].titleKey) : STAGE_LABEL_KEYS[stage] ? t(STAGE_LABEL_KEYS[stage]) : stage}
              </p>
              <p className="text-fg-muted text-sm max-w-md mx-auto leading-relaxed">
                {STAGE_DESC_KEYS[stage] ? t(STAGE_DESC_KEYS[stage].descKey) : ""}
              </p>
            </div>
            {pendingStage && (
              <div className="bg-warning/[0.06] border border-warning/25 rounded-lg p-4 mb-4 text-center">
                <p className="text-warning text-sm font-semibold mb-1">{t("detail.pending.title")}</p>
                <p className="text-fg-muted text-xs">{t("detail.pending.desc")}</p>
              </div>
            )}
            {stageError && !pendingStage && (
              <div className="bg-error/[0.06] border border-error/25 rounded-lg p-3 mb-4 text-error text-sm text-center">
                {stageError}
              </div>
            )}
            <div className="text-center">
              <button
                className={`px-8 py-2.5 rounded-lg font-semibold text-sm transition-colors ${
                  pendingStage
                    ? "bg-warning/20 text-warning cursor-wait animate-pulse"
                    : "bg-primary hover:bg-primary-hover text-canvas"
                }`}
                onClick={() => !pendingStage && handleRunStage(stage)}
                disabled={!!pendingStage}
              >
                {pendingStage
                  ? t("detail.pending.waiting")
                  : t("detail.run_stage", {
                      stage: STAGE_LABEL_KEYS[stage] ? t(STAGE_LABEL_KEYS[stage]) : stage,
                    })}
              </button>
            </div>
          </div>
        )}

      {/* VAD 실행 (파라미터 설정 포함) */}
      {!isProcessing && stage === "vad" && (
        <div className="bg-surface border border-line rounded-xl p-6">
          {/* 단계 설명 */}
          <div className="text-center mb-6">
            <p className="text-primary text-[11px] font-bold uppercase tracking-[1.5px] mb-2 font-display">NEXT STEP</p>
            <p className="font-display text-xl font-bold text-fg mb-2">
              {t("detail.stage_desc.vad.title")}
            </p>
            <p className="text-fg-muted text-sm max-w-lg mx-auto leading-relaxed">
              {t("detail.stage_desc.vad.desc")}
            </p>
          </div>

          {/* 프리셋 */}
          <div className="max-w-lg mx-auto mb-6">
            <p className="text-fg text-sm font-semibold mb-2">{t("detail.vad.quick_settings")}</p>
            <div className="grid grid-cols-3 gap-2">
              {VAD_PRESETS.map((preset) => (
                <button
                  key={preset.labelKey}
                  className="bg-canvas border border-line hover:border-primary/40 rounded-lg px-3 py-2.5 text-left transition-colors"
                  onClick={() => setVadParams({ ...preset.params })}
                >
                  <p className="text-fg text-xs font-semibold">{t(preset.labelKey)}</p>
                  <p className="text-fg-dim text-[10px] mt-0.5">{t(preset.descKey)}</p>
                </button>
              ))}
            </div>
          </div>

          {/* 세부 파라미터 */}
          <div className="space-y-4 mb-6 max-w-lg mx-auto text-sm">
            <label className="block text-fg-muted">
              <div className="flex items-center justify-between">
                <span>{t("detail.vad.min_segment")}</span>
                <input
                  type="number"
                  step="0.1"
                  className="w-24 bg-canvas border border-line rounded-lg px-3 py-1.5 text-fg text-right focus:outline-none focus:border-primary/50 transition-colors"
                  value={vadParams.min_segment_sec}
                  onChange={(e) => setVadParams({ ...vadParams, min_segment_sec: +e.target.value })}
                />
              </div>
              <p className="text-fg-dim text-xs mt-1">
                {t("detail.vad.min_segment_hint")}
              </p>
            </label>
            <label className="block text-fg-muted">
              <div className="flex items-center justify-between">
                <span>{t("detail.vad.max_segment")}</span>
                <input
                  type="number"
                  step="0.5"
                  className="w-24 bg-canvas border border-line rounded-lg px-3 py-1.5 text-fg text-right focus:outline-none focus:border-primary/50 transition-colors"
                  value={vadParams.max_segment_sec}
                  onChange={(e) => setVadParams({ ...vadParams, max_segment_sec: +e.target.value })}
                />
              </div>
              <p className="text-fg-dim text-xs mt-1">
                {t("detail.vad.max_segment_hint")}
              </p>
            </label>
            <label className="block text-fg-muted">
              <div className="flex items-center justify-between">
                <span>{t("detail.vad.threshold")}</span>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  className="w-24 bg-canvas border border-line rounded-lg px-3 py-1.5 text-fg text-right focus:outline-none focus:border-primary/50 transition-colors"
                  value={vadParams.threshold}
                  onChange={(e) => setVadParams({ ...vadParams, threshold: +e.target.value })}
                />
              </div>
              <p className="text-fg-dim text-xs mt-1">
                {t("detail.vad.threshold_hint")}
              </p>
            </label>
            <label className="block text-fg-muted">
              <div className="flex items-center justify-between">
                <span>{t("detail.vad.min_silence")}</span>
                <input
                  type="number"
                  step="10"
                  className="w-24 bg-canvas border border-line rounded-lg px-3 py-1.5 text-fg text-right focus:outline-none focus:border-primary/50 transition-colors"
                  value={vadParams.min_silence_ms}
                  onChange={(e) => setVadParams({ ...vadParams, min_silence_ms: +e.target.value })}
                />
              </div>
              <p className="text-fg-dim text-xs mt-1">
                {t("detail.vad.min_silence_hint")}
              </p>
            </label>
          </div>

          {/* 상황별 팁 */}
          <details className="max-w-lg mx-auto mb-6">
            <summary className="text-fg-muted text-xs cursor-pointer hover:text-fg transition-colors">
              {t("detail.vad.tips_toggle")}
            </summary>
            <div className="bg-canvas border border-line rounded-lg p-4 mt-2 text-xs text-fg-muted space-y-1.5 leading-relaxed">
              <p>
                <span className="text-primary font-semibold">{t("detail.vad.tip_fast_dialogue_title")}</span>
                {t("detail.vad.tip_fast_dialogue_desc")}
              </p>
              <p>
                <span className="text-primary font-semibold">{t("detail.vad.tip_noisy_title")}</span>
                {t("detail.vad.tip_noisy_desc")}
              </p>
              <p>
                <span className="text-primary font-semibold">{t("detail.vad.tip_monologue_title")}</span>
                {t("detail.vad.tip_monologue_desc")}
              </p>
              <p>
                <span className="text-primary font-semibold">{t("detail.vad.tip_interjection_title")}</span>
                {t("detail.vad.tip_interjection_desc")}
              </p>
            </div>
          </details>

          <div className="text-center">
            <button
              className="bg-primary hover:bg-primary-hover text-canvas px-8 py-2.5 rounded-lg font-semibold text-sm transition-colors"
              onClick={async () => {
                if (!videoId) return;
                await videosApi.startVad(videoId, vadParams);
                setTimeout(refreshStatus, 500);
              }}
            >
              {t("detail.vad.run")}
            </button>
            <p className="text-fg-dim text-xs mt-3">
              {t("detail.run_after_hint")}
            </p>
          </div>
        </div>
      )}

      {/* 분류 UI */}
      {(stage === "classify" || stage === "classifying") && videoId && (
        <CardClassifier videoId={videoId} sourceFile={status.source_file} onDone={refreshStatus} />
      )}

      {/* 검토 UI */}
      {stage === "review" && videoId && (
        <ReviewEditor videoId={videoId} onDone={refreshStatus} />
      )}

      {/* 완료 */}
      {stage === "done" && (
        <div className="bg-success/[0.06] border border-success/25 rounded-xl p-8 text-center">
          <div className="w-14 h-14 rounded-full bg-success/15 border border-success/40 text-success flex items-center justify-center text-xl mx-auto mb-4">
            {"✓"}
          </div>
          <p className="font-display text-xl font-bold text-success mb-2">{t("detail.done.title")}</p>
          <p className="text-fg-muted text-sm mb-6">{t("detail.done.description")}</p>

          {status.summary && status.summary.length > 0 && (
            <div className="max-w-sm mx-auto space-y-2 mb-6">
              {status.summary.map((s) => (
                <div
                  key={s.name}
                  className="flex justify-between items-center bg-canvas border border-line rounded-lg px-4 py-2.5 text-sm"
                >
                  <span className={s.name === "discarded" ? "text-fg-dim" : "text-fg font-semibold"}>
                    {s.name === "discarded" ? t("detail.done.discarded") : s.name}
                  </span>
                  <span className="text-fg-muted font-mono text-xs">
                    {t("common.count_duration", { count: s.count, duration: formatDuration(s.total_duration) })}
                  </span>
                </div>
              ))}
            </div>
          )}

          <p className="text-fg-dim text-sm">
            {t("detail.done.back_hint")}
          </p>
        </div>
      )}
    </div>
  );
}
