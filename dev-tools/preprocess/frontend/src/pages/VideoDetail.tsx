import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router";
import * as videosApi from "../api/videos";
import type { VideoStatus, VadParams } from "../api/videos";
import { usePolling } from "../hooks/usePolling";
import ProgressBar from "../components/ProgressBar";
import CardClassifier from "../components/CardClassifier";
import ReviewEditor from "../components/ReviewEditor";
import PipelineStepper from "../components/PipelineStepper";
import StageRunCard from "../components/StageRunCard";
import VadRunCard from "../components/VadRunCard";
import DoneSummary from "../components/DoneSummary";
import { STAGE_ORDER, isKnownStage, stageDescKeys, stageLabelOf } from "../utils/stages";
import { t } from "../i18n";

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
    if (
      !confirm(t("detail.rollback_confirm", { stage: stageLabelOf(stage) }))
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
        refreshStatus();
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

      <PipelineStepper stage={stage} currentIdx={currentIdx} onRollback={handleRollback} />

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
            {t("detail.error.stage_error", { stage: stageLabelOf(status.error.stage) })}
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
            {isKnownStage(status.processing.stage)
              ? t(stageDescKeys(status.processing.stage).titleKey)
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
      {!isProcessing && stage in STAGE_ACTIONS && (
        <StageRunCard
          stage={stage}
          pendingStage={pendingStage}
          stageError={stageError}
          onRun={handleRunStage}
        />
      )}

      {/* VAD 실행 (파라미터 설정 포함) */}
      {!isProcessing && stage === "vad" && (
        <VadRunCard
          params={vadParams}
          onChange={setVadParams}
          onRun={async () => {
            if (!videoId) return;
            await videosApi.startVad(videoId, vadParams);
            setTimeout(refreshStatus, 500);
          }}
        />
      )}

      {/* 분류 UI */}
      {(stage === "classify" || stage === "classifying") && videoId && (
        <CardClassifier videoId={videoId} sourceFile={status.source_file} onDone={refreshStatus} />
      )}

      {/* 검토 UI */}
      {stage === "review" && videoId && (
        <ReviewEditor videoId={videoId} onDone={refreshStatus} />
      )}

      {stage === "done" && <DoneSummary summary={status.summary} />}
    </div>
  );
}
