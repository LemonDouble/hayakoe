import { Fragment, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router";
import * as videosApi from "../api/videos";
import type { VideoStatus, VadParams } from "../api/videos";
import { usePolling } from "../hooks/usePolling";
import ProgressBar from "../components/ProgressBar";
import CardClassifier from "../components/CardClassifier";
import ReviewEditor from "../components/ReviewEditor";

const STAGE_ORDER = ["extract", "separate", "vad", "classify", "transcribe", "review"];

const STAGE_LABELS: Record<string, string> = {
  extract: "추출",
  separate: "배경음 제거",
  vad: "VAD 세그먼팅",
  classify: "분류",
  transcribe: "전사",
  review: "검토",
};

const STAGE_DESCRIPTIONS: Record<string, { title: string; desc: string }> = {
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

const VAD_PRESETS: { label: string; desc: string; params: VadParams }[] = [
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
        setError("상태 조회 실패");
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
      .catch(() => setError("영상을 찾을 수 없습니다."));
  }, [videoId]);

  const refreshStatus = async () => {
    if (!videoId) return;
    setStatus(await videosApi.getStatus(videoId));
  };

  const handleRollback = async (stage: string) => {
    if (!videoId) return;
    if (
      !confirm(
        `"${STAGE_LABELS[stage]}" 단계부터 재처리하시겠습니까?\n이후 단계 데이터가 모두 삭제됩니다.`
      )
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
        setStageError(resp?.data?.detail || "실행에 실패했습니다.");
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
        <p className="text-red-400">{error}</p>
        <button className="text-blue-400 mt-2" onClick={() => navigate("/")}>
          돌아가기
        </button>
      </div>
    );
  }

  if (!status) {
    return <div className="max-w-4xl mx-auto p-6">로딩 중...</div>;
  }

  const stage = status.stage;
  const currentIdx = stageIndex(stage);
  const allStages = [...STAGE_ORDER, "done"];

  return (
    <div className="max-w-4xl mx-auto p-6 pb-16">
      {/* 헤더 */}
      <div className="flex items-center gap-4 mb-8">
        <button
          className="bg-slate-700 hover:bg-slate-600 text-slate-300 hover:text-white px-3 py-1.5 rounded-lg text-sm transition-colors"
          onClick={() => navigate("/")}
        >
          &larr; 목록
        </button>
        <div>
          <h1 className="text-xl font-bold">{status.filename}</h1>
          <span className="text-slate-500 text-xs font-mono">#{videoId}</span>
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

            return (
              <Fragment key={s}>
                {i > 0 && (
                  <div
                    className={`flex-1 h-0.5 mt-4 transition-colors ${
                      stageIdx <= currentIdx || (isDone && stage === "done")
                        ? "bg-green-600"
                        : "bg-slate-700"
                    }`}
                  />
                )}
                <div
                  className={`flex flex-col items-center ${isClickable ? "cursor-pointer group" : ""}`}
                  onClick={() => isClickable && handleRollback(s)}
                  title={isClickable ? `"${STAGE_LABELS[s]}" 단계부터 재처리` : ""}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 transition-colors ${
                      isCompleted
                        ? "bg-green-600 text-white group-hover:bg-green-500"
                        : isCurrent
                          ? "bg-blue-600 text-white ring-2 ring-blue-400/50"
                          : "bg-slate-700 text-slate-500"
                    }`}
                  >
                    {isCompleted ? "\u2713" : isDone ? "" : i + 1}
                  </div>
                  <span
                    className={`text-[10px] mt-1.5 whitespace-nowrap transition-colors ${
                      isCompleted
                        ? "text-green-400 group-hover:text-green-300"
                        : isCurrent
                          ? "text-blue-300 font-semibold"
                          : "text-slate-500"
                    }`}
                  >
                    {isDone ? "완료" : STAGE_LABELS[s]}
                  </span>
                </div>
              </Fragment>
            );
          })}
        </div>
      </div>

      {/* 롤백 힌트 */}
      {currentIdx > 0 && stage !== "done" && (
        <p className="text-slate-600 text-[11px] text-center mb-6">
          초록색 단계를 클릭하면 해당 단계부터 재처리할 수 있습니다
        </p>
      )}
      {(currentIdx === 0 || stage === "done") && <div className="mb-6" />}

      {/* 에러 표시 */}
      {status.error && (
        <div className="bg-red-900/30 border border-red-700 rounded-xl p-6 mb-6">
          <p className="text-red-300 font-semibold mb-2">
            {STAGE_LABELS[status.error.stage] || status.error.stage} 단계에서 오류 발생
          </p>
          <pre className="text-red-400 text-sm whitespace-pre-wrap break-words bg-red-950/50 rounded-lg p-3">
            {status.error.message}
          </pre>
          <p className="text-slate-500 text-xs mt-3">
            아래 버튼으로 재실행하거나, 위 스테퍼에서 이전 단계를 클릭하여 롤백할 수 있습니다.
          </p>
        </div>
      )}

      {/* 처리 중 (폴링으로 진행률 표시) */}
      {isProcessing && status.processing && (
        <div className="bg-slate-800 rounded-xl p-6 mb-6">
          <p className="text-slate-300 text-sm font-medium mb-3">
            {STAGE_DESCRIPTIONS[status.processing.stage]?.title || STAGE_LABELS[status.processing.stage] || "처리 중"}
          </p>
          <ProgressBar
            progress={status.processing.progress}
            message={status.processing.message}
          />
          {status.processing.stage === "separate" && (
            <p className="text-slate-500 text-xs mt-3">
              배경음 제거는 음원 길이에 따라 수 분~수십 분 소요될 수 있습니다. 이 페이지를 벗어나도 처리는 계속됩니다.
            </p>
          )}
        </div>
      )}

      {/* 수동 실행 버튼: extract, separate, transcribe */}
      {!isProcessing &&
        stage in STAGE_ACTIONS &&
        stage !== "classify" &&
        stage !== "classifying" && (
          <div className="bg-slate-800 rounded-xl p-6">
            <div className="text-center mb-5">
              <p className="text-blue-300 font-semibold text-lg mb-2">
                {STAGE_DESCRIPTIONS[stage]?.title || STAGE_LABELS[stage]}
              </p>
              <p className="text-slate-400 text-sm max-w-md mx-auto">
                {STAGE_DESCRIPTIONS[stage]?.desc}
              </p>
            </div>
            {pendingStage && (
              <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-lg p-4 mb-4 text-center">
                <p className="text-yellow-300 text-sm mb-1">다른 영상의 배경음 제거가 진행 중입니다</p>
                <p className="text-slate-400 text-xs">완료되면 자동으로 시작됩니다. 이 페이지에서 기다려주세요.</p>
              </div>
            )}
            {stageError && !pendingStage && (
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 mb-4 text-red-300 text-sm text-center">
                {stageError}
              </div>
            )}
            <div className="text-center">
              <button
                className={`px-8 py-2.5 rounded-lg font-medium transition-colors ${
                  pendingStage
                    ? "bg-yellow-600/50 text-yellow-200 cursor-wait animate-pulse"
                    : "bg-blue-600 hover:bg-blue-700"
                }`}
                onClick={() => !pendingStage && handleRunStage(stage)}
                disabled={!!pendingStage}
              >
                {pendingStage ? "대기 중..." : `${STAGE_LABELS[stage]} 실행`}
              </button>
            </div>
          </div>
        )}

      {/* VAD 실행 (파라미터 설정 포함) */}
      {!isProcessing && stage === "vad" && (
        <div className="bg-slate-800 rounded-xl p-6">
          {/* 단계 설명 */}
          <div className="text-center mb-6">
            <p className="text-blue-300 font-semibold text-lg mb-2">
              {STAGE_DESCRIPTIONS.vad.title}
            </p>
            <p className="text-slate-400 text-sm max-w-lg mx-auto">
              {STAGE_DESCRIPTIONS.vad.desc}
            </p>
          </div>

          {/* 프리셋 */}
          <div className="max-w-lg mx-auto mb-6">
            <p className="text-slate-300 text-sm font-medium mb-2">빠른 설정</p>
            <div className="grid grid-cols-3 gap-2">
              {VAD_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  className="bg-slate-700/60 hover:bg-slate-700 border border-slate-600 hover:border-blue-500/50 rounded-lg px-3 py-2.5 text-left transition-colors"
                  onClick={() => setVadParams({ ...preset.params })}
                >
                  <p className="text-slate-200 text-xs font-medium">{preset.label}</p>
                  <p className="text-slate-500 text-[10px] mt-0.5">{preset.desc}</p>
                </button>
              ))}
            </div>
          </div>

          {/* 세부 파라미터 */}
          <div className="space-y-4 mb-6 max-w-lg mx-auto text-sm">
            <label className="block text-slate-400">
              <div className="flex items-center justify-between">
                <span>세그먼트 최소 길이 (초)</span>
                <input
                  type="number"
                  step="0.1"
                  className="w-24 bg-slate-700 rounded-lg px-3 py-1.5 text-white text-right focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={vadParams.min_segment_sec}
                  onChange={(e) => setVadParams({ ...vadParams, min_segment_sec: +e.target.value })}
                />
              </div>
              <p className="text-slate-500 text-xs mt-1">
                이보다 짧은 대사는 버립니다. 너무 짧은 음성은 학습에 부적합하므로 1~2초를 권장합니다.
              </p>
            </label>
            <label className="block text-slate-400">
              <div className="flex items-center justify-between">
                <span>세그먼트 최대 길이 (초)</span>
                <input
                  type="number"
                  step="0.5"
                  className="w-24 bg-slate-700 rounded-lg px-3 py-1.5 text-white text-right focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={vadParams.max_segment_sec}
                  onChange={(e) => setVadParams({ ...vadParams, max_segment_sec: +e.target.value })}
                />
              </div>
              <p className="text-slate-500 text-xs mt-1">
                이보다 긴 대사는 자동으로 분할됩니다. TTS 학습에는 5~15초가 적합합니다.
              </p>
            </label>
            <label className="block text-slate-400">
              <div className="flex items-center justify-between">
                <span>음성 감지 임계값</span>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  className="w-24 bg-slate-700 rounded-lg px-3 py-1.5 text-white text-right focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={vadParams.threshold}
                  onChange={(e) => setVadParams({ ...vadParams, threshold: +e.target.value })}
                />
              </div>
              <p className="text-slate-500 text-xs mt-1">
                낮추면 더 많은 대사를 잡아내고, 올리면 확실한 음성만 남깁니다.
                배경 잡음이 많으면 0.6~0.7로 올리고, 조용한 환경이면 0.3~0.5로 낮추세요.
              </p>
            </label>
            <label className="block text-slate-400">
              <div className="flex items-center justify-between">
                <span>대사 간 최소 무음 (ms)</span>
                <input
                  type="number"
                  step="10"
                  className="w-24 bg-slate-700 rounded-lg px-3 py-1.5 text-white text-right focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={vadParams.min_silence_ms}
                  onChange={(e) => setVadParams({ ...vadParams, min_silence_ms: +e.target.value })}
                />
              </div>
              <p className="text-slate-500 text-xs mt-1">
                대사와 대사 사이에 이 길이 이상의 무음이 있어야 별도 세그먼트로 분리합니다.
                값이 너무 작으면 한 문장이 여러 조각으로 쪼개지고, 너무 크면 서로 다른 대사가 하나로 합쳐집니다.
              </p>
            </label>
          </div>

          {/* 상황별 팁 */}
          <details className="max-w-lg mx-auto mb-6">
            <summary className="text-slate-400 text-xs cursor-pointer hover:text-slate-300 transition-colors">
              상황별 조정 가이드 보기
            </summary>
            <div className="bg-slate-700/40 rounded-lg p-4 mt-2 text-xs text-slate-400 space-y-1.5">
              <p>
                <span className="text-blue-400">두 화자가 빠르게 대화하는 영상</span>
                {" \u2192 최소 무음을 30~50ms로 낮추면 대사가 더 잘 분리됩니다."}
              </p>
              <p>
                <span className="text-blue-400">배경 음악/잡음이 남아있는 경우</span>
                {" \u2192 감지 임계값을 0.6~0.7로 올려 잡음을 걸러내세요."}
              </p>
              <p>
                <span className="text-blue-400">긴 독백이 많은 영상</span>
                {" \u2192 최대 길이를 10~15초로 설정하면 자연스러운 분할이 됩니다."}
              </p>
              <p>
                <span className="text-blue-400">짧은 감탄사/추임새가 많은 경우</span>
                {" \u2192 최소 길이를 1.5~2초로 올려 불필요한 세그먼트를 줄이세요."}
              </p>
            </div>
          </details>

          <div className="text-center">
            <button
              className="bg-blue-600 hover:bg-blue-700 px-8 py-2.5 rounded-lg font-medium transition-colors"
              onClick={async () => {
                if (!videoId) return;
                await videosApi.startVad(videoId, vadParams);
                setTimeout(refreshStatus, 500);
              }}
            >
              VAD 세그먼팅 실행
            </button>
            <p className="text-slate-500 text-xs mt-3">
              실행 후 에러 메시지가 나타나지 않으면 정상적으로 진행 중입니다. 잠시 기다려주세요.
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
        <div className="bg-gradient-to-br from-green-900/30 to-slate-800/30 border border-green-700/50 rounded-xl p-8 text-center">
          <div className="w-14 h-14 rounded-full bg-green-600 flex items-center justify-center text-xl mx-auto mb-4">
            {"\u2713"}
          </div>
          <p className="text-green-300 text-xl font-semibold mb-2">전처리 완료</p>
          <p className="text-slate-400 text-sm mb-6">이 영상의 모든 전처리 단계가 완료되었습니다.</p>

          {status.summary && status.summary.length > 0 && (
            <div className="max-w-sm mx-auto space-y-2 mb-6">
              {status.summary.map((s) => (
                <div
                  key={s.name}
                  className="flex justify-between items-center bg-green-900/20 rounded-lg px-4 py-2.5 text-sm"
                >
                  <span className={s.name === "discarded" ? "text-slate-500" : "text-slate-200"}>
                    {s.name === "discarded" ? "버림" : s.name}
                  </span>
                  <span className="text-slate-400 font-mono text-xs">
                    {s.count}개 / {Math.floor(s.total_duration / 60)}분 {Math.round(s.total_duration % 60)}초
                  </span>
                </div>
              ))}
            </div>
          )}

          <p className="text-slate-500 text-sm">
            대시보드로 돌아가 데이터셋을 생성하거나, 추가 영상을 업로드하여 데이터를 더 수집하세요.
          </p>
        </div>
      )}
    </div>
  );
}
