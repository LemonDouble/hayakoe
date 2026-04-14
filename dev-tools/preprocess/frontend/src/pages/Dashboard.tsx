import { Fragment, useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router";
import api from "../api/client";
import * as speakersApi from "../api/speakers";
import * as videosApi from "../api/videos";
import * as datasetApi from "../api/dataset";
import type { DatasetResult } from "../api/dataset";
import type { SpeakerSummary } from "../api/speakers";
import type { VideoInfo } from "../api/videos";

const STAGE_LABELS: Record<string, string> = {
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

function stageLabel(stage: string) {
  if (stage.startsWith("processing:")) {
    const s = stage.split(":")[1];
    return `처리 중 (${s})`;
  }
  return STAGE_LABELS[stage] || stage;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}분 ${s}초` : `${s}초`;
}

const WORKFLOW_STEPS = [
  { n: "1", label: "화자 등록", desc: "목소리 주인 등록" },
  { n: "2", label: "영상 업로드", desc: "학습 소스 준비" },
  { n: "3", label: "영상별 전처리", desc: "6단계 파이프라인" },
  { n: "4", label: "데이터셋 생성", desc: "학습 데이터 출력" },
  { n: "5", label: "CLI 학습", desc: "남은 학습은 CLI에서" },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const [speakers, setSpeakers] = useState<string[]>([]);
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [speakerSummary, setSpeakerSummary] = useState<SpeakerSummary[]>([]);
  const [newSpeaker, setNewSpeaker] = useState("");
  const [error, setError] = useState("");
  const [valRatio, setValRatio] = useState(0.1);
  const [datasetResult, setDatasetResult] = useState<DatasetResult | null>(null);
  const [building, setBuilding] = useState(false);
  const [dataDir, setDataDir] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    const [s, v] = await Promise.all([
      speakersApi.listSpeakers(),
      videosApi.listVideos(),
    ]);
    setSpeakers(s);
    setVideos(v);
    // 화자 요약은 별도 (에러 무시 — 세그먼트 없을 수 있음)
    speakersApi.getSummary().then(setSpeakerSummary).catch(() => {});
  };

  useEffect(() => {
    refresh();
    api.get("/info").then(({ data }) => setDataDir(data.data_dir));
  }, []);

  const handleAddSpeaker = async () => {
    if (!newSpeaker.trim()) return;
    try {
      const s = await speakersApi.addSpeaker(newSpeaker.trim());
      setSpeakers(s);
      setNewSpeaker("");
      setError("");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  };

  const handleDeleteSpeaker = async (name: string) => {
    if (!confirm(`화자 "${name}"을(를) 삭제하시겠습니까?`)) return;
    const s = await speakersApi.deleteSpeaker(name);
    setSpeakers(s);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await videosApi.uploadVideo(file);
    await refresh();
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleDeleteVideo = async (id: string) => {
    if (!confirm(`영상 ${id}를 삭제하시겠습니까?`)) return;
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
    } catch (e: unknown) {
      const detail =
        (e as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || String(e);
      setError(detail);
    } finally {
      setBuilding(false);
    }
  };

  const hasSpeakers = speakers.length > 0;
  const hasDoneVideos = videos.some((v) => v.stage === "done");

  return (
    <div className="max-w-4xl mx-auto p-6 pb-16">
      {/* 헤더 */}
      <h1 className="font-display text-3xl font-bold mb-1 text-fg tracking-tight">
        <span className="text-primary">HayaKoe</span> 전처리
      </h1>
      {dataDir && (
        <p className="text-fg-dim text-xs mb-8 font-mono">
          데이터 경로: {dataDir}
        </p>
      )}

      {/* 워크플로우 안내 */}
      <div className="bg-primary/[0.08] border border-primary/25 rounded-xl p-5 mb-8">
        <p className="text-primary text-[11px] font-semibold uppercase tracking-[1.5px] mb-1.5 font-display">WORKFLOW</p>
        <p className="text-fg-muted text-xs mb-4">
          음성 데이터를 수집하여 TTS 학습용 데이터셋을 만드는 도구입니다. 아래 순서대로 진행하세요.
        </p>
        <div className="flex items-center gap-3">
          {WORKFLOW_STEPS.map((step, i) => (
            <Fragment key={step.n}>
              {i > 0 && (
                <span
                  className="text-primary/60 text-base shrink-0 font-bold select-none"
                  aria-hidden="true"
                >
                  &rarr;
                </span>
              )}
              <div className="flex items-center justify-center gap-2 flex-1 min-w-0">
                <span className="w-6 h-6 rounded-full bg-primary text-canvas flex items-center justify-center text-xs font-bold shrink-0">
                  {step.n}
                </span>
                <div className="min-w-0">
                  <p className="text-fg text-xs font-semibold truncate">{step.label}</p>
                  <p className="text-fg-dim text-[10px] truncate">{step.desc}</p>
                </div>
              </div>
            </Fragment>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-error/10 border border-error/25 rounded-lg p-3 mb-4 text-error text-sm">
          {error}
        </div>
      )}

      {/* Step 1: 화자 등록 */}
      <section className="mb-5 bg-surface border border-line rounded-xl p-6">
        <div className="flex items-center gap-2.5 mb-1">
          <span className="w-7 h-7 rounded-full bg-primary text-canvas flex items-center justify-center text-xs font-bold shrink-0">1</span>
          <h2 className="font-display text-xl font-bold text-fg">화자 등록</h2>
        </div>
        <p className="text-fg-muted text-sm mb-4 ml-10 leading-relaxed">
          TTS 모델을 학습할 화자(목소리의 주인)를 등록합니다. 영상에 여러 사람이 등장하면, 나중에 각 음성 구간을 화자별로 분류하게 됩니다.
        </p>
        <div className="ml-10">
          <div className="flex gap-2 mb-3">
            <input
              className="bg-surface border border-line rounded-lg px-4 py-2 flex-1 text-fg placeholder:text-fg-dim focus:border-primary/50 focus:outline-none transition-colors"
              placeholder="화자 이름 (예: 홍길동)"
              value={newSpeaker}
              onChange={(e) => setNewSpeaker(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddSpeaker()}
            />
            <button
              className="bg-primary hover:bg-primary-hover text-canvas px-5 py-2 rounded-lg font-semibold text-sm transition-colors"
              onClick={handleAddSpeaker}
            >
              추가
            </button>
          </div>
          {speakers.length === 0 ? (
            <div className="text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center">
              등록된 화자가 없습니다. 위 입력란에 화자 이름을 입력하고 추가 버튼을 눌러주세요.
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {speakers.map((s) => (
                <span
                  key={s}
                  className="bg-primary/[0.12] border border-primary/25 text-primary px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-2"
                >
                  {s}
                  <button
                    className="text-primary/60 hover:text-error transition-colors"
                    onClick={() => handleDeleteSpeaker(s)}
                  >
                    &times;
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Step 2: 영상 업로드 */}
      <section className={`mb-5 bg-surface border border-line rounded-xl p-6 transition-opacity ${!hasSpeakers ? "opacity-40" : ""}`}>
        <div className="flex items-center gap-2.5 mb-1">
          <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${hasSpeakers ? "bg-primary text-canvas" : "bg-line text-fg-dim"}`}>2</span>
          <h2 className="font-display text-xl font-bold text-fg">영상 업로드</h2>
        </div>
        <p className="text-fg-muted text-sm mb-4 ml-10 leading-relaxed">
          학습에 사용할 영상 또는 오디오 파일을 업로드합니다.
          <br />
          여러 영상을 추가하여 데이터를 더 많이 수집할 수 있습니다. 업로드 후 영상을 클릭하면 전처리를 시작합니다.
        </p>
        <div className="ml-10">
          {!hasSpeakers ? (
            <div className="text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center">
              먼저 위에서 화자를 등록해주세요.
            </div>
          ) : (
            <>
              <div className="mb-4">
                <label className="inline-flex items-center gap-2 bg-secondary/[0.12] border border-secondary/25 rounded-lg px-4 py-2.5 cursor-pointer hover:bg-secondary/20 transition-colors">
                  <span className="text-secondary text-sm font-semibold">파일 선택</span>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="video/*,audio/*"
                    onChange={handleUpload}
                    className="hidden"
                  />
                </label>
                <span className="text-fg-dim text-xs ml-3">동영상 또는 오디오 파일</span>
              </div>

              {videos.length === 0 ? (
                <div className="text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center">
                  업로드된 영상이 없습니다. 위 버튼으로 파일을 업로드해주세요.
                </div>
              ) : (
                <div className="space-y-2">
                  {videos.map((v) => (
                    <div
                      key={v.id}
                      className="flex items-center gap-4 bg-canvas border border-line hover:border-primary/25 rounded-lg px-4 py-3 cursor-pointer transition-colors group"
                      onClick={() => navigate(`/video/${v.id}`)}
                    >
                      <span className="font-mono text-xs text-fg-dim w-20 shrink-0">{v.id}</span>
                      <span className="flex-1 text-sm text-fg truncate">{v.filename}</span>
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold shrink-0 border ${
                          v.stage === "done"
                            ? "bg-success/10 border-success/25 text-success"
                            : v.stage.startsWith("processing")
                              ? "bg-warning/10 border-warning/25 text-warning animate-pulse"
                              : "bg-surface-2 border-line text-fg-muted"
                        }`}
                      >
                        {stageLabel(v.stage)}
                      </span>
                      <button
                        className="text-fg-dim hover:text-error text-xs shrink-0 transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteVideo(v.id);
                        }}
                      >
                        삭제
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </section>

      {/* 수집 현황 */}
      {speakerSummary.some((s) => s.count > 0) && (
        <section className="mb-5 bg-surface border border-line rounded-xl p-6">
          <h2 className="font-display text-xl font-bold text-fg mb-1">수집 현황</h2>
          <p className="text-fg-muted text-sm mb-1">
            전처리가 완료된 영상에서 수집된 화자별 음성 데이터입니다.
          </p>
          <p className="text-fg-dim text-xs mb-4">
            빠르게 테스트하려면 <span className="text-fg">약 10분</span>, 안정적인 품질을 원하면 <span className="text-fg">20~45분</span> 정도의 깨끗한 음성 데이터를 준비하면 좋습니다.
          </p>
          <div className="grid grid-cols-2 gap-3">
            {speakerSummary
              .filter((s) => s.count > 0)
              .map((s) => (
                <div
                  key={s.name}
                  className="bg-canvas border border-line rounded-lg px-4 py-3"
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-fg font-semibold text-sm">{s.name}</span>
                    <span className="text-primary text-sm font-mono font-semibold">{s.count}개</span>
                  </div>
                  <p className="text-xs text-fg-dim">
                    총 {formatDuration(s.total_duration)}
                  </p>
                </div>
              ))}
          </div>
        </section>
      )}

      {/* Step 4: 데이터셋 생성 */}
      <section className={`bg-surface border border-line rounded-xl p-6 transition-opacity ${!hasDoneVideos ? "opacity-40" : ""}`}>
        <div className="flex items-center gap-2.5 mb-1">
          <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${hasDoneVideos ? "bg-primary text-canvas" : "bg-line text-fg-dim"}`}>4</span>
          <h2 className="font-display text-xl font-bold text-fg">데이터셋 생성</h2>
        </div>
        <p className="text-fg-muted text-sm mb-4 ml-10 leading-relaxed">
          수집된 음성 데이터를 TTS 학습에 사용할 수 있는 데이터셋으로 변환합니다. 모든 영상의 전처리를 완료한 후 실행하세요.
        </p>
        <div className="ml-10">
          {!hasDoneVideos ? (
            <div className="text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center">
              전처리가 완료된 영상이 없습니다. 먼저 영상을 업로드하고 전처리를 진행해주세요.
            </div>
          ) : (
            <>
              <div className="bg-canvas border border-line rounded-lg p-4 mb-4">
                <label className="text-sm text-fg font-semibold block mb-1">
                  검증(Validation) 데이터 비율
                </label>
                <p className="text-xs text-fg-dim mb-3 leading-relaxed">
                  전체 데이터 중 학습 품질 검증에 사용할 비율입니다. 일반적으로 10~20%가 적절합니다. 나머지는 학습(Training)에 사용됩니다.
                </p>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min="0.01"
                    max="0.5"
                    step="0.01"
                    className="flex-1 accent-primary"
                    value={valRatio}
                    onChange={(e) => setValRatio(+e.target.value)}
                  />
                  <span className="text-primary font-mono font-semibold text-sm w-12 text-right">
                    {Math.round(valRatio * 100)}%
                  </span>
                </div>
              </div>

              <button
                className="bg-primary hover:bg-primary-hover text-canvas px-6 py-2.5 rounded-lg font-semibold text-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                onClick={handleBuildDataset}
                disabled={building}
              >
                {building ? "생성 중..." : "데이터셋 생성"}
              </button>
              <p className="text-fg-dim text-xs mt-2">
                이미 데이터셋이 있는 경우, 기존 데이터셋을 삭제하고 새로 생성합니다.
              </p>

              {datasetResult && (
                <div className="bg-success/[0.06] border border-success/25 rounded-lg p-5 mt-4">
                  <p className="text-success font-semibold mb-1">데이터셋 생성 완료</p>
                  <p className="text-fg-dim text-xs mb-4 font-mono">{datasetResult.dataset_dir}</p>
                  {Object.entries(datasetResult.speakers).map(([name, s]) => (
                    <div key={name} className="mb-3 last:mb-0">
                      <p className="text-fg font-semibold mb-1">{name}</p>
                      <div className="grid grid-cols-3 gap-2 text-xs text-fg-muted bg-canvas border border-line rounded-lg p-3">
                        <div>
                          <p className="text-fg-dim mb-0.5">전체</p>
                          <p>{s.count}개 / {formatDuration(s.duration)}</p>
                        </div>
                        <div>
                          <p className="text-fg-dim mb-0.5">Train</p>
                          <p>{s.train_count}개 / {formatDuration(s.train_duration)}</p>
                        </div>
                        <div>
                          <p className="text-fg-dim mb-0.5">Val</p>
                          <p>{s.val_count}개 / {formatDuration(s.val_duration)}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </section>
    </div>
  );
}
