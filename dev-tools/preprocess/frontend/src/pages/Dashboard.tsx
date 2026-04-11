import { useEffect, useState, useRef } from "react";
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
      <h1 className="text-2xl font-bold mb-1">HayaKoe 전처리</h1>
      {dataDir && (
        <p className="text-slate-500 text-sm mb-6 font-mono">
          데이터 경로: {dataDir}
        </p>
      )}

      {/* 워크플로우 안내 */}
      <div className="bg-gradient-to-r from-blue-950/50 to-slate-800/50 border border-slate-700 rounded-xl p-5 mb-8">
        <p className="text-slate-200 text-sm font-semibold mb-1">작업 순서</p>
        <p className="text-slate-400 text-xs mb-2">
          음성 데이터를 수집하여 TTS 학습용 데이터셋을 만드는 도구입니다. 아래 순서대로 진행하세요.
        </p>
        <div className="flex items-center gap-1">
          {WORKFLOW_STEPS.map((step, i) => (
            <div key={step.n} className="flex items-center gap-1 flex-1">
              {i > 0 && <div className="w-6 h-px bg-slate-600 shrink-0" />}
              <div className="flex items-center gap-2 min-w-0">
                <span className="w-6 h-6 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold shrink-0">
                  {step.n}
                </span>
                <div className="min-w-0">
                  <p className="text-slate-200 text-xs font-medium truncate">{step.label}</p>
                  <p className="text-slate-500 text-[10px] truncate">{step.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-3 mb-4 text-red-200">
          {error}
        </div>
      )}

      {/* Step 1: 화자 등록 */}
      <section className="mb-5 bg-slate-800/60 border border-slate-700 rounded-xl p-5">
        <div className="flex items-center gap-2.5 mb-1">
          <span className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold shrink-0">1</span>
          <h2 className="text-lg font-semibold">화자 등록</h2>
        </div>
        <p className="text-slate-400 text-sm mb-4 ml-10">
          TTS 모델을 학습할 화자(목소리의 주인)를 등록합니다. 영상에 여러 사람이 등장하면, 나중에 각 음성 구간을 화자별로 분류하게 됩니다.
        </p>
        <div className="ml-10">
          <div className="flex gap-2 mb-3">
            <input
              className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-1.5 flex-1 focus:border-blue-500 focus:outline-none transition-colors"
              placeholder="화자 이름 (예: 홍길동)"
              value={newSpeaker}
              onChange={(e) => setNewSpeaker(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddSpeaker()}
            />
            <button
              className="bg-blue-600 hover:bg-blue-700 px-4 py-1.5 rounded-lg font-medium transition-colors"
              onClick={handleAddSpeaker}
            >
              추가
            </button>
          </div>
          {speakers.length === 0 ? (
            <div className="text-slate-500 text-sm bg-slate-700/30 rounded-lg p-4 text-center">
              등록된 화자가 없습니다. 위 입력란에 화자 이름을 입력하고 추가 버튼을 눌러주세요.
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {speakers.map((s) => (
                <span
                  key={s}
                  className="bg-slate-700 px-3 py-1.5 rounded-full text-sm flex items-center gap-2"
                >
                  {s}
                  <button
                    className="text-slate-400 hover:text-red-400 transition-colors"
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
      <section className={`mb-5 bg-slate-800/60 border border-slate-700 rounded-xl p-5 transition-opacity ${!hasSpeakers ? "opacity-50" : ""}`}>
        <div className="flex items-center gap-2.5 mb-1">
          <span className={`w-7 h-7 rounded-full text-white flex items-center justify-center text-xs font-bold shrink-0 ${hasSpeakers ? "bg-blue-600" : "bg-slate-600"}`}>2</span>
          <h2 className="text-lg font-semibold">영상 업로드</h2>
        </div>
        <p className="text-slate-400 text-sm mb-4 ml-10">
          학습에 사용할 영상 또는 오디오 파일을 업로드합니다.
          <br />
          여러 영상을 추가하여 데이터를 더 많이 수집할 수 있습니다. 업로드 후 영상을 클릭하면 전처리를 시작합니다.
        </p>
        <div className="ml-10">
          {!hasSpeakers ? (
            <div className="text-slate-500 text-sm bg-slate-700/30 rounded-lg p-4 text-center">
              먼저 위에서 화자를 등록해주세요.
            </div>
          ) : (
            <>
              <div className="mb-4">
                <label className="inline-flex items-center gap-2 bg-blue-600/20 border border-blue-500/30 rounded-lg px-4 py-2.5 cursor-pointer hover:bg-blue-600/30 transition-colors">
                  <span className="text-blue-300 text-sm font-medium">파일 선택</span>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="video/*,audio/*"
                    onChange={handleUpload}
                    className="hidden"
                  />
                </label>
                <span className="text-slate-500 text-xs ml-3">동영상 또는 오디오 파일</span>
              </div>

              {videos.length === 0 ? (
                <div className="text-slate-500 text-sm bg-slate-700/30 rounded-lg p-4 text-center">
                  업로드된 영상이 없습니다. 위 버튼으로 파일을 업로드해주세요.
                </div>
              ) : (
                <div className="space-y-2">
                  {videos.map((v) => (
                    <div
                      key={v.id}
                      className="flex items-center gap-4 bg-slate-700/40 hover:bg-slate-700/70 rounded-lg px-4 py-3 cursor-pointer transition-colors group"
                      onClick={() => navigate(`/video/${v.id}`)}
                    >
                      <span className="font-mono text-xs text-slate-500 w-20 shrink-0">{v.id}</span>
                      <span className="flex-1 text-sm truncate group-hover:text-white transition-colors">{v.filename}</span>
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-xs font-medium shrink-0 ${
                          v.stage === "done"
                            ? "bg-green-900/60 text-green-300"
                            : v.stage.startsWith("processing")
                              ? "bg-yellow-900/60 text-yellow-300 animate-pulse"
                              : "bg-slate-600 text-slate-300"
                        }`}
                      >
                        {stageLabel(v.stage)}
                      </span>
                      <button
                        className="text-slate-500 hover:text-red-400 text-xs shrink-0 transition-colors"
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
        <section className="mb-5 bg-slate-800/60 border border-slate-700 rounded-xl p-5">
          <h2 className="text-lg font-semibold mb-1">수집 현황</h2>
          <p className="text-slate-400 text-sm mb-1">
            전처리가 완료된 영상에서 수집된 화자별 음성 데이터입니다.
          </p>
          <p className="text-slate-500 text-xs mb-4">
            빠르게 테스트하려면 <span className="text-slate-300">약 10분</span>, 안정적인 품질을 원하면 <span className="text-slate-300">20~45분</span> 정도의 깨끗한 음성 데이터를 준비하면 좋습니다.
          </p>
          <div className="grid grid-cols-2 gap-3">
            {speakerSummary
              .filter((s) => s.count > 0)
              .map((s) => (
                <div
                  key={s.name}
                  className="bg-slate-700/50 rounded-lg px-4 py-3"
                >
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-200 font-medium">{s.name}</span>
                    <span className="text-slate-400 text-sm font-mono">{s.count}개</span>
                  </div>
                  <p className="text-xs text-slate-500">
                    총 {formatDuration(s.total_duration)}
                  </p>
                </div>
              ))}
          </div>
        </section>
      )}

      {/* Step 4: 데이터셋 생성 */}
      <section className={`bg-slate-800/60 border border-slate-700 rounded-xl p-5 transition-opacity ${!hasDoneVideos ? "opacity-50" : ""}`}>
        <div className="flex items-center gap-2.5 mb-1">
          <span className={`w-7 h-7 rounded-full text-white flex items-center justify-center text-xs font-bold shrink-0 ${hasDoneVideos ? "bg-blue-600" : "bg-slate-600"}`}>4</span>
          <h2 className="text-lg font-semibold">데이터셋 생성</h2>
        </div>
        <p className="text-slate-400 text-sm mb-4 ml-10">
          수집된 음성 데이터를 TTS 학습에 사용할 수 있는 데이터셋으로 변환합니다. 모든 영상의 전처리를 완료한 후 실행하세요.
        </p>
        <div className="ml-10">
          {!hasDoneVideos ? (
            <div className="text-slate-500 text-sm bg-slate-700/30 rounded-lg p-4 text-center">
              전처리가 완료된 영상이 없습니다. 먼저 영상을 업로드하고 전처리를 진행해주세요.
            </div>
          ) : (
            <>
              <div className="bg-slate-700/30 rounded-lg p-4 mb-4">
                <label className="text-sm text-slate-300 block mb-1">
                  검증(Validation) 데이터 비율
                </label>
                <p className="text-xs text-slate-500 mb-3">
                  전체 데이터 중 학습 품질 검증에 사용할 비율입니다. 일반적으로 10~20%가 적절합니다. 나머지는 학습(Training)에 사용됩니다.
                </p>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min="0.01"
                    max="0.5"
                    step="0.01"
                    className="flex-1 accent-blue-500"
                    value={valRatio}
                    onChange={(e) => setValRatio(+e.target.value)}
                  />
                  <span className="text-white font-mono text-sm w-12 text-right">
                    {Math.round(valRatio * 100)}%
                  </span>
                </div>
              </div>

              <button
                className="bg-green-600 hover:bg-green-700 px-6 py-2.5 rounded-lg font-medium disabled:opacity-50 transition-colors"
                onClick={handleBuildDataset}
                disabled={building}
              >
                {building ? "생성 중..." : "데이터셋 생성"}
              </button>
              <p className="text-slate-500 text-xs mt-2">
                이미 데이터셋이 있는 경우, 기존 데이터셋을 삭제하고 새로 생성합니다.
              </p>

              {datasetResult && (
                <div className="bg-green-900/20 border border-green-800/50 rounded-lg p-5 mt-4">
                  <p className="text-green-300 font-semibold mb-1">데이터셋 생성 완료</p>
                  <p className="text-slate-500 text-xs mb-4 font-mono">{datasetResult.dataset_dir}</p>
                  {Object.entries(datasetResult.speakers).map(([name, s]) => (
                    <div key={name} className="mb-3 last:mb-0">
                      <p className="text-slate-200 font-semibold mb-1">{name}</p>
                      <div className="grid grid-cols-3 gap-2 text-xs text-slate-400 bg-slate-700/50 rounded-lg p-3">
                        <div>
                          <p className="text-slate-500 mb-0.5">전체</p>
                          <p>{s.count}개 / {formatDuration(s.duration)}</p>
                        </div>
                        <div>
                          <p className="text-slate-500 mb-0.5">Train</p>
                          <p>{s.train_count}개 / {formatDuration(s.train_duration)}</p>
                        </div>
                        <div>
                          <p className="text-slate-500 mb-0.5">Val</p>
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
