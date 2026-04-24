import { Fragment, useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router";
import api from "../api/client";
import * as speakersApi from "../api/speakers";
import * as videosApi from "../api/videos";
import * as datasetApi from "../api/dataset";
import type { DatasetResult } from "../api/dataset";
import type { SpeakerSummary } from "../api/speakers";
import type { VideoInfo } from "../api/videos";
import { t } from "../i18n";

const STAGE_KEYS: Record<string, string> = {
  extract: "dashboard.stages.extract",
  separate: "dashboard.stages.separate",
  vad: "dashboard.stages.vad",
  classify: "dashboard.stages.classify",
  classifying: "dashboard.stages.classifying",
  transcribe: "dashboard.stages.transcribe",
  review: "dashboard.stages.review",
  done: "dashboard.stages.done",
  empty: "dashboard.stages.empty",
};

function stageLabel(stage: string) {
  if (stage.startsWith("processing:")) {
    const s = stage.split(":")[1];
    return t("dashboard.stages.processing", { stage: s });
  }
  return STAGE_KEYS[stage] ? t(STAGE_KEYS[stage]) : stage;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? t("dashboard.format_duration.min_sec", { m, s }) : t("dashboard.format_duration.sec", { s });
}

const WORKFLOW_STEPS = [
  { n: "1", labelKey: "dashboard.workflow.step1.label", descKey: "dashboard.workflow.step1.desc" },
  { n: "2", labelKey: "dashboard.workflow.step2.label", descKey: "dashboard.workflow.step2.desc" },
  { n: "3", labelKey: "dashboard.workflow.step3.label", descKey: "dashboard.workflow.step3.desc" },
  { n: "4", labelKey: "dashboard.workflow.step4.label", descKey: "dashboard.workflow.step4.desc" },
  { n: "5", labelKey: "dashboard.workflow.step5.label", descKey: "dashboard.workflow.step5.desc" },
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
    if (!confirm(t("dashboard.speaker.confirm_delete", { name }))) return;
    const s = await speakersApi.deleteSpeaker(name);
    setSpeakers(s);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.name.toLowerCase() === "extracted.wav") {
      alert(t("dashboard.upload.filename_conflict"));
      if (fileRef.current) fileRef.current.value = "";
      return;
    }
    await videosApi.uploadVideo(file);
    await refresh();
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleDeleteVideo = async (id: string) => {
    if (!confirm(t("dashboard.upload.confirm_delete", { id }))) return;
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
        <span className="text-primary">HayaKoe</span> {t("dashboard.title")}
      </h1>
      {dataDir && (
        <p className="text-fg-dim text-xs mb-8 font-mono">
          {t("dashboard.data_path", { path: dataDir })}
        </p>
      )}

      {/* 워크플로우 안내 */}
      <div className="bg-primary/[0.08] border border-primary/25 rounded-xl p-5 mb-8">
        <p className="text-primary text-[11px] font-semibold uppercase tracking-[1.5px] mb-1.5 font-display">WORKFLOW</p>
        <p className="text-fg-muted text-xs mb-4">
          {t("dashboard.workflow.description")}
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
                  <p className="text-fg text-xs font-semibold truncate">{t(step.labelKey)}</p>
                  <p className="text-fg-dim text-[10px] truncate">{t(step.descKey)}</p>
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
          <h2 className="font-display text-xl font-bold text-fg">{t("dashboard.speaker.title")}</h2>
        </div>
        <p className="text-fg-muted text-sm mb-4 ml-10 leading-relaxed">
          {t("dashboard.speaker.description")}
        </p>
        <div className="ml-10">
          <div className="flex gap-2 mb-3">
            <input
              className="bg-surface border border-line rounded-lg px-4 py-2 flex-1 text-fg placeholder:text-fg-dim focus:border-primary/50 focus:outline-none transition-colors"
              placeholder={t("dashboard.speaker.placeholder")}
              value={newSpeaker}
              onChange={(e) => setNewSpeaker(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddSpeaker()}
            />
            <button
              className="bg-primary hover:bg-primary-hover text-canvas px-5 py-2 rounded-lg font-semibold text-sm transition-colors"
              onClick={handleAddSpeaker}
            >
              {t("dashboard.speaker.add")}
            </button>
          </div>
          {speakers.length === 0 ? (
            <div className="text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center">
              {t("dashboard.speaker.empty")}
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
          <h2 className="font-display text-xl font-bold text-fg">{t("dashboard.upload.title")}</h2>
        </div>
        <p className="text-fg-muted text-sm mb-4 ml-10 leading-relaxed">
          {t("dashboard.upload.description_line1")}
          <br />
          {t("dashboard.upload.description_line2")}
        </p>
        <div className="ml-10">
          {!hasSpeakers ? (
            <div className="text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center">
              {t("dashboard.upload.need_speaker")}
            </div>
          ) : (
            <>
              <div className="mb-4">
                <label className="inline-flex items-center gap-2 bg-secondary/[0.12] border border-secondary/25 rounded-lg px-4 py-2.5 cursor-pointer hover:bg-secondary/20 transition-colors">
                  <span className="text-secondary text-sm font-semibold">{t("dashboard.upload.select_file")}</span>
                  <input
                    ref={fileRef}
                    type="file"
                    accept="video/*,audio/*"
                    onChange={handleUpload}
                    className="hidden"
                  />
                </label>
                <span className="text-fg-dim text-xs ml-3">{t("dashboard.upload.file_hint")}</span>
              </div>

              {videos.length === 0 ? (
                <div className="text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center">
                  {t("dashboard.upload.empty")}
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
                        {t("dashboard.upload.delete")}
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
          <h2 className="font-display text-xl font-bold text-fg mb-1">{t("dashboard.summary.title")}</h2>
          <p className="text-fg-muted text-sm mb-1">
            {t("dashboard.summary.description")}
          </p>
          <p className="text-fg-dim text-xs mb-4">
            {t("dashboard.summary.hint")}
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
                    <span className="text-primary text-sm font-mono font-semibold">{t("dashboard.summary.count_unit", { count: s.count })}</span>
                  </div>
                  <p className="text-xs text-fg-dim">
                    {t("dashboard.summary.total_duration", { duration: formatDuration(s.total_duration) })}
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
          <h2 className="font-display text-xl font-bold text-fg">{t("dashboard.dataset.title")}</h2>
        </div>
        <p className="text-fg-muted text-sm mb-4 ml-10 leading-relaxed">
          {t("dashboard.dataset.description")}
        </p>
        <div className="ml-10">
          {!hasDoneVideos ? (
            <div className="text-fg-dim text-sm border border-dashed border-line rounded-lg p-5 text-center">
              {t("dashboard.dataset.empty")}
            </div>
          ) : (
            <>
              <div className="bg-canvas border border-line rounded-lg p-4 mb-4">
                <label className="text-sm text-fg font-semibold block mb-1">
                  {t("dashboard.dataset.val_ratio_label")}
                </label>
                <p className="text-xs text-fg-dim mb-3 leading-relaxed">
                  {t("dashboard.dataset.val_ratio_desc")}
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
                {building ? t("dashboard.dataset.building") : t("dashboard.dataset.build")}
              </button>
              <p className="text-fg-dim text-xs mt-2">
                {t("dashboard.dataset.overwrite_note")}
              </p>

              {datasetResult && (
                <div className="bg-success/[0.06] border border-success/25 rounded-lg p-5 mt-4">
                  <p className="text-success font-semibold mb-1">{t("dashboard.dataset.result_title")}</p>
                  <p className="text-fg-dim text-xs mb-4 font-mono">{datasetResult.dataset_dir}</p>
                  {Object.entries(datasetResult.speakers).map(([name, s]) => (
                    <div key={name} className="mb-3 last:mb-0">
                      <p className="text-fg font-semibold mb-1">{name}</p>
                      <div className="grid grid-cols-3 gap-2 text-xs text-fg-muted bg-canvas border border-line rounded-lg p-3">
                        <div>
                          <p className="text-fg-dim mb-0.5">{t("dashboard.dataset.result_total")}</p>
                          <p>{t("dashboard.summary.count_unit", { count: s.count })} / {formatDuration(s.duration)}</p>
                        </div>
                        <div>
                          <p className="text-fg-dim mb-0.5">{t("dashboard.dataset.result_train")}</p>
                          <p>{t("dashboard.summary.count_unit", { count: s.train_count })} / {formatDuration(s.train_duration)}</p>
                        </div>
                        <div>
                          <p className="text-fg-dim mb-0.5">{t("dashboard.dataset.result_val")}</p>
                          <p>{t("dashboard.summary.count_unit", { count: s.val_count })} / {formatDuration(s.val_duration)}</p>
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
