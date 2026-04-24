import { useCallback, useEffect, useRef, useState } from "react";
import * as reviewApi from "../api/review";
import type { TranscriptionEntry } from "../api/review";
import { t } from "../i18n";

interface Props {
  videoId: string;
  onDone: () => void;
}

export default function ReviewEditor({ videoId, onDone }: Props) {
  const [entries, setEntries] = useState<TranscriptionEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [playingFile, setPlayingFile] = useState<string | null>(null);
  const [editingFile, setEditingFile] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [saving, setSaving] = useState(false);
  const [activeSpeaker, setActiveSpeaker] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const load = useCallback(async () => {
    const data = await reviewApi.getTranscriptions(videoId);
    setEntries(data.entries);
    setLoading(false);
  }, [videoId]);

  useEffect(() => {
    load();
  }, [load]);

  // 화자 목록
  const speakerList = [...new Set(entries.map((e) => e.speaker))];

  // 첫 화자 자동 선택
  useEffect(() => {
    if (activeSpeaker === null && speakerList.length > 0) {
      setActiveSpeaker(speakerList[0]);
    }
  }, [speakerList, activeSpeaker]);

  // 현재 화자의 엔트리
  const filtered = activeSpeaker
    ? entries.filter((e) => e.speaker === activeSpeaker)
    : entries;

  // 화자별 통계
  const speakerCounts = new Map<string, number>();
  for (const e of entries) {
    speakerCounts.set(e.speaker, (speakerCounts.get(e.speaker) || 0) + 1);
  }

  const play = (entry: TranscriptionEntry) => {
    if (!audioRef.current) return;
    const url = `/api/media/videos/${videoId}/segments/${entry.speaker}/${entry.file}`;
    audioRef.current.src = url;
    audioRef.current.play().catch(() => {});
    setPlayingFile(entry.file);
  };

  const handleAudioEnded = () => setPlayingFile(null);

  const startEdit = (entry: TranscriptionEntry) => {
    setEditingFile(entry.file);
    setEditText(entry.text);
  };

  const cancelEdit = () => {
    setEditingFile(null);
    setEditText("");
  };

  const saveEdit = async () => {
    if (editingFile === null) return;
    setSaving(true);
    await reviewApi.editTranscription(videoId, editingFile, editText);
    setEntries((prev) =>
      prev.map((e) => (e.file === editingFile ? { ...e, text: editText } : e))
    );
    setEditingFile(null);
    setEditText("");
    setSaving(false);
  };

  const handleDelete = async (entry: TranscriptionEntry) => {
    if (!confirm(t("review.confirm_delete", { file: entry.file }))) return;
    await reviewApi.deleteTranscription(videoId, entry.file);
    setEntries((prev) => prev.filter((e) => e.file !== entry.file));
    if (editingFile === entry.file) cancelEdit();
  };

  const handleDone = async () => {
    if (!confirm(t("review.confirm_done"))) return;
    await reviewApi.markReviewDone(videoId);
    onDone();
  };

  if (loading) return <div className="p-6 text-fg-muted">{t("review.loading")}</div>;

  return (
    <div className="bg-surface border border-line rounded-xl p-6">
      <audio ref={audioRef} onEnded={handleAudioEnded} />

      {/* 안내 배너 */}
      <div className="bg-primary/[0.08] border border-primary/25 rounded-lg p-4 mb-5">
        <p className="text-primary text-[11px] font-bold uppercase tracking-[1.5px] mb-1.5 font-display">{t("review.title")}</p>
        <p className="text-fg-muted text-xs leading-relaxed mb-2">
          {t("review.description")}
        </p>
        <ul className="text-xs text-fg-dim space-y-0.5 list-disc list-inside marker:text-primary">
          <li>{t("review.hint_play")}</li>
          <li>{t("review.hint_edit")}</li>
          <li>{t("review.hint_delete")}</li>
        </ul>
      </div>

      {/* 헤더 */}
      <div className="flex justify-between items-center mb-4">
        <p className="text-sm text-fg-muted">
          {t("review.total_segments", { count: entries.length })}
        </p>
        <button
          className="bg-primary hover:bg-primary-hover text-canvas px-5 py-2 rounded-lg text-sm font-semibold transition-colors"
          onClick={handleDone}
        >
          {t("review.done")}
        </button>
      </div>

      {/* 화자 탭 */}
      <div className="flex gap-0 mb-4 overflow-x-auto border-b border-line">
        {speakerList.map((speaker) => (
          <button
            key={speaker}
            className={`px-5 py-2.5 text-sm font-semibold transition-colors shrink-0 border-b-2 -mb-px ${
              activeSpeaker === speaker
                ? "text-primary border-primary"
                : "text-fg-dim border-transparent hover:text-fg-muted"
            }`}
            onClick={() => setActiveSpeaker(speaker)}
          >
            {speaker}
            <span className="ml-1.5 text-xs opacity-70 font-mono">
              ({speakerCounts.get(speaker) || 0})
            </span>
          </button>
        ))}
      </div>

      {/* 세그먼트 리스트 */}
      <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
        {filtered.length === 0 ? (
          <div className="text-fg-dim text-sm text-center py-8">
            {t("review.empty_speaker")}
          </div>
        ) : (
          filtered.map((entry) => (
            <div
              key={entry.file}
              className={`flex items-start gap-3 p-3 rounded-lg border transition-colors ${
                playingFile === entry.file
                  ? "bg-primary/[0.08] border-primary/25"
                  : "bg-canvas border-line hover:border-line-strong"
              }`}
            >
              {/* 재생 */}
              <button
                className={`mt-0.5 shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs transition-colors border ${
                  playingFile === entry.file
                    ? "bg-primary border-primary text-canvas"
                    : "bg-surface border-line text-fg-muted hover:border-primary/50 hover:text-primary"
                }`}
                onClick={() => play(entry)}
                title={t("review.hint_play")}
              >
                {playingFile === entry.file ? "⏹" : "▶"}
              </button>

              {/* 내용 */}
              <div className="flex-1 min-w-0">
                <p className="text-[10px] text-fg-dim mb-1 font-mono">{entry.file}</p>
                {editingFile === entry.file ? (
                  <div className="flex gap-2">
                    <input
                      type="text"
                      className="flex-1 bg-surface border border-line rounded-lg px-3 py-1.5 text-sm text-fg focus:outline-none focus:border-primary/50 transition-colors"
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") saveEdit();
                        if (e.key === "Escape") cancelEdit();
                      }}
                      autoFocus
                    />
                    <button
                      className="bg-primary hover:bg-primary-hover text-canvas px-3 py-1 rounded-md text-xs font-semibold transition-colors disabled:opacity-40"
                      onClick={saveEdit}
                      disabled={saving}
                    >
                      {t("review.save")}
                    </button>
                    <button
                      className="bg-transparent border border-line hover:border-line-strong text-fg-muted hover:text-fg px-3 py-1 rounded-md text-xs font-semibold transition-colors"
                      onClick={cancelEdit}
                    >
                      {t("review.cancel")}
                    </button>
                  </div>
                ) : (
                  <p
                    className="text-sm text-fg cursor-pointer hover:text-primary transition-colors"
                    onClick={() => startEdit(entry)}
                    title={t("review.hint_edit")}
                  >
                    {entry.text || <span className="text-fg-dim italic">{t("review.empty_text")}</span>}
                  </p>
                )}
              </div>

              {/* 삭제 */}
              <button
                className="text-fg-dim hover:text-error text-sm shrink-0 mt-0.5 transition-colors"
                onClick={() => handleDelete(entry)}
                title={t("review.hint_delete")}
              >
                &times;
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
