import { useCallback, useEffect, useRef, useState } from "react";
import * as reviewApi from "../api/review";
import type { TranscriptionEntry } from "../api/review";

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
    if (!confirm(`"${entry.file}" 전사를 삭제하시겠습니까?`)) return;
    await reviewApi.deleteTranscription(videoId, entry.file);
    setEntries((prev) => prev.filter((e) => e.file !== entry.file));
    if (editingFile === entry.file) cancelEdit();
  };

  const handleDone = async () => {
    if (!confirm("검토를 완료하시겠습니까?")) return;
    await reviewApi.markReviewDone(videoId);
    onDone();
  };

  if (loading) return <div className="p-6">로딩 중...</div>;

  return (
    <div className="bg-slate-800 rounded-xl p-6">
      <audio ref={audioRef} onEnded={handleAudioEnded} />

      {/* 안내 배너 */}
      <div className="bg-blue-900/20 border border-blue-800/40 rounded-lg p-4 mb-5">
        <p className="text-blue-300 text-sm font-medium mb-1.5">전사 검토</p>
        <p className="text-slate-400 text-xs leading-relaxed mb-2">
          자동 전사(STT) 결과를 확인하고 오류를 수정하세요. 정확한 텍스트가 TTS 학습 품질에 직접 영향을 미칩니다.
        </p>
        <ul className="text-xs text-slate-500 space-y-0.5 list-disc list-inside">
          <li>재생 버튼을 눌러 실제 발화를 들으며 텍스트와 비교하세요</li>
          <li>텍스트를 클릭하면 바로 수정할 수 있습니다 (Enter로 저장, Esc로 취소)</li>
          <li>의미 없는 구간이나 잘못된 항목은 &times; 버튼으로 삭제하세요</li>
        </ul>
      </div>

      {/* 헤더 */}
      <div className="flex justify-between items-center mb-4">
        <p className="text-sm text-slate-300">
          전체 <span className="font-mono text-slate-400">{entries.length}</span>개 세그먼트
        </p>
        <button
          className="bg-green-600 hover:bg-green-700 px-5 py-2 rounded-lg text-sm font-medium transition-colors"
          onClick={handleDone}
        >
          검토 완료
        </button>
      </div>

      {/* 화자 탭 */}
      <div className="flex gap-1.5 mb-4 overflow-x-auto">
        {speakerList.map((speaker) => (
          <button
            key={speaker}
            className={`px-3.5 py-1.5 rounded-lg text-sm transition-colors shrink-0 ${
              activeSpeaker === speaker
                ? "bg-blue-600 text-white"
                : "bg-slate-700 text-slate-400 hover:bg-slate-600"
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
          <div className="text-slate-500 text-sm text-center py-8">
            이 화자의 세그먼트가 없습니다.
          </div>
        ) : (
          filtered.map((entry) => (
            <div
              key={entry.file}
              className={`flex items-start gap-3 p-3 rounded-lg transition-colors ${
                playingFile === entry.file ? "bg-blue-900/30 ring-1 ring-blue-700/50" : "bg-slate-700/50 hover:bg-slate-700/70"
              }`}
            >
              {/* 재생 */}
              <button
                className={`mt-0.5 shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs transition-colors ${
                  playingFile === entry.file
                    ? "bg-blue-600 text-white"
                    : "bg-slate-600 text-slate-400 hover:bg-slate-500 hover:text-white"
                }`}
                onClick={() => play(entry)}
                title="재생"
              >
                {playingFile === entry.file ? "\u23F9" : "\u25B6"}
              </button>

              {/* 내용 */}
              <div className="flex-1 min-w-0">
                <p className="text-[10px] text-slate-500 mb-1 font-mono">{entry.file}</p>
                {editingFile === entry.file ? (
                  <div className="flex gap-2">
                    <input
                      type="text"
                      className="flex-1 bg-slate-700 rounded-lg px-2 py-1 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") saveEdit();
                        if (e.key === "Escape") cancelEdit();
                      }}
                      autoFocus
                    />
                    <button
                      className="bg-blue-600 hover:bg-blue-700 px-2.5 py-1 rounded-lg text-xs transition-colors"
                      onClick={saveEdit}
                      disabled={saving}
                    >
                      저장
                    </button>
                    <button
                      className="bg-slate-600 hover:bg-slate-500 px-2.5 py-1 rounded-lg text-xs transition-colors"
                      onClick={cancelEdit}
                    >
                      취소
                    </button>
                  </div>
                ) : (
                  <p
                    className="text-sm text-slate-200 cursor-pointer hover:text-white transition-colors"
                    onClick={() => startEdit(entry)}
                    title="클릭하여 수정"
                  >
                    {entry.text || <span className="text-slate-500 italic">(빈 텍스트 - 클릭하여 입력하거나 삭제하세요)</span>}
                  </p>
                )}
              </div>

              {/* 삭제 */}
              <button
                className="text-slate-500 hover:text-red-400 text-sm shrink-0 mt-0.5 transition-colors"
                onClick={() => handleDelete(entry)}
                title="삭제"
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
