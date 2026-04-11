import api from "./client";

export interface TranscriptionEntry {
  file: string;
  speaker: string;
  text: string;
  language: string;
}

export async function getTranscriptions(
  videoId: string
): Promise<{ entries: TranscriptionEntry[]; total: number }> {
  const { data } = await api.get(`/videos/${videoId}/review`);
  return data;
}

export async function editTranscription(
  videoId: string,
  file: string,
  text: string
): Promise<void> {
  await api.post(`/videos/${videoId}/review/edit`, { file, text });
}

export async function deleteTranscription(
  videoId: string,
  file: string
): Promise<void> {
  await api.post(`/videos/${videoId}/review/delete`, { file });
}

export async function markReviewDone(videoId: string): Promise<void> {
  await api.post(`/videos/${videoId}/review/done`);
}
