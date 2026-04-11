import api from "./client";

export interface SegmentInfo {
  file: string;
  start: number;
  end: number;
  duration: number;
}

export interface ClassificationState {
  done: boolean;
  history_count: number;
  speakers: { name: string; count: number; total_duration: number }[];
}

export interface UnclassifiedResult {
  segments: SegmentInfo[];
  total: number;
  classified: number;
  total_all: number;
}

export async function getClassification(
  videoId: string
): Promise<ClassificationState> {
  const { data } = await api.get(`/videos/${videoId}/classification`);
  return data;
}

export async function getUnclassified(
  videoId: string,
  offset = 0,
  limit = 20
): Promise<UnclassifiedResult> {
  const { data } = await api.get(
    `/videos/${videoId}/classification/segments?offset=${offset}&limit=${limit}`
  );
  return data;
}

export async function classifySegment(
  videoId: string,
  segmentFile: string,
  speaker: string
): Promise<void> {
  await api.post(`/videos/${videoId}/classification/classify`, {
    segment_file: segmentFile,
    speaker,
  });
}

export async function undoClassification(videoId: string): Promise<void> {
  await api.post(`/videos/${videoId}/classification/undo`);
}

export async function markDone(videoId: string): Promise<void> {
  await api.post(`/videos/${videoId}/classification/done`);
}
