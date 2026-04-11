import api from "./client";

export interface VideoInfo {
  id: string;
  filename: string;
  uploaded_at: string;
  stage: string;
}

export interface VideoStatus {
  stage: string;
  filename: string;
  source_file: string | null;
  processing: {
    stage: string;
    progress: number;
    message: string;
    updated_at: string;
  } | null;
  error: {
    stage: string;
    message: string;
    updated_at: string;
  } | null;
  summary?: { name: string; count: number; total_duration: number }[];
}

export async function listVideos(): Promise<VideoInfo[]> {
  const { data } = await api.get("/videos");
  return data.videos;
}

export async function uploadVideo(file: File): Promise<{ id: string }> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/videos/upload", form);
  return data;
}

export async function getStatus(videoId: string): Promise<VideoStatus> {
  const { data } = await api.get(`/videos/${videoId}/status`);
  return data;
}

export async function startExtract(videoId: string): Promise<void> {
  await api.post(`/videos/${videoId}/extract`);
}

export async function startSeparate(videoId: string): Promise<void> {
  await api.post(`/videos/${videoId}/separate`);
}

export interface VadParams {
  min_segment_sec: number;
  max_segment_sec: number;
  threshold: number;
  min_silence_ms: number;
}

export async function startVad(videoId: string, params?: VadParams): Promise<void> {
  await api.post(`/videos/${videoId}/vad`, params ?? {});
}

export async function startTranscription(videoId: string): Promise<void> {
  await api.post(`/videos/${videoId}/transcribe`);
}

export async function rollbackVideo(
  videoId: string,
  stage: string
): Promise<void> {
  await api.post(`/videos/${videoId}/rollback`, { stage });
}

export async function deleteVideo(videoId: string): Promise<void> {
  await api.delete(`/videos/${videoId}`);
}
