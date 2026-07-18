import api from "./client";

export async function listSpeakers(): Promise<string[]> {
  const { data } = await api.get("/speakers");
  return data.speakers;
}

export async function addSpeaker(name: string): Promise<string[]> {
  const { data } = await api.post("/speakers", { name });
  return data.speakers;
}

export async function deleteSpeaker(name: string): Promise<string[]> {
  const { data } = await api.delete(`/speakers/${encodeURIComponent(name)}`);
  return data.speakers;
}

export interface SpeakerSummary {
  name: string;
  count: number;
  total_duration: number;
}

export async function getSummary(): Promise<SpeakerSummary[]> {
  const { data } = await api.get("/speakers/summary");
  return data.summary;
}
