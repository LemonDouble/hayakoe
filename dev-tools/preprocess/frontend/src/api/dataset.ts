import api from "./client";

export interface SpeakerStats {
  count: number;
  duration: number;
  train_count: number;
  train_duration: number;
  val_count: number;
  val_duration: number;
  path: string;
}

export interface DatasetResult {
  speakers: Record<string, SpeakerStats>;
  total: number;
  dataset_dir: string;
}

export async function buildDataset(valRatio: number): Promise<DatasetResult> {
  const { data } = await api.post("/dataset/build", { val_ratio: valRatio });
  return data;
}
