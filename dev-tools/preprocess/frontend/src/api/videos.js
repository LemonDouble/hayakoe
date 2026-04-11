import api from "./client";
export async function listVideos() {
    const { data } = await api.get("/videos");
    return data.videos;
}
export async function uploadVideo(file) {
    const form = new FormData();
    form.append("file", file);
    const { data } = await api.post("/videos/upload", form);
    return data;
}
export async function getStatus(videoId) {
    const { data } = await api.get(`/videos/${videoId}/status`);
    return data;
}
export async function startExtract(videoId) {
    await api.post(`/videos/${videoId}/extract`);
}
export async function startSeparate(videoId) {
    await api.post(`/videos/${videoId}/separate`);
}
export async function startVad(videoId, params) {
    await api.post(`/videos/${videoId}/vad`, params ?? {});
}
export async function startTranscription(videoId) {
    await api.post(`/videos/${videoId}/transcribe`);
}
export async function rollbackVideo(videoId, stage) {
    await api.post(`/videos/${videoId}/rollback`, { stage });
}
export async function deleteVideo(videoId) {
    await api.delete(`/videos/${videoId}`);
}
