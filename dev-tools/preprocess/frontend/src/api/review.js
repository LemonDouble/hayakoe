import api from "./client";
export async function getTranscriptions(videoId) {
    const { data } = await api.get(`/videos/${videoId}/review`);
    return data;
}
export async function editTranscription(videoId, file, text) {
    await api.post(`/videos/${videoId}/review/edit`, { file, text });
}
export async function deleteTranscription(videoId, file) {
    await api.post(`/videos/${videoId}/review/delete`, { file });
}
export async function markReviewDone(videoId) {
    await api.post(`/videos/${videoId}/review/done`);
}
