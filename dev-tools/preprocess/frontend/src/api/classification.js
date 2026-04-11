import api from "./client";
export async function getClassification(videoId) {
    const { data } = await api.get(`/videos/${videoId}/classification`);
    return data;
}
export async function getUnclassified(videoId, offset = 0, limit = 20) {
    const { data } = await api.get(`/videos/${videoId}/classification/segments?offset=${offset}&limit=${limit}`);
    return data;
}
export async function classifySegment(videoId, segmentFile, speaker) {
    await api.post(`/videos/${videoId}/classification/classify`, {
        segment_file: segmentFile,
        speaker,
    });
}
export async function undoClassification(videoId) {
    await api.post(`/videos/${videoId}/classification/undo`);
}
export async function markDone(videoId) {
    await api.post(`/videos/${videoId}/classification/done`);
}
