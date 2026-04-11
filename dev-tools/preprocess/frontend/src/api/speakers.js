import api from "./client";
export async function listSpeakers() {
    const { data } = await api.get("/speakers");
    return data.speakers;
}
export async function addSpeaker(name) {
    const { data } = await api.post("/speakers", { name });
    return data.speakers;
}
export async function renameSpeaker(oldName, newName) {
    const { data } = await api.put("/speakers", {
        old_name: oldName,
        new_name: newName,
    });
    return data.speakers;
}
export async function deleteSpeaker(name) {
    const { data } = await api.delete(`/speakers/${encodeURIComponent(name)}`);
    return data.speakers;
}
export async function getSummary() {
    const { data } = await api.get("/speakers/summary");
    return data.summary;
}
