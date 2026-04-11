import api from "./client";
export async function buildDataset(valRatio) {
    const { data } = await api.post("/dataset/build", { val_ratio: valRatio });
    return data;
}
