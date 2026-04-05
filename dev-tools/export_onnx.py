"""HayaKoe 모델을 ONNX 형식으로 내보내기.

두 개의 ONNX 모델을 내보낸다:
  1. BERT (DeBERTa) — 텍스트 토큰 → 히든 스테이트
  2. Synthesizer (VITS) — 음소 + BERT 특징 + 스타일 → 오디오

사용법:
    python -m dev_tools.export_onnx [--model-dir ./model_assets/Elaina] [--output-dir ./onnx_models]

프로젝트 루트 디렉토리(hayakoe/)에서 실행한다.
"""

import argparse
import sys
import time
from pathlib import Path

# 직접 실행 시 프로젝트 루트가 sys.path에 포함되도록 보장
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.models.infer import get_net_g
from hayakoe.nlp import bert_models
from hayakoe.nlp.symbols import SYMBOLS


def _print_onnx_size(path: Path) -> None:
    """외부 데이터 파일을 포함한 ONNX 모델 크기를 출력한다."""
    total = path.stat().st_size
    data_file = Path(str(path) + ".data")
    if data_file.exists():
        total += data_file.stat().st_size
    print(f"  Size: {total / 1024 / 1024:.1f}MB")


def export_bert(output_dir: Path, opset: int = 17) -> Path:
    """DeBERTa BERT 모델을 ONNX로 내보낸다."""
    print("\n=== Exporting BERT to ONNX ===")
    output_path = output_dir / "bert.onnx"

    model = bert_models.load_model(device_map="cpu")
    model.eval()

    from hayakoe.constants import BERT_JP_REPO
    tokenizer = bert_models.load_tokenizer(pretrained_model_name_or_path=BERT_JP_REPO)
    dummy_text = "テスト文章です。"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids")

    # DeBERTa v2는 token_type_ids를 사용하지 않음
    if token_type_ids is not None and not hasattr(model.config, "type_vocab_size"):
        token_type_ids = None

    # 필요한 히든 스테이트만 반환하도록 모델을 래핑 (뒤에서 3번째)
    class BertWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # bert_feature.py:54와 일치하는 뒤에서 3번째 히든 스테이트를 사용
            hidden = outputs.hidden_states[-3]
            return hidden

    wrapper = BertWrapper(model)
    wrapper.eval()

    dummy_inputs = (input_ids, attention_mask)
    input_names = ["input_ids", "attention_mask"]
    output_names = ["hidden_states"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "hidden_states": {0: "batch", 1: "seq_len"},
    }

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Exported to {output_path} ({time.time() - t0:.1f}s)")
    _print_onnx_size(output_path)

    # 검증
    _verify_bert(wrapper, output_path, input_ids, attention_mask)

    # 정리
    bert_models.unload_model()

    return output_path


def _verify_bert(wrapper, onnx_path, input_ids, attention_mask):
    """ONNX 출력이 PyTorch와 일치하는지 검증한다."""
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = wrapper(input_ids, attention_mask).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    })[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"  Verification: max diff = {diff:.6e} {'OK' if diff < 1e-4 else 'WARN'}")


def export_synthesizer(model_dir: Path, output_dir: Path, opset: int = 17) -> Path:
    """SynthesizerTrn을 ONNX로 내보낸다."""
    print("\n=== Exporting Synthesizer to ONNX ===")
    output_path = output_dir / "synthesizer.onnx"

    config_path = model_dir / "config.json"
    hps = HyperParameters.load_from_json(config_path)

    # 최신 체크포인트 찾기
    safetensors_files = sorted(model_dir.glob("*.safetensors"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors in {model_dir}")
    model_path = safetensors_files[0]
    print(f"  Using checkpoint: {model_path.name}")

    net_g = get_net_g(str(model_path), hps.version, "cpu", hps)
    net_g.eval()

    # 깔끔한 내보내기를 위해 weight norm 제거
    if hasattr(net_g.dec, "remove_weight_norm"):
        net_g.dec.remove_weight_norm()

    # 내보내기를 위해 infer 메서드를 적절한 nn.Module로 래핑
    class SynthesizerWrapper(nn.Module):
        def __init__(self, net_g):
            super().__init__()
            self.net_g = net_g

        def forward(self, x, x_lengths, sid, tone, language, bert, style_vec,
                    noise_scale, length_scale, noise_scale_w, sdp_ratio):
            # SynthesizerTrn.infer() 로직 복제
            g = self.net_g.emb_g(sid).unsqueeze(-1)

            x_enc, m_p, logs_p, x_mask = self.net_g.enc_p(
                x, x_lengths, tone, language, bert, style_vec, g=g
            )

            logw = self.net_g.sdp(
                x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w
            ) * sdp_ratio + self.net_g.dp(x_enc, x_mask, g=g) * (1 - sdp_ratio)

            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(
                self._sequence_mask(y_lengths, None), 1
            ).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = self._generate_path(w_ceil, attn_mask)

            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            z = self.net_g.flow(z_p, y_mask, g=g, reverse=True)
            o = self.net_g.dec((z * y_mask), g=g)
            return o

        @staticmethod
        def _sequence_mask(length, max_length=None):
            if max_length is None:
                max_length = length.max()
            x = torch.arange(max_length, dtype=length.dtype, device=length.device)
            return x.unsqueeze(0) < length.unsqueeze(1)

        @staticmethod
        def _generate_path(duration, mask):
            b, _, t_y, t_x = mask.shape
            cum_duration = torch.cumsum(duration, -1)
            cum_duration_flat = cum_duration.view(b * t_x)
            path = SynthesizerWrapper._sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
            path = path.view(b, t_x, t_y)
            pad = torch.zeros(b, 1, t_y, dtype=path.dtype, device=path.device)
            path = path - torch.cat([pad, path[:, :-1, :]], dim=1)
            path = path.unsqueeze(1).transpose(2, 3) * mask
            return path

    wrapper = SynthesizerWrapper(net_g)
    wrapper.eval()

    # 더미 입력 생성
    seq_len = 20
    x = torch.randint(0, len(SYMBOLS), (1, seq_len))
    x_lengths = torch.LongTensor([seq_len])
    sid = torch.LongTensor([0])
    tone = torch.randint(0, 10, (1, seq_len))
    language = torch.zeros(1, seq_len, dtype=torch.long)
    bert = torch.randn(1, 1024, seq_len)
    style_vec = torch.randn(1, 256)
    noise_scale = torch.FloatTensor([0.667])
    length_scale = torch.FloatTensor([1.0])
    noise_scale_w = torch.FloatTensor([0.8])
    sdp_ratio = torch.FloatTensor([0.2])

    dummy_inputs = (x, x_lengths, sid, tone, language, bert, style_vec,
                    noise_scale, length_scale, noise_scale_w, sdp_ratio)

    input_names = [
        "x", "x_lengths", "sid", "tone", "language", "bert", "style_vec",
        "noise_scale", "length_scale", "noise_scale_w", "sdp_ratio",
    ]
    output_names = ["audio"]
    dynamic_axes = {
        "x": {1: "phone_len"},
        "tone": {1: "phone_len"},
        "language": {1: "phone_len"},
        "bert": {2: "phone_len"},
        "audio": {2: "audio_len"},
    }

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Exported to {output_path} ({time.time() - t0:.1f}s)")
    _print_onnx_size(output_path)

    # 검증
    _verify_synthesizer(wrapper, output_path, dummy_inputs, input_names)

    return output_path


def _verify_synthesizer(wrapper, onnx_path, dummy_inputs, input_names):
    """ONNX 출력이 PyTorch와 일치하는지 검증한다 (randn으로 인해 근사값)."""
    import onnxruntime as ort

    # 비교를 위해 랜덤 시드 고정
    torch.manual_seed(42)
    with torch.no_grad():
        pt_out = wrapper(*dummy_inputs).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    feeds = {}
    for name, tensor in zip(input_names, dummy_inputs):
        feeds[name] = tensor.numpy()

    # 참고: ONNX의 randn_like는 다른 RNG를 사용하므로 출력이 다를 수 있음
    ort_out = sess.run(None, feeds)[0]

    # 형상 일치 확인
    if pt_out.shape == ort_out.shape:
        print(f"  Verification: shapes match {pt_out.shape} OK")
    else:
        print(f"  Verification: shape mismatch PT={pt_out.shape} ONNX={ort_out.shape} WARN")
    print(f"  Note: exact values differ due to torch.randn_like vs ONNX RandomNormalLike")


def quantize_onnx(onnx_path: Path, output_dir: Path) -> Path:
    """ONNX 모델에 INT8 동적 양자화를 적용한다.

    MatMul과 Gemm 연산(Linear 레이어)만 양자화한다.
    Conv 연산은 ONNX Runtime에 ConvInteger 구현이 없어 제외한다.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    stem = onnx_path.stem
    q8_path = output_dir / f"{stem}_q8.onnx"

    print(f"\n=== Quantizing {stem} to INT8 (MatMul/Gemm only) ===")
    t0 = time.time()
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(q8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"  Quantized to {q8_path} ({time.time() - t0:.1f}s)")
    orig_size = onnx_path.stat().st_size / 1024 / 1024
    q8_size = q8_path.stat().st_size / 1024 / 1024
    print(f"  Size: {orig_size:.1f}MB → {q8_size:.1f}MB ({q8_size/orig_size*100:.0f}%)")
    return q8_path


def main():
    parser = argparse.ArgumentParser(description="Export HayaKoe to ONNX")
    parser.add_argument("--model-dir", type=str, default="./model_assets/Elaina")
    parser.add_argument("--output-dir", type=str, default="./onnx_models")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_bert:
        bert_path = export_bert(output_dir, args.opset)
        if not args.skip_quantize:
            quantize_onnx(bert_path, output_dir)

    if not args.skip_synth:
        synth_path = export_synthesizer(model_dir, output_dir, args.opset)
        if not args.skip_quantize:
            quantize_onnx(synth_path, output_dir)

    print("\n=== Done ===")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
