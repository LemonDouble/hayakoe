"""Synthesizer ONNX 내보내기."""

import contextlib
import io
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from rich.progress import Progress, SpinnerColumn, TextColumn

from cli.i18n import t
from cli.ui.console import console

from hayakoe.models.hyper_parameters import HyperParameters
from hayakoe.models.infer import get_net_g
from hayakoe.nlp.symbols import SYMBOLS


class _SynthesizerWrapper(nn.Module):
    """SynthesizerTrn.infer() 로직을 ONNX 내보내기 가능한 Module로 래핑."""

    def __init__(self, net_g):
        super().__init__()
        self.net_g = net_g

    def forward(self, x, x_lengths, sid, tone, language, bert, style_vec,
                noise_scale, length_scale, noise_scale_w, sdp_ratio):
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
        path = _SynthesizerWrapper._sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
        path = path.view(b, t_x, t_y)
        pad = torch.zeros(b, 1, t_y, dtype=path.dtype, device=path.device)
        path = path - torch.cat([pad, path[:, :-1, :]], dim=1)
        path = path.unsqueeze(1).transpose(2, 3) * mask
        return path


class _DurationPredictorWrapper(nn.Module):
    """TextEncoder + DurationPredictor만 실행해 phoneme별 frame 수를 반환한다.

    문장 경계 무음 길이 예측에 사용된다 — Decoder/Flow를 타지 않으므로
    Synthesizer ONNX 대비 매우 가볍다.
    """

    def __init__(self, net_g):
        super().__init__()
        self.net_g = net_g

    def forward(self, x, x_lengths, sid, tone, language, bert, style_vec,
                length_scale, noise_scale_w, sdp_ratio):
        g = self.net_g.emb_g(sid).unsqueeze(-1)
        x_enc, _, _, x_mask = self.net_g.enc_p(
            x, x_lengths, tone, language, bert, style_vec, g=g
        )
        logw = self.net_g.sdp(
            x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w
        ) * sdp_ratio + self.net_g.dp(x_enc, x_mask, g=g) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale
        return torch.ceil(w).squeeze(1)  # [batch, phone_len]


def export_duration_predictor(
    config_path: Path, checkpoint: Path, output_dir: Path, opset: int = 17,
) -> Path:
    """TextEncoder + DurationPredictor만 FP32 ONNX로 내보낸다.

    문장 경계 무음 길이 예측 (자연스러운 다문장 합성용) 에 사용된다.

    Args:
        config_path: 모델 ``config.json`` 경로.
        checkpoint: 내보낼 ``.safetensors`` 체크포인트.
        output_dir: 출력 디렉토리.
        opset: ONNX opset 버전.

    Returns:
        생성된 ONNX 파일 경로.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "duration_predictor.onnx"

    hps = HyperParameters.load_from_json(config_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(t("export.model_loading"), total=None)
        net_g = get_net_g(str(checkpoint), hps.version, "cpu", hps)
        net_g.eval()

        wrapper = _DurationPredictorWrapper(net_g)
        wrapper.eval()

        seq_len = 20
        dummy_inputs = (
            torch.randint(0, len(SYMBOLS), (1, seq_len)),       # x
            torch.LongTensor([seq_len]),                         # x_lengths
            torch.LongTensor([0]),                               # sid
            torch.randint(0, 10, (1, seq_len)),                  # tone
            torch.zeros(1, seq_len, dtype=torch.long),           # language
            torch.randn(1, 1024, seq_len),                       # bert
            torch.randn(1, 256),                                 # style_vec
            torch.FloatTensor([1.0]),                            # length_scale
            torch.FloatTensor([0.8]),                            # noise_scale_w
            torch.FloatTensor([0.0]),                            # sdp_ratio
        )

        input_names = [
            "x", "x_lengths", "sid", "tone", "language", "bert", "style_vec",
            "length_scale", "noise_scale_w", "sdp_ratio",
        ]
        output_names = ["durations"]
        dynamic_axes = {
            "x": {1: "phone_len"},
            "tone": {1: "phone_len"},
            "language": {1: "phone_len"},
            "bert": {2: "phone_len"},
            "durations": {1: "phone_len"},
        }

        progress.update(task, description=t("export.onnx_exporting"))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", message="Constant folding")
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
        elapsed = time.time() - t0

        total_size = output_path.stat().st_size
        data_file = Path(str(output_path) + ".data")
        if data_file.exists():
            total_size += data_file.stat().st_size

        progress.update(task, description=t("export.done"))

    console.print(t("export.time", elapsed=elapsed))
    console.print(t("export.size", size=total_size / 1024 / 1024))

    return output_path


def export_synthesizer(
    config_path: Path, checkpoint: Path, output_dir: Path, opset: int = 17,
) -> Path:
    """Synthesizer를 FP32 ONNX로 내보낸다.

    Args:
        config_path: 모델 ``config.json`` 경로.
        checkpoint: 내보낼 ``.safetensors`` 체크포인트.
        output_dir: 출력 디렉토리.
        opset: ONNX opset 버전.

    Returns:
        생성된 ONNX 파일 경로.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthesizer.onnx"

    hps = HyperParameters.load_from_json(config_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 모델 로드
        task = progress.add_task(t("export.model_loading"), total=None)
        net_g = get_net_g(str(checkpoint), hps.version, "cpu", hps)
        net_g.eval()

        # remove_weight_norm 내부의 print() 억제
        if hasattr(net_g.dec, "remove_weight_norm"):
            with contextlib.redirect_stdout(io.StringIO()):
                net_g.dec.remove_weight_norm()

        wrapper = _SynthesizerWrapper(net_g)
        wrapper.eval()

        # 더미 입력 생성
        seq_len = 20
        dummy_inputs = (
            torch.randint(0, len(SYMBOLS), (1, seq_len)),       # x
            torch.LongTensor([seq_len]),                         # x_lengths
            torch.LongTensor([0]),                               # sid
            torch.randint(0, 10, (1, seq_len)),                  # tone
            torch.zeros(1, seq_len, dtype=torch.long),           # language
            torch.randn(1, 1024, seq_len),                       # bert
            torch.randn(1, 256),                                 # style_vec
            torch.FloatTensor([0.667]),                          # noise_scale
            torch.FloatTensor([1.0]),                            # length_scale
            torch.FloatTensor([0.8]),                            # noise_scale_w
            torch.FloatTensor([0.2]),                            # sdp_ratio
        )

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

        # ONNX 내보내기 (TracerWarning / constant folding 경고 억제)
        progress.update(task, description=t("export.onnx_exporting"))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", message="Constant folding")
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
        elapsed = time.time() - t0

        # 크기 표시
        total_size = output_path.stat().st_size
        data_file = Path(str(output_path) + ".data")
        if data_file.exists():
            total_size += data_file.stat().st_size

        progress.update(task, description=t("export.done"))

    console.print(t("export.time", elapsed=elapsed))
    console.print(t("export.size", size=total_size / 1024 / 1024))

    return output_path
