"""Independent scalar-FP64 RoPE reference for actual cached model buffers."""
from .common import *
import math

SOURCE = 'https://raw.githubusercontent.com/huggingface/transformers/v4.43.4/src/transformers/modeling_rope_utils.py'
POSITIONS = [0, 1, 127, 128, 2047, 2048, 2175, 8191, 8192, 8319]


def check(model):
    import torch
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    config = model.config
    width = config.hidden_size // config.num_attention_heads
    scaling = config.llama3_rope_scaling
    frequencies = []
    # Scalar double arithmetic is deliberately independent of the vectorized
    # torch.where helper and of the buffers being validated.
    for channel in range(width // 2):
        frequency = float(config.rope_theta) ** (-2.0 * channel / width)
        if scaling:
            wavelength = 2.0 * math.pi / frequency
            low_edge = scaling['original_max_position_embeddings'] / scaling['low_freq_factor']
            high_edge = scaling['original_max_position_embeddings'] / scaling['high_freq_factor']
            if wavelength >= low_edge:
                frequency /= scaling['factor']
            elif wavelength > high_edge:
                blend = (scaling['original_max_position_embeddings'] / wavelength - scaling['low_freq_factor']) / (scaling['high_freq_factor'] - scaling['low_freq_factor'])
                frequency *= blend + (1.0 - blend) / scaling['factor']
        frequencies.append(frequency)
    expected_frequencies = torch.tensor(frequencies, dtype=torch.float64)
    angles = torch.tensor(POSITIONS, dtype=torch.float64)[:, None] * expected_frequencies[None, :]
    expected_cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    expected_sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    positions = torch.tensor(POSITIONS, device='cuda', dtype=torch.int64)[None, :]
    gen = torch.Generator(device='cpu').manual_seed(61)
    q = torch.randn((1, config.num_attention_heads, len(POSITIONS), width), generator=gen).half().cuda()
    k = torch.randn((1, config.num_key_value_heads, len(POSITIONS), width), generator=gen).half().cuda()
    phase = torch.complex(angles.cos(), angles.sin())[None, None, :, :]
    def complex_reference(x):
        x = x.cpu().double()
        paired = torch.complex(x[..., :width // 2], x[..., width // 2:]) * phase
        return torch.cat([paired.real, paired.imag], dim=-1)
    qref, kref = complex_reference(q), complex_reference(k)
    def error(actual, expected):
        actual = actual.cpu().double()
        if not bool(torch.isfinite(actual).all()): return None
        return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))
    rows = []
    for index, layer in enumerate(model.model.layers):
        rotary = layer.self_attn.rotary_emb
        inverse = rotary.inv_freq.cpu().double()
        frequency_error = float(((inverse - expected_frequencies).abs() / expected_frequencies.abs()).max()) if bool(torch.isfinite(inverse).all()) else None
        cos_error = error(rotary.cos_cached[positions[0]], expected_cos)
        sin_error = error(rotary.sin_cached[positions[0]], expected_sin)
        qout, kout = apply_rotary_pos_emb(q, k, rotary.cos_cached, rotary.sin_cached, positions, unsqueeze_dim=1)
        q_error, k_error = error(qout, qref), error(kout, kref)
        errors = [cos_error, sin_error, q_error, k_error]
        valid = frequency_error is not None and frequency_error <= 2e-5 and all(v is not None and v <= .002 for v in errors)
        rows.append({'layer': index, 'status': 'PASS' if valid else 'FAIL',
                     'inverse_frequency_max_relative_error': frequency_error,
                     'cosine_relative_l2': cos_error, 'sine_relative_l2': sin_error,
                     'query_rotation_relative_l2': q_error, 'key_rotation_relative_l2': k_error,
                     'inverse_frequency_dtype': str(rotary.inv_freq.dtype), 'cache_dtype': str(rotary.cos_cached.dtype),
                     'cache_capacity': rotary.max_seq_len_cached})
    return {'status': 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL', 'rows': rows,
            'reference_source': SOURCE, 'reference': 'Scalar double frequency calculation and complex-plane rotation; no replacement model/kernel is used.',
            'positions': POSITIONS, 'rope_theta': config.rope_theta, 'scaling': scaling,
            'thresholds': {'inverse_frequency_max_relative_error': 2e-5, 'cache_and_rotation_relative_l2': .002},
            'scope': 'All actual layer RoPE buffers in the FP16 model structure; shared RoPE construction is identical across methods. GQA tests separately use dequantized cached KV.'}


def main():
    a = arguments(__doc__).parse_args()
    dest = a.run / 'correctness' / f'rope_reference_{a.model}.json'
    if dest.exists(): return
    import torch
    initialize()
    with torch.inference_mode():
        model = build(a.model, 'fp16')
        write(dest, {'started': now(), 'environment': environment(), **check(model), 'ended': now()})


if __name__ == '__main__': main()
