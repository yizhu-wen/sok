import argparse
import sys
from pathlib import Path


VC_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(VC_DIR))

def resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(SCRIPT_DIR / path)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare clean victim recordings for adaptive MediumVC fine-tuning.")
    parser.add_argument("--clean-dir", required=True, help="directory containing clean victim wavs")
    parser.add_argument("--output", default="Any2Any/pre_feature/spk_emb_mel_label_adaptive.pkl")
    parser.add_argument("--wav2mel-model-path", default="Any2Any/model/dvector/pre_model/wav2mel.pt")
    parser.add_argument("--dvector-model-path", default="Any2Any/model/dvector/pre_model/dvector-step250000.pt")
    parser.add_argument("--sampling-rate", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--num-mels", type=int, default=80)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--win-size", type=int, default=1024)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    import pickle

    import torch
    import torchaudio
    from librosa.util import normalize

    from Any2Any.pre_feature.figure_spkemb_mel import (
        get_spk_encoder,
        load_wav,
        mel_normalize,
        mel_spectrogram,
    )

    clean_dir = Path(args.clean_dir)
    wav_paths = sorted(clean_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found under {clean_dir}")

    wav2mel, dvector = get_spk_encoder(resolve_path(args.wav2mel_model_path), resolve_path(args.dvector_model_path))
    mel_label_list = []

    for wav_path in wav_paths:
        wav_tensor, sample_rate = torchaudio.load(str(wav_path))
        mel_tensor = wav2mel(wav_tensor, sample_rate)
        spk_emb = dvector.embed_utterance(mel_tensor).detach().squeeze().cpu().numpy()

        audio, _ = load_wav(str(wav_path))
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio).unsqueeze(0)
        mel = mel_spectrogram(
            audio,
            args.n_fft,
            args.num_mels,
            args.sampling_rate,
            args.hop_size,
            args.win_size,
            args.fmin,
            args.fmax,
            center=False,
        )
        mel = mel.squeeze(0).transpose(0, 1)
        mel = mel_normalize(mel)
        mel_label_list.append([spk_emb, mel, wav_path.stem])

    output = Path(resolve_path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump({"victim": mel_label_list}, file)

    print(f"wrote {output} with {len(mel_label_list)} clean victim utterances")


if __name__ == "__main__":
    main()
