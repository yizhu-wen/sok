import argparse
import sys
from pathlib import Path


TTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TTS_DIR))

from common.experiment import DEFAULT_FILELIST, prepare_ljspeech_dataset, run_command  # noqa: E402


def import_coqui_training():
    try:
        from trainer import Trainer, TrainerArgs
        from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
        from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.models.forward_tts import ForwardTTS
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies first, for example: pip install -r requirements.txt") from exc

    return {
        "Trainer": Trainer,
        "TrainerArgs": TrainerArgs,
        "BaseAudioConfig": BaseAudioConfig,
        "BaseDatasetConfig": BaseDatasetConfig,
        "Fastspeech2Config": Fastspeech2Config,
        "load_tts_samples": load_tts_samples,
        "ForwardTTS": ForwardTTS,
        "TTSTokenizer": TTSTokenizer,
        "AudioProcessor": AudioProcessor,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FastSpeech-2 on watermarked victim recordings.")
    parser.add_argument("--target", "-t", required=True, help="directory with victim watermarked wavs")
    parser.add_argument("--train-filelist", default=str(DEFAULT_FILELIST), help="filename|text train filelist")
    parser.add_argument("--output", "-o", required=True, help="experiment name under experiments/")
    parser.add_argument("--work-dir", default="experiments", help="where run artifacts are written")
    parser.add_argument("--copy-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--limit", type=int, help="limit training utterances for few-shot/adaptation tests")
    parser.add_argument("--restore-path", help="pretrained FastSpeech-2 checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-loader-workers", type=int, default=4)
    parser.add_argument("--num-eval-loader-workers", type=int, default=2)
    parser.add_argument("--precompute-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--compute-attention-masks", action="store_true", help="precompute duration masks with Tacotron2-DCA")
    parser.add_argument("--attention-model-name", default="tts_models/en/ljspeech/tacotron2-DCA")
    parser.add_argument("--attention-model-path", help="Tacotron/Tacotron2 checkpoint for attention-mask extraction")
    parser.add_argument("--attention-config-path", help="config.json for --attention-model-path")
    parser.add_argument("--attention-batch-size", type=int, default=16)
    parser.add_argument("--use-cuda-for-attention", action="store_true")
    return parser.parse_args()


def resolve_attention_model(args):
    if args.attention_model_path and args.attention_config_path:
        return args.attention_model_path, args.attention_config_path

    try:
        from TTS.utils.manage import ModelManager
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies first, for example: pip install -r requirements.txt") from exc

    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(args.attention_model_name)
    return model_path, config_path


def compute_attention_masks(args, dataset_dir):
    model_path, config_path = resolve_attention_model(args)
    command = [
        sys.executable,
        "-m",
        "TTS.bin.compute_attention_masks",
        "--model_path",
        model_path,
        "--config_path",
        config_path,
        "--dataset",
        "ljspeech",
        "--dataset_metafile",
        "metadata.csv",
        "--data_path",
        str(dataset_dir),
        "--batch_size",
        str(args.attention_batch_size),
    ]
    if args.use_cuda_for_attention:
        command.extend(["--use_cuda", "True"])
    run_command(command)


def main():
    args = parse_args()
    run_dir = Path(args.work_dir) / args.output
    dataset_dir = run_dir / "dataset"
    coqui_dir = run_dir / "coqui"

    prepared = prepare_ljspeech_dataset(
        audio_dir=args.target,
        filelist_path=args.train_filelist,
        dataset_dir=dataset_dir,
        copy_mode=args.copy_mode,
        limit=args.limit,
    )
    print(f"Prepared {prepared['num_items']} training utterances at {prepared['dataset_dir']}")

    if args.compute_attention_masks:
        compute_attention_masks(args, dataset_dir)

    coqui = import_coqui_training()
    dataset_config = coqui["BaseDatasetConfig"](
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        meta_file_attn_mask="metadata_attn_mask.txt" if args.compute_attention_masks else "",
        path=str(dataset_dir),
    )
    audio_config = coqui["BaseAudioConfig"](
        sample_rate=args.sample_rate,
        do_trim_silence=True,
        trim_db=60.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )
    config = coqui["Fastspeech2Config"](
        run_name=f"fastspeech2_{args.output}",
        audio=audio_config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_eval_loader_workers,
        compute_input_seq_cache=True,
        compute_f0=True,
        f0_cache_path=str(run_dir / "f0_cache"),
        compute_energy=True,
        energy_cache_path=str(run_dir / "energy_cache"),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=str(run_dir / "phoneme_cache"),
        precompute_num_workers=args.precompute_workers,
        print_step=50,
        print_eval=False,
        mixed_precision=args.mixed_precision,
        max_seq_len=500000,
        output_path=str(coqui_dir),
        lr=args.lr,
        datasets=[dataset_config],
    )

    if hasattr(config.model_args, "use_aligner"):
        config.model_args.use_aligner = not args.compute_attention_masks

    audio_processor = coqui["AudioProcessor"].init_from_config(config)
    tokenizer, config = coqui["TTSTokenizer"].init_from_config(config)
    train_samples, eval_samples = coqui["load_tts_samples"](
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    model = coqui["ForwardTTS"](config, audio_processor, tokenizer, speaker_manager=None)

    trainer_args = coqui["TrainerArgs"]()
    if args.restore_path:
        trainer_args.restore_path = args.restore_path

    trainer = coqui["Trainer"](
        trainer_args,
        config,
        str(coqui_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
