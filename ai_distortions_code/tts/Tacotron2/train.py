import argparse
import sys
from pathlib import Path


TTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TTS_DIR))

from common.experiment import DEFAULT_FILELIST, prepare_ljspeech_dataset  # noqa: E402


def import_coqui_training():
    try:
        from trainer import Trainer, TrainerArgs
        from TTS.config.shared_configs import BaseAudioConfig
        from TTS.tts.configs.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.tacotron2_config import Tacotron2Config
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.models.tacotron2 import Tacotron2
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies first, for example: pip install -r requirements.txt") from exc

    return {
        "Trainer": Trainer,
        "TrainerArgs": TrainerArgs,
        "BaseAudioConfig": BaseAudioConfig,
        "BaseDatasetConfig": BaseDatasetConfig,
        "Tacotron2Config": Tacotron2Config,
        "load_tts_samples": load_tts_samples,
        "Tacotron2": Tacotron2,
        "TTSTokenizer": TTSTokenizer,
        "AudioProcessor": AudioProcessor,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Tacotron-2 on watermarked victim recordings.")
    parser.add_argument("--target", "-t", required=True, help="directory with victim watermarked wavs")
    parser.add_argument("--train-filelist", default=str(DEFAULT_FILELIST), help="filename|text train filelist")
    parser.add_argument("--output", "-o", required=True, help="experiment name under experiments/")
    parser.add_argument("--work-dir", default="experiments", help="where run artifacts are written")
    parser.add_argument("--copy-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--limit", type=int, help="limit training utterances for few-shot/adaptation tests")
    parser.add_argument("--restore-path", help="pretrained Tacotron-2 checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-loader-workers", type=int, default=4)
    parser.add_argument("--num-eval-loader-workers", type=int, default=2)
    parser.add_argument("--precompute-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


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

    coqui = import_coqui_training()

    dataset_config = coqui["BaseDatasetConfig"](
        formatter="ljspeech",
        meta_file_train="metadata.csv",
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
    config = coqui["Tacotron2Config"](
        run_name=f"tacotron2_{args.output}",
        audio=audio_config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_eval_loader_workers,
        run_eval=True,
        test_delay_epochs=-1,
        r=6,
        gradual_training=[[0, 6, args.batch_size], [10000, 4, args.batch_size], [50000, 3, args.batch_size]],
        double_decoder_consistency=True,
        epochs=args.epochs,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=str(run_dir / "phoneme_cache"),
        precompute_num_workers=args.precompute_workers,
        print_step=25,
        print_eval=True,
        mixed_precision=args.mixed_precision,
        output_path=str(coqui_dir),
        lr=args.lr,
        datasets=[dataset_config],
    )

    audio_processor = coqui["AudioProcessor"].init_from_config(config)
    tokenizer, config = coqui["TTSTokenizer"].init_from_config(config)
    train_samples, eval_samples = coqui["load_tts_samples"](
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    model = coqui["Tacotron2"](config, audio_processor, tokenizer, speaker_manager=None)

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
