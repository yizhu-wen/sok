import argparse
from pathlib import Path

from common.experiment import DEFAULT_FILELIST, prepare_ljspeech_dataset


def import_coqui_vocoder_training():
    try:
        from trainer import Trainer, TrainerArgs
        from TTS.config import BaseAudioConfig
        from TTS.utils.audio import AudioProcessor
        from TTS.vocoder.configs import HifiganConfig
        from TTS.vocoder.datasets.preprocess import load_wav_data
        from TTS.vocoder.models.gan import GAN
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies first, for example: pip install -r requirements.txt") from exc

    return {
        "Trainer": Trainer,
        "TrainerArgs": TrainerArgs,
        "BaseAudioConfig": BaseAudioConfig,
        "AudioProcessor": AudioProcessor,
        "HifiganConfig": HifiganConfig,
        "load_wav_data": load_wav_data,
        "GAN": GAN,
    }


def build_parser(backbone_name):
    parser = argparse.ArgumentParser(description=f"Fine-tune HiFi-GAN vocoder for {backbone_name} adaptive* TTS.")
    parser.add_argument("--target", "-t", required=True, help="directory with victim watermarked wavs")
    parser.add_argument("--train-filelist", default=str(DEFAULT_FILELIST), help="filename|text train filelist")
    parser.add_argument("--output", "-o", required=True, help="experiment name under experiments/")
    parser.add_argument("--work-dir", default="experiments", help="where run artifacts are written")
    parser.add_argument("--copy-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--limit", type=int, help="limit training utterances for few-shot/adaptation tests")
    parser.add_argument("--restore-path", help="pretrained HiFi-GAN checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-split-size", type=int, default=1)
    parser.add_argument("--num-loader-workers", type=int, default=4)
    parser.add_argument("--num-eval-loader-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--pad-short", type=int, default=2000)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--no-noise-augment", action="store_true")
    return parser


def train_hifigan(backbone_name):
    args = build_parser(backbone_name).parse_args()
    run_dir = Path(args.work_dir) / args.output
    dataset_dir = run_dir / "dataset"
    coqui_dir = run_dir / "hifigan"

    prepared = prepare_ljspeech_dataset(
        audio_dir=args.target,
        filelist_path=args.train_filelist,
        dataset_dir=dataset_dir,
        copy_mode=args.copy_mode,
        limit=args.limit,
    )
    print(f"Prepared {prepared['num_items']} vocoder utterances at {prepared['dataset_dir']}")

    coqui = import_coqui_vocoder_training()
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
    config = coqui["HifiganConfig"](
        run_name=f"hifigan_{backbone_name}_{args.output}",
        audio=audio_config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_eval_loader_workers,
        run_eval=True,
        test_delay_epochs=5,
        epochs=args.epochs,
        seq_len=args.seq_len,
        pad_short=args.pad_short,
        use_noise_augment=not args.no_noise_augment,
        eval_split_size=args.eval_split_size,
        print_step=25,
        print_eval=False,
        mixed_precision=args.mixed_precision,
        lr_gen=args.lr,
        lr_disc=args.lr,
        data_path=str(dataset_dir / "wavs"),
        output_path=str(coqui_dir),
    )
    config.l1_spec_loss_params["sample_rate"] = args.sample_rate
    config.l1_spec_loss_params["mel_fmax"] = 8000

    audio_processor = coqui["AudioProcessor"](**config.audio.to_dict())
    eval_samples, train_samples = coqui["load_wav_data"](config.data_path, config.eval_split_size)
    model = coqui["GAN"](config, audio_processor)

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
