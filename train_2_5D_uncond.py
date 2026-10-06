import argparse
import os
import time
import torch.backends.cudnn
from dataset_2_5D_uncond import BrainImageGenerator
from generative_model.trainer_fm import Flow, Trainer
from generative_model.unet import create_model

torch.backends.cuda.matmul_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args():
    parser = argparse.ArgumentParser(
        description='Train 2.5D Unconditional Flow Matching Model'
    )
    parser.add_argument(
        '--train_setup',
        type=str,
        default='dataset/setup/train.txt',
        help='Path to train split file',
    )
    parser.add_argument(
        '--val_setup',
        type=str,
        default='dataset/setup/val.txt',
        help='Path to validation split file',
    )
    parser.add_argument(
        '--input_size', type=int, default=208, help='Image slice size'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Training batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Training learning rate'
    )
    parser.add_argument(
        '--epochs', type=int, default=5000, help='Total training epochs'
    )
    parser.add_argument(
        '--ode_steps',
        type=int,
        default=32,
        help='ODE integration steps for Flow model',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='experiments/train',
        help='Base output folder',
    )
    parser.add_argument(
        '--monitor_every',
        type=int,
        default=5,
        help='Evaluate/monitor every N epochs',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    in_channels = out_channels = 3
    unet_num_input_channels = 128

    # Output folder path
    experiment_name = '2_5D_uncond'
    output_folder = os.path.join(
        args.save_dir,
        time.strftime('%Y%m%d-%H%M%S') + '_' + experiment_name,
    )
    os.makedirs(output_folder, exist_ok=True)

    # Datasets
    train_dataset = BrainImageGenerator(
        setup_file_path=args.train_setup,
        train=True
    )
    validation_dataset = BrainImageGenerator(
        setup_file_path=args.val_setup,
        train=False
    )

    # Model
    unet_model = create_model(
        args.input_size,
        unet_num_input_channels,
        num_res_blocks=2,
        in_channels=in_channels,
        out_channels=out_channels,
        use_scale_shift_norm=True,
        dims=2
    ).cuda()

    flow_model = Flow(
        unet_model,
        image_size=args.input_size,
        input_channels=in_channels,
        ode_steps=args.ode_steps
    ).cuda()

    # Trainer
    trainer = Trainer(
        flow_model,
        train_dataset,
        validation_dataset,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        ema_decay=0.995,
        num_epochs=args.epochs,
        steps_per_epoch=len(train_dataset) // args.batch_size,
        monitor_every_n_epochs=args.monitor_every,
        output_folder=output_folder,
        use_ema=True
    )

    trainer.train()
