import copy

import torchvision
from torch import nn
from torch.utils import data
from pathlib import Path
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F

import numpy as np
import os
from skimage.metrics import structural_similarity as ssim_metric
import warnings
from torchdiffeq import odeint

from utils import scheduler


warnings.filterwarnings("ignore", category=UserWarning)


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    def __init__(self, decay=0.999):
        self.decay = decay

    @torch.no_grad()
    def update(self, ema_model, model):
        model_params = model.state_dict()
        ema_params = ema_model.state_dict()

        for name in model_params.keys():
            if ema_params[name].is_floating_point():
                ema_params[name].lerp_(model_params[name], 1.0 - self.decay)
            else:
                ema_params[name].copy_(model_params[name])


class Flow(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image_size,
            input_channels,
            ode_solver='euler',
            ode_steps=8
    ):
        super().__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        self.ode_solver = ode_solver
        self.denoise_fn = denoise_fn
        self.ode_steps = ode_steps
        self.device = 'cuda'

    def forward(self, x, mask_unhealthy, t=None):
        t = t if t is not None else torch.rand(x.shape[0], device=x.device)
        return self.p_losses(x, t, mask_unhealthy)

    def p_losses(self, x_start, t, mask_unhealthy):
        batch_size = x_start.shape[0]
        sigma_min = 1e-5

        noise = torch.randn_like(x_start)

        sigma_t = 1 - (1 - sigma_min) * t
        t_ = t.view(batch_size, 1, 1, 1)
        sigma_t_ = sigma_t.view(batch_size, 1, 1, 1)

        mu_t = t_ * x_start
        xt = mu_t + sigma_t_ * noise
        u = (x_start - (1 - sigma_min) * xt) / (1 - (1 - sigma_min) * t_)

        v = self.denoise_fn(xt, t)

        healthy_mask = 1.0 - mask_unhealthy
        raw_loss = F.mse_loss(v, u, reduction='none')
        masked_loss = raw_loss * healthy_mask

        loss = masked_loss.sum() / (healthy_mask.sum() + 1e-8)

        return loss

    def p_sample_loop(self, shape, training=False):
        if training:
            ode_steps = 8
        else:
            ode_steps = self.ode_steps

        x0 = torch.randn(shape, device=self.device)

        def func_conditional(t_scalar, x):
            t_batch = torch.full((x.shape[0],), t_scalar, device=self.device)
            v = self.denoise_fn(x, t_batch)

            return v

        t_schedule = torch.linspace(0, 1, ode_steps, device=self.device)

        traj = odeint(
            func_conditional,
            y0=x0,
            t=t_schedule,
            method=self.ode_solver,
            atol=1e-5,
            rtol=1e-5
        )

        x_final = traj[-1]

        return x_final

    @torch.no_grad()
    def sample(self, batch_size=1, training=False):
        image_size = self.image_size
        channels = self.input_channels

        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  training=training)

    # Restora-Flow: https://github.com/imigraz/Restora-Flow
    def sample_restora_flow_denoising(
            self,
            input_img,
            sigma_noise,
            ode_steps
    ):
        x = torch.randn_like(input_img, device=self.device)  # initialize x with noise
        x_obs = input_img * (1 - sigma_noise)

        torch_linspace = torch.linspace(0, 1, int(ode_steps), device=self.device)
        delta_t = 1 / len(torch_linspace)

        for t in torch_linspace:
            mask = torch.ones(input_img.shape, device=self.device)

            if t < (1 - sigma_noise):
                x = mask * x_obs + (1 - mask) * x
            else:
                x = x + delta_t * self.denoise_fn(x, torch.tensor(t, device=self.device).repeat(x.shape[0]))

        return x

    # Restora-Flow: https://github.com/imigraz/Restora-Flow
    def sample_restora_flow_mask_guided(
            self,
            input_img,
            mask,
            ode_steps,
            correction_steps,
            progress=False
    ):

        batch_size = input_img.shape[0]
        x = torch.randn_like(input_img, device=self.device)  # initialize x with noise
        pred_x_start = None

        times = scheduler.get_schedule_jump(
            t_T=ode_steps,
            n_sample=1,
            jump_length=1,
            jump_n_sample=correction_steps + 1
        )

        times = [((x - min(times)) / (max(times) - min(times))) for x in times]
        times.reverse()
        time_pairs = list(zip(times[:-1], times[1:]))

        if progress:
            from tqdm.auto import tqdm
            time_pairs = tqdm(time_pairs)

        for t_last, t_cur in time_pairs:
            t_last_t = torch.tensor([t_last] * batch_size, device=self.device).view(batch_size, 1, 1, 1)
            t_cur_t = torch.tensor([t_cur] * batch_size, device=self.device).view(batch_size, 1, 1, 1)

            if t_last < t_cur:
                with torch.no_grad():
                    if pred_x_start is not None:
                        # mask-based update
                        eps = torch.randn_like(x)
                        z_prim = t_last_t * input_img + (1 - t_last_t) * eps
                        x = mask * z_prim + (1 - mask) * x

                    # flow update
                    delta_t = t_cur_t - t_last_t
                    x = x + delta_t * self.denoise_fn(x, torch.tensor(t_last, device=self.device).repeat(batch_size))
                    out_sample = x.clone()

                    pred_x_start = True
            else:
                # trajectory correction
                x_1_prim = x + (1 - t_last_t) * self.denoise_fn(x, torch.tensor(t_last, device=self.device).repeat(batch_size))
                x = t_cur_t * x_1_prim + (1 - t_cur_t) * torch.randn_like(x)

        return out_sample


def compute_masked_metrics(gt_img, pred_img, healthy_mask):
    if hasattr(gt_img, 'cpu'):
        gt_np = gt_img.cpu().numpy()
        pred_np = pred_img.cpu().numpy()
        mask_np = healthy_mask.cpu().numpy()
    else:
        gt_np = gt_img
        pred_np = pred_img
        mask_np = healthy_mask

    valid_gt = gt_np[mask_np == 1]
    valid_pred = pred_np[mask_np == 1]

    if len(valid_gt) == 0:
        return 0.0, 100.0, 1.0

    mse = np.mean((valid_gt - valid_pred) ** 2)

    data_range = 2.0  # [-1, 1] range
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10 * np.log10((data_range ** 2) / mse)

    _, ssim_map = ssim_metric(
        gt_np,
        pred_np,
        data_range=data_range,
        full=True
    )
    valid_ssim = ssim_map[mask_np == 1]
    masked_ssim = np.mean(valid_ssim)

    return mse, psnr, masked_ssim


class Trainer(object):
    def __init__(
            self,
            denoiser_model,
            train_dataset,
            val_dataset,
            train_batch_size=1,
            val_batch_size=4,
            train_lr=2e-5,
            num_epochs=500,
            steps_per_epoch=10,
            monitor_every_n_epochs=1,
            ema_decay=0.999,
            ema_update_every=1,
            use_ema=False,
            output_folder="./results"
    ):
        super().__init__()

        self.model = denoiser_model
        self.use_ema = use_ema
        self.device = torch.device("cuda")

        if use_ema:
            self.ema = EMA(ema_decay)
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad = False
        else:
            self.ema_model = None

        self.ema_update_every = ema_update_every

        self.opt = AdamW(
            self.model.parameters(),
            lr=train_lr,
            weight_decay=1e-5
        )

        self.train_dl = cycle(
            data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
        )

        self.val_dl = data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs
        self.monitor_every_n_epochs = monitor_every_n_epochs

        self.global_step = 0
        self.current_epoch = 0

        self.output_folder = Path(output_folder)
        (self.output_folder.mkdir(exist_ok=True))

        self.writer = SummaryWriter(self.create_log_dir())

        self.best_pe_mean = float('inf')

    def create_log_dir(self):
        timestamp = self.output_folder.name
        log_dir = os.path.join("./logs/train/", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def save(self, epoch):
        data = {
            "epoch": epoch,
            "step": self.global_step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict() if self.use_ema else None,
            "best_pe_mean": self.best_pe_mean
        }
        path = self.output_folder / f"model-{epoch}.pt"
        print(f"Saving model to {path}")
        torch.save(data, path)

    def load(self, milestone):
        data = torch.load(str(self.output_folder / f'model-{milestone}.pt'))
        self.global_step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    @torch.no_grad()
    def validate(self):
        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()

        total_val_loss = 0.0
        total_samples = 0

        for batch_img, batch_mask_unhealthy, _, _ in self.val_dl:
            batch_img = batch_img.to(self.device)
            batch_mask_unhealthy = batch_mask_unhealthy.to(self.device)

            t = torch.rand(batch_img.shape[0], device=self.device)
            loss = eval_model(batch_img, batch_mask_unhealthy, t=t)

            total_val_loss += loss.item() * batch_img.shape[0]
            total_samples += batch_img.shape[0]

        avg_val_loss = total_val_loss / total_samples

        sample_images = eval_model.sample(batch_size=4, training=False)

        return avg_val_loss, sample_images

    @torch.no_grad()
    def validate_reconstruction(self, num_samples=10, ode_steps=20, correction_steps=1):
        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()

        total_mse = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        valid_samples = 0

        for i, (gt_img, _, healthy_mask, full_mask) in enumerate(self.val_dl):
            if i >= num_samples:
                break

            gt_img = gt_img.to(self.device)
            healthy_mask = healthy_mask.to(self.device)
            full_mask = full_mask.to(self.device)

            restora_mask = 1.0 - full_mask   # 0 = inpaint (unknown), 1 = keep (known)
            degraded_img = gt_img * restora_mask

            # Run Restora-Flow
            pred_img = eval_model.sample_restora_flow_mask_guided(
                input_img=degraded_img,
                mask=restora_mask,
                ode_steps=ode_steps,
                correction_steps=correction_steps
            )

            # Calculate metrics per item in batch
            for b in range(gt_img.shape[0]):
                # Skip if the random slice happens to not contain any healthy mask
                if healthy_mask[b].sum() < 10:
                    continue

                mse, psnr, ssim = compute_masked_metrics(
                    gt_img[b, 1], pred_img[b, 1], healthy_mask[b, 1]
                )

                total_mse += mse
                total_psnr += psnr
                total_ssim += ssim
                valid_samples += 1

        avg_mse = total_mse / max(valid_samples, 1)
        avg_psnr = total_psnr / max(valid_samples, 1)
        avg_ssim = total_ssim / max(valid_samples, 1)

        return avg_mse, avg_psnr, avg_ssim, pred_img

    def train(self):
        self.model.to(self.device)
        if self.use_ema:
            self.ema_model.to(self.device)

        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            self.model.train()
            epoch_loss = 0.0

            for _ in range(self.steps_per_epoch):
                input_img, mask_unhealthy = next(self.train_dl)

                loss = self.model(input_img.to(self.device), mask_unhealthy.to(self.device))
                loss.backward()

                self.opt.step()
                self.opt.zero_grad()

                if self.use_ema and self.global_step % self.ema_update_every == 0:
                    self.ema.update(self.ema_model, self.model)

                epoch_loss += loss.item()
                self.global_step += 1

            epoch_loss /= self.steps_per_epoch
            self.writer.add_scalar("train/loss", epoch_loss, epoch)

            # Validation
            if epoch % self.monitor_every_n_epochs == 0:
                print("Validate at epoch ", epoch)
                avg_val_loss, sample_images = self.validate()
                self.writer.add_scalar("val/loss", avg_val_loss, epoch)

                # Log a grid of generated images
                # Reshape from (B, 3, H, W) to (B*3, 1, H, W)
                sample_images_1ch = sample_images.view(-1, 1, 208, 208)
                grid = torchvision.utils.make_grid(sample_images_1ch, nrow=3, normalize=True, value_range=(-1, 1))
                self.writer.add_image('generated_brains', grid, epoch)
                self.writer.flush()

            # Validation reconstruction (e.g., every 50 epochs)
            if epoch % 50 == 0:
                print(f"Running Restora-Flow Validation...")
                val_mse, val_psnr, val_ssim, sample_preds = self.validate_reconstruction(
                    num_samples=4)
                
                self.writer.add_scalar("val/MSE", val_mse, epoch)
                self.writer.add_scalar("val/PSNR", val_psnr, epoch)
                self.writer.add_scalar("val/SSIM", val_ssim, epoch)
                print(f"Epoch {epoch} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | MSE: {val_mse:.4f}")

                if sample_preds is not None:
                    sample_preds_1ch = sample_preds.view(-1, 1, 208, 208)
                    grid = torchvision.utils.make_grid(
                        sample_preds_1ch,
                        nrow=3,
                        normalize=True,
                        value_range=(-1, 1)
                    )
                    self.writer.add_image('val/reconstructions', grid, epoch)
                    self.writer.flush()

                # Save best model
                if val_ssim > self.best_pe_mean:
                    self.best_pe_mean = val_ssim
                    self.save(f"best_ssim_{epoch}")

            if epoch % 100 == 0:
                self.save(epoch)
