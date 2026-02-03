"""
airbench94_softmuon.py
SoftMuon variant of the CIFAR-10 speedrun benchmark.
Based on airbench94_muon.py by Keller Jordan.

Key change: Replace hard polar (Muon) with regularized polar (SoftMuon)
Formula: Q_λ(G) = G @ (G^T G + λI)^{-1/2}
"""

import os
import sys
import argparse
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

#############################################
#        SoftMuon optimizer (Tauon)         #
#############################################

@torch.compile
def coupled_newton_invsqrt(B, steps=5, eps=1e-7):
    """
    Coupled Newton iteration for B^{-1/2}.
    B should be symmetric positive definite.
    """
    # Scale B to have spectral norm ~1
    B_norm = B.norm()
    B_scaled = B / (B_norm + eps)

    d = B.size(0)
    I = torch.eye(d, device=B.device, dtype=B.dtype)

    # Initialize
    Y = B_scaled
    Z = I.clone()

    for _ in range(steps):
        T = Z @ Y
        Y_new = 0.5 * Y @ (3 * I - T)
        Z_new = 0.5 * (3 * I - T) @ Z
        Y = Y_new
        Z = Z_new

    # Rescale: B^{-1/2} = (B/s)^{-1/2} * s^{-1/2}
    return Z / torch.sqrt(B_norm + eps)


@torch.compile
def regularized_polar_softmuon(G, lambda_reg=0.1, steps=5, eps=1e-7):
    """
    Compute regularized polar factor: Q_λ(G) = G @ (G^T G + λI)^{-1/2}

    This applies spectral filter φ_λ(σ) = σ / sqrt(σ² + λ)
    - λ → 0: approaches Muon (hard polar, φ → 1)
    - λ → ∞: approaches scaled gradient descent (φ → 0)
    """
    assert len(G.shape) == 2
    m, n = G.shape

    # Work with smaller dimension for efficiency
    if m >= n:
        # G is tall: use G^T G (n x n)
        GtG = G.T @ G
        d = n
        I = torch.eye(d, device=G.device, dtype=G.dtype)
        B = GtG + lambda_reg * I
        B_invsqrt = coupled_newton_invsqrt(B, steps=steps, eps=eps)
        return G @ B_invsqrt
    else:
        # G is wide: use G G^T (m x m)
        GGt = G @ G.T
        d = m
        I = torch.eye(d, device=G.device, dtype=G.dtype)
        B = GGt + lambda_reg * I
        B_invsqrt = coupled_newton_invsqrt(B, steps=steps, eps=eps)
        return B_invsqrt @ G


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Original Muon: Newton-Schulz iteration for orthogonalization.
    Kept for comparison.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class SoftMuon(torch.optim.Optimizer):
    """
    SoftMuon optimizer with regularized polar decomposition.
    """
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False,
                 lambda_reg=0.1, ns_iters=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                       lambda_reg=lambda_reg, ns_iters=ns_iters)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            lambda_reg = group["lambda_reg"]
            ns_iters = group["ns_iters"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # Normalize the weight (same as Muon)
                p.data.mul_(len(p.data)**0.5 / p.data.norm())

                # Apply regularized polar (SoftMuon) instead of hard polar (Muon)
                g_2d = g.reshape(len(g), -1)
                if lambda_reg > 0:
                    update = regularized_polar_softmuon(
                        g_2d.float(), lambda_reg=lambda_reg, steps=ns_iters
                    ).to(g.dtype).view(g.shape)
                else:
                    # λ=0 falls back to original Muon
                    update = zeropower_via_newtonschulz5(g_2d, steps=ns_iters).view(g.shape)

                p.data.add_(update, alpha=-lr)


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def train_one_run(run, model, lambda_reg=0.1, ns_iters=5, softmuon_lr=0.24, verbose=True):
    """Train one run with given hyperparameters."""

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))

    if run == "warmup":
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)

    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]

    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd/bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)
    ]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)

    # SoftMuon for conv filters
    optimizer2 = SoftMuon(filter_params, lr=softmuon_lr, momentum=0.6, nesterov=True,
                          lambda_reg=lambda_reg, ns_iters=ns_iters)

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Timing
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()
    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    # Initialize whitening
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()

            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)

            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        if verbose and run != "warmup":
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            val_acc = evaluate(model, test_loader, tta_level=0)
            print(f"  Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, time={time_seconds:.2f}s")

    # TTA evaluation
    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()

    return tta_val_acc, time_seconds


def run_sweep(model, lambda_values, n_trials=5, ns_iters=5, softmuon_lr=0.24):
    """Run hyperparameter sweep over lambda values."""

    results = {}

    for lambda_reg in lambda_values:
        print(f"\n{'='*60}")
        print(f"Testing lambda={lambda_reg}")
        print(f"{'='*60}")

        accs = []
        times = []

        for trial in range(n_trials):
            acc, t = train_one_run(trial, model, lambda_reg=lambda_reg,
                                   ns_iters=ns_iters, softmuon_lr=softmuon_lr,
                                   verbose=(trial == 0))  # Only verbose for first trial
            accs.append(acc)
            times.append(t)
            print(f"  Trial {trial}: TTA acc = {acc:.4f}, time = {t:.2f}s")

        accs = torch.tensor(accs)
        times = torch.tensor(times)

        results[lambda_reg] = {
            "mean_acc": accs.mean().item(),
            "std_acc": accs.std().item(),
            "mean_time": times.mean().item(),
            "accs": accs.tolist()
        }

        print(f"  Lambda={lambda_reg}: {accs.mean():.4f} ± {accs.std():.4f} ({times.mean():.2f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="SoftMuon CIFAR-10 benchmark")
    parser.add_argument("--lambda-reg", type=float, default=None,
                       help="Single lambda value to test")
    parser.add_argument("--sweep", action="store_true",
                       help="Run full lambda sweep")
    parser.add_argument("--n-trials", type=int, default=5,
                       help="Number of trials per lambda")
    parser.add_argument("--ns-iters", type=int, default=5,
                       help="Newton-Schulz iterations")
    parser.add_argument("--lr", type=float, default=0.24,
                       help="SoftMuon learning rate")
    parser.add_argument("--muon-baseline", action="store_true",
                       help="Run original Muon (lambda=0) as baseline")
    args = parser.parse_args()

    # Initialize model
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")

    # Warmup run
    print("Warming up...")
    train_one_run("warmup", model, lambda_reg=0.1, verbose=False)
    print("Warmup complete.\n")

    if args.sweep:
        # Full sweep over lambda values
        lambda_values = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        results = run_sweep(model, lambda_values, n_trials=args.n_trials,
                           ns_iters=args.ns_iters, softmuon_lr=args.lr)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Lambda':<10} {'Mean Acc':<12} {'Std':<10} {'Time (s)':<10}")
        print("-"*42)

        best_lambda = None
        best_acc = 0
        for lam, r in sorted(results.items()):
            print(f"{lam:<10.4f} {r['mean_acc']:<12.4f} {r['std_acc']:<10.4f} {r['mean_time']:<10.2f}")
            if r['mean_acc'] > best_acc:
                best_acc = r['mean_acc']
                best_lambda = lam

        print("-"*42)
        print(f"Best: lambda={best_lambda} with {best_acc:.4f} accuracy")

    elif args.lambda_reg is not None:
        # Single lambda test
        print(f"Testing lambda={args.lambda_reg}")
        accs = []
        for trial in range(args.n_trials):
            acc, t = train_one_run(trial, model, lambda_reg=args.lambda_reg,
                                   ns_iters=args.ns_iters, softmuon_lr=args.lr)
            accs.append(acc)
            print(f"Trial {trial}: TTA acc = {acc:.4f}, time = {t:.2f}s")

        accs = torch.tensor(accs)
        print(f"\nResult: {accs.mean():.4f} ± {accs.std():.4f}")

    elif args.muon_baseline:
        # Run Muon baseline (lambda=0)
        print("Running Muon baseline (lambda=0)")
        accs = []
        for trial in range(args.n_trials):
            acc, t = train_one_run(trial, model, lambda_reg=0.0,
                                   ns_iters=args.ns_iters, softmuon_lr=args.lr)
            accs.append(acc)
            print(f"Trial {trial}: TTA acc = {acc:.4f}, time = {t:.2f}s")

        accs = torch.tensor(accs)
        print(f"\nMuon baseline: {accs.mean():.4f} ± {accs.std():.4f}")

    else:
        # Default: quick test with lambda=0.1
        print("Quick test with lambda=0.1 (use --sweep for full sweep)")
        acc, t = train_one_run(0, model, lambda_reg=0.1)
        print(f"\nResult: TTA acc = {acc:.4f}, time = {t:.2f}s")


if __name__ == "__main__":
    main()
