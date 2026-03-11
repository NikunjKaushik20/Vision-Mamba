
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import load_config


class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, config=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.config = config
        self.split = split
        
        # Discover classes and images
        self.classes = sorted([
            d for d in os.listdir(self.root_dir) 
            if os.path.isdir(self.root_dir / d)
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Build image list
        self.images = []
        self.labels = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    self.images.append(str(cls_dir / img_name))
                    self.labels.append(self.class_to_idx[cls])
        
        self.labels = np.array(self.labels)
        
        # Compute class weights for balanced sampling
        class_counts = np.bincount(self.labels, minlength=len(self.classes))
        self.class_weights = 1.0 / (class_counts + 1e-6)
        self.class_weights = self.class_weights / self.class_weights.sum()
        self.sample_weights = self.class_weights[self.labels]
        
        print(f"[DATA] {split}: {len(self.images)} images, {len(self.classes)} classes")
        for cls, idx in self.class_to_idx.items():
            count = (self.labels == idx).sum()
            print(f"       Class '{cls}' (idx={idx}): {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback: try with PIL
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, torch.tensor(label, dtype=torch.long)


def get_clahe_transform(clip_limit=2.0, tile_size=8):
    """Apply CLAHE for X-ray contrast enhancement."""
    return A.CLAHE(clip_limit=clip_limit, tile_grid_size=(tile_size, tile_size), p=1.0)


def get_train_transforms(config):
    """Build training augmentation pipeline using Albumentations."""
    aug_config = config["augmentation"]["train"]
    norm_config = config["augmentation"]["normalize"]
    img_size = config["data"]["image_size"]
    
    transform_list = [
        A.Resize(img_size, img_size),
    ]
    
    # CLAHE for X-ray contrast enhancement
    if aug_config.get("clahe", {}).get("enabled", True):
        clahe_cfg = aug_config["clahe"]
        transform_list.append(
            get_clahe_transform(clahe_cfg.get("clip_limit", 2.0), clahe_cfg.get("tile_grid_size", 8))
        )
    
    # Geometric augmentations
    if aug_config.get("horizontal_flip", 0) > 0:
        transform_list.append(A.HorizontalFlip(p=aug_config["horizontal_flip"]))
    
    if aug_config.get("rotation", 0) > 0:
        transform_list.append(A.Rotate(limit=aug_config["rotation"], p=0.5, border_mode=cv2.BORDER_REFLECT_101))
    
    if aug_config.get("affine"):
        aff = aug_config["affine"]
        transform_list.append(A.Affine(
            translate_percent={"x": (-aff["translate"][0], aff["translate"][0]),
                             "y": (-aff["translate"][1], aff["translate"][1])},
            scale=(aff["scale"][0], aff["scale"][1]),
            rotate=(-aff.get("shear", 5), aff.get("shear", 5)),
            p=0.4,
            border_mode=cv2.BORDER_REFLECT_101,
        ))
    
    # Color augmentations
    if aug_config.get("color_jitter"):
        cj = aug_config["color_jitter"]
        transform_list.append(A.ColorJitter(
            brightness=cj["brightness"],
            contrast=cj["contrast"],
            saturation=cj["saturation"],
            hue=cj["hue"],
            p=0.4,
        ))
    
    # Gaussian blur
    if aug_config.get("gaussian_blur", {}).get("probability", 0) > 0:
        gb = aug_config["gaussian_blur"]
        ks = gb.get("kernel_size", 3)
        transform_list.append(A.GaussianBlur(blur_limit=(ks, ks), p=gb["probability"]))
    
    # Additional medical-imaging augmentations
    transform_list.extend([
        A.GaussNoise(p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.15),
    ])
    
    # CoarseDropout (similar to Random Erasing)
    if aug_config.get("random_erasing", {}).get("probability", 0) > 0:
        re = aug_config["random_erasing"]
        try:
            # albumentations >= 1.4
            transform_list.append(A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(int(img_size * 0.02), int(img_size * 0.15)),
                hole_width_range=(int(img_size * 0.02), int(img_size * 0.15)),
                fill=0,
                p=re["probability"],
            ))
        except TypeError:
            # albumentations < 1.4 fallback
            transform_list.append(A.CoarseDropout(
                max_holes=4,
                max_height=int(img_size * 0.15),
                max_width=int(img_size * 0.15),
                min_holes=1,
                min_height=int(img_size * 0.02),
                min_width=int(img_size * 0.02),
                fill_value=0,
                p=re["probability"],
            ))
            
    # Real-world WhatsApp / Camera Monitor Photo Augmentations
    if aug_config.get("real_world_robustness", {}).get("enabled", False):
        rw = aug_config["real_world_robustness"]
        
        # JPEG WhatsApp Compression
        if rw.get("jpeg_compression", {}).get("probability", 0) > 0:
            jc = rw["jpeg_compression"]
            transform_list.append(A.ImageCompression(quality_lower=jc["quality_lower"], quality_upper=jc["quality_upper"], p=jc["probability"]))
            
        # Perspective shift (from taking photo of a monitor from an angle)
        if rw.get("perspective", {}).get("probability", 0) > 0:
            p = rw["perspective"]
            transform_list.append(A.Perspective(scale=p["scale"], p=p["probability"]))
            
        # Camera blur from shaky hands
        if rw.get("motion_blur", {}).get("probability", 0) > 0:
            mb = rw["motion_blur"]
            transform_list.append(A.MotionBlur(blur_limit=mb["blur_limit"], p=mb["probability"]))
            
        # Cheap smartphone sensor ISO noise
        if rw.get("iso_noise", {}).get("probability", 0) > 0:
            iso = rw["iso_noise"]
            transform_list.append(A.ISONoise(color_shift=iso["color_shift"], intensity=iso["intensity"], p=iso["probability"]))
            
        # Monitor / Overheard hospital lights glare
        if rw.get("sun_flare", {}).get("probability", 0) > 0:
            sf = rw["sun_flare"]
            transform_list.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=3, src_radius=150, src_color=(255, 255, 255), p=sf["probability"]))
            
        # Moiré lines / Monitor grid
        if rw.get("grid_dropout", {}).get("probability", 0) > 0:
            gd = rw["grid_dropout"]
            transform_list.append(A.GridDropout(unit_size_min=gd["unit_size"][0], unit_size_max=gd["unit_size"][1], random_offset=gd["random_offset"], p=gd["probability"]))
    
    # Normalize and convert to tensor
    transform_list.extend([
        A.Normalize(mean=norm_config["mean"], std=norm_config["std"]),
        ToTensorV2(),
    ])
    
    return A.Compose(transform_list)


def get_val_transforms(config):
    """Build validation/test transform pipeline (no augmentation)."""
    norm_config = config["augmentation"]["normalize"]
    img_size = config["data"]["image_size"]
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=norm_config["mean"], std=norm_config["std"]),
        ToTensorV2(),
    ])


def get_tta_transforms(config):
    """Build Test-Time Augmentation (TTA) transforms."""
    norm_config = config["augmentation"]["normalize"]
    img_size = config["data"]["image_size"]
    
    tta_list = []
    
    # Original
    tta_list.append(A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=norm_config["mean"], std=norm_config["std"]),
        ToTensorV2(),
    ]))
    
    # Horizontal flip
    tta_list.append(A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=norm_config["mean"], std=norm_config["std"]),
        ToTensorV2(),
    ]))
    
    # Slight rotation +
    tta_list.append(A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Rotate(limit=(5, 5), p=1.0, border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=norm_config["mean"], std=norm_config["std"]),
        ToTensorV2(),
    ]))
    
    # Slight rotation -
    tta_list.append(A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Rotate(limit=(-5, -5), p=1.0, border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=norm_config["mean"], std=norm_config["std"]),
        ToTensorV2(),
    ]))
    
    # Slightly zoomed
    tta_list.append(A.Compose([
        A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
        A.CenterCrop(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=norm_config["mean"], std=norm_config["std"]),
        ToTensorV2(),
    ]))
    
    return tta_list


class MixupCutmix:
    def __init__(self, mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5, num_classes=2):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
    
    def __call__(self, images, labels):
        batch_size = images.size(0)
        
        # Convert to one-hot
        labels_onehot = torch.zeros(batch_size, self.num_classes, device=images.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        if np.random.random() > self.prob:
            return images, labels_onehot
        
        # Randomly choose mixup or cutmix
        if np.random.random() < 0.5:
            return self._mixup(images, labels_onehot)
        else:
            return self._cutmix(images, labels_onehot)
    
    def _mixup(self, images, labels):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5
        
        indices = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels
    
    def _cutmix(self, images, labels):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        indices = torch.randperm(images.size(0), device=images.device)
        
        _, _, H, W = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        
        # Adjust lambda to reflect the actual area ratio
        lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels


def get_dataloaders(config):
    """
    Create train, validation, and test data loaders.
    
    Returns:
        train_loader, val_loader, test_loader, class_names, class_weights
    """
    data_config = config["data"]
    train_config = config["training"]
    
    root_dir = data_config["root_dir"]
    batch_size = train_config["batch_size"]
    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", True)
    
    # Build transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    # Create datasets
    train_dataset = BoneFractureDataset(root_dir, split="train", transform=train_transform, config=config)
    val_dataset = BoneFractureDataset(root_dir, split="val", transform=val_transform, config=config)
    test_dataset = BoneFractureDataset(root_dir, split="test", transform=val_transform, config=config)
    
    # Weighted sampler for class imbalance
    sample_weights = torch.DoubleTensor(train_dataset.sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Compute class weights for focal loss
    class_counts = np.bincount(train_dataset.labels, minlength=len(train_dataset.classes))
    total = class_counts.sum()
    class_weights = torch.FloatTensor(total / (len(train_dataset.classes) * class_counts))
    
    class_names = train_dataset.classes
    
    print(f"\n[DATA] DataLoaders created:")
    print(f"       Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"       Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"       Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"       Classes: {class_names}")
    print(f"       Class weights: {class_weights.tolist()}")
    
    return train_loader, val_loader, test_loader, class_names, class_weights
