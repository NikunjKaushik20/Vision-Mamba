from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {'GPU (CUDA)' if device == '0' else 'CPU'}")

    # Fine-tune from clean YOLOv8 nano model instead of corrupted checkpoint
    model = YOLO("weights/yolov8n.pt")

    results = model.train(
        data     = "d:/Vision-Mamba/configs/fracatlas_yolo.yaml",
        epochs   = 100,
        patience = 30,
        imgsz    = 640,
        batch    = 4,       # Reduced to 4 to save memory and prevent OOM crash
        workers  = 0,       # 0 workers prevents OpenCV RAM Out-Of-Memory crash on Windows
        device   = device,

        optimizer     = "auto", # auto uses SGD by default, saving VRAM compared to AdamW
        warmup_epochs = 3,
        cos_lr        = True,

        hsv_h     = 0.01,
        hsv_s     = 0.3,
        hsv_v     = 0.4,
        degrees   = 15,
        translate = 0.1,
        scale     = 0.5,
        fliplr    = 0.5,
        flipud    = 0.1,
        mosaic    = 1.0,
        mixup     = 0.1,
        copy_paste= 0.0,

        weight_decay = 5e-4,
        dropout      = 0.0,

        project  = "runs/detect",
        name     = "fracture_yolo_ft",  # new folder so best.pt is not overwritten
        exist_ok = True,
        save     = True,
        plots    = True,
        conf     = 0.25,
        iou      = 0.5,
    )

    print("\n✅ Training complete!")
    print("Best model: runs/detect/fracture_yolo_ft/weights/best.pt")
    metrics = model.val()
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
