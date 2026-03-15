"""Create a Kaggle-compatible zip of the FracAtlas YOLO dataset (forward slashes)."""
import zipfile
import os

root = os.path.join(os.path.dirname(__file__), "..", "data", "fracatlas_yolo")
root = os.path.abspath(root)
out = os.path.join(os.path.dirname(__file__), "..", "fracatlas_yolo.zip")
out = os.path.abspath(out)

print(f"Zipping: {root}")
print(f"Output:  {out}")

count = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            arcname = os.path.relpath(full, root).replace("\\", "/")
            zf.write(full, arcname)
            count += 1
            if count % 500 == 0:
                print(f"  {count} files zipped...")

size_mb = os.path.getsize(out) / 1e6
print(f"\n✅ Done! {count} files → {size_mb:.0f} MB")
print(f"   File: {out}")
