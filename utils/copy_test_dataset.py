import os
import shutil
from pathlib import Path

def fill_missing_images(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"not valid: {folder_path}")

    png_files = {}
    for f in folder.glob("*.png"):
        name = f.stem
        if name.isdigit() and len(name) == 4:
            png_files[int(name)] = f

    if not png_files:
        print("not found xxxx.png file")
        return

    existing_numbers = sorted(png_files.keys())
    min_num = min(existing_numbers)
    max_num = max(existing_numbers)

    filled = set(existing_numbers)
    next_existing_map = {}

    current_fill_source = None
    for num in range(max_num, min_num - 1, -1):
        if num in png_files:
            current_fill_source = num
        else:
            if current_fill_source is not None:
                next_existing_map[num] = current_fill_source

    copied_count = 0
    for missing_num, source_num in sorted(next_existing_map.items()):
        source_file = png_files[source_num]
        target_name = f"{missing_num:04d}.png"
        target_path = folder / target_name
        if not target_path.exists():
            shutil.copy2(source_file, target_path)
            print(f"copy {source_file.name} -> {target_name}")
            copied_count += 1
        else:
            print(f"warning：{target_name} exist")

    print(f"\n {copied_count} images filled in total")

# demo
if __name__ == "__main__":
    folder = ('/home/nv/ws/GO-MVF/data/CSRD_O/original_img_backup/hor5_video')
    fill_missing_images(folder)