import requests
import time
import os
import sys
import uuid
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(folder_path):
    images = []
    for f in sorted(os.listdir(folder_path)):
        if Path(f).suffix.lower() in IMAGE_EXTS:
            images.append(f)
    return images


def main():
    port = input("请输入服务器端口号: ").strip()
    folder_path = input("请输入图片文件夹路径: ").strip()

    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        sys.exit(1)

    images = collect_images(folder_path)
    if not images:
        print("文件夹内没有找到图片文件")
        sys.exit(1)

    print(f"共找到 {len(images)} 张图片，开始发送推理请求...")
    print(f"服务器地址: http://127.0.0.1:{port}/industry/image_defect")
    print("-" * 60)

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    sample_id = f"S{uuid.uuid4().hex[:7].upper()}"
    pose_id = "pose_1"

    success_count = 0
    fail_count = 0
    total_defects = 0

    for idx, img_name in enumerate(images, 1):
        payload = {
            "job_id": job_id,
            "sample_id": sample_id,
            "pose_id": pose_id,
            "file_names": [img_name],
            "relative_dir": folder_path,
        }

        try:
            resp = requests.post(
                f"http://127.0.0.1:{port}/industry/image_defect",
                json=payload,
                timeout=60,
            )
            data = resp.json()
            error_code = data.get("error_code", -1)
            results = data.get("data", {}).get("results", [])

            if error_code == 0:
                success_count += 1
                total_defects += len(results)
                status = f"OK, {len(results)} defects"
            else:
                fail_count += 1
                status = f"ERROR: {data.get('error_msg', 'unknown')}"

        except Exception as e:
            fail_count += 1
            status = f"EXCEPTION: {e}"

        print(f"[{idx}/{len(images)}] {img_name} -> {status}")

        if idx < len(images):
            time.sleep(0.5)

    print("-" * 60)
    print(f"完成! 成功: {success_count}, 失败: {fail_count}, 总缺陷数: {total_defects}")


if __name__ == "__main__":
    main()
