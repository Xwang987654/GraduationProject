import os
import sys
from pathlib import Path

def rename_images(folder_path, start=1, extensions=None, sort_by='name', preview=False):
    """
    将指定文件夹内的所有图片重命名为数字序号（1, 2, 3, ...）

    参数:
        folder_path (str): 目标文件夹路径
        start (int): 起始序号，默认为1
        extensions (list): 图片扩展名列表，默认常见格式 ['.jpg','.jpeg','.png','.gif','.bmp','.tiff','.webp']
        sort_by (str): 排序方式，可选 'name'（按文件名）、'mtime'（按修改时间）、'ctime'（按创建时间）
        preview (bool): 为True时只打印重命名计划而不实际执行，默认为False
    返回:
        int: 成功重命名的文件数量
    """
    # 默认支持的图片扩展名
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg']
    else:
        # 确保扩展名都是小写且以点开头
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"错误：文件夹 '{folder_path}' 不存在或不是一个目录。")
        return 0

    # 收集所有图片文件（不递归子文件夹）
    all_files = [f for f in folder.iterdir() if f.is_file()]
    image_files = []
    for f in all_files:
        if f.suffix.lower() in extensions:
            image_files.append(f)

    if not image_files:
        print(f"在 '{folder_path}' 中未找到任何支持的图片文件。")
        return 0

    # 排序
    if sort_by == 'name':
        image_files.sort(key=lambda x: x.name)
    elif sort_by == 'time':
        image_files.sort(key=lambda x: x.stat().st_mtime)
    elif sort_by == 'ctime':
        image_files.sort(key=lambda x: x.stat().st_ctime)
    else:
        print(f"不支持的排序方式 '{sort_by}'，将使用文件名排序。")
        image_files.sort(key=lambda x: x.name)

    # 检查目标文件名是否已存在，避免覆盖
    existing_names = {f.name for f in folder.iterdir() if f.is_file()}
    renamed_count = 0
    current_num = start

    print(f"找到 {len(image_files)} 张图片，准备重命名...\n")

    for old_path in image_files:
        # 生成新文件名：数字 + 原扩展名
        while True:
            new_name = f"{current_num}{old_path.suffix}"
            new_path = folder / new_name
            # if new_path.exists():
            #     # 目标文件已存在，序号+1后继续尝试
            #     print(f"  警告：'{new_name}' 已存在，跳过序号 {current_num}")
            #     current_num += 1
            # else:
            break

        # 执行重命名（预览模式下仅打印）
        if preview:
            print(f"  [预览] 将重命名: {old_path.name} -> {new_name}")
        else:
            try:
                old_path.rename(new_path)
                print(f"  [成功] {old_path.name} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"  [失败] {old_path.name} 重命名出错: {e}")
                # 跳过此文件，序号不变（下次尝试同一序号）
                continue

        current_num += 1

    print(f"\n操作完成。共重命名 {renamed_count} 个文件。")
    return renamed_count


if __name__ == '__main__':
    # 从命令行参数获取文件夹路径（如果没有则使用当前目录）
    folder  = 'datasets/GT'

    rename_images(folder)