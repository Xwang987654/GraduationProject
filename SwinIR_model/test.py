import os
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): 存放图片的文件夹路径
            transform (callable, optional): 图像预处理/变换模块
        """
        self.folder_path = folder_path
        # 筛选出常见的图片格式
        self.image_files = [f for f in os.listdir(folder_path)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. 获取图片路径
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)

        # 2. 读取图片并转为 RGB (防止某些 PNG 是 RGBA 或灰度)
        image = Image.open(img_path).convert('RGB')

        # 3. 应用预处理 (转为 Tensor 等)
        if self.transform:
            image = self.transform(image)

        return image, img_name


if __name__ == '__main__':
    # --- 使用示例 ---

    parser = argparse.ArgumentParser(description='图片文件夹数据加载测试')
    parser.add_argument('--folder', type=str, required=True, help='存放图片的文件夹路径')
    args = parser.parse_args()

    # 1. 定义预处理流程
    # 注意：如果要 Batch 处理，Resize 是必须的，否则图片尺寸不同无法打包成一个 Tensor
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 缩放到统一尺寸
        transforms.ToTensor(),  # 转化为 Tensor 且数值归一化到 [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 2. 实例化 Dataset
    dataset = ImageFolderDataset(folder_path=args.folder, transform=data_transform)

    # 3. 实例化 DataLoader (这就是你批量取数据的地方)
    data_loader = DataLoader(
        dataset,
        batch_size=4,  # 每一批处理 4 张图，3060 显卡可以根据显存调大
        shuffle=False,  # 推理任务通常不需要打乱
        num_workers=2  # 多线程读取
    )

    # 4. 在神经网络中使用
    for images, names in data_loader:
        # images 的形状是 [4, 3, 224, 224]
        # outputs = model(images)
        print(f"Batch processed: {names}")