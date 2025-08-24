import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 전처리 정의
transform = transforms.Compose([
    transforms.ToTensor(),
])

# CIFAR-10 불러오기 (학습 데이터)
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# CIFAR-10 클래스 이름
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 이미지 보여주기 함수
def imshow(img):
    img = img / 2 + 0.5   # Normalize 되었다면 복원
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # CxHxW → HxWxC
    plt.show()

# 데이터에서 샘플 뽑기
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 이미지 표시
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]}' for j in range(8)))

