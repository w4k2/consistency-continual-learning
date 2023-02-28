import torch
import resnet
import matplotlib.pyplot as plt

model = resnet.resnet18(num_classes=100)

angles = list()
norms = list()
classes = list(range(100))  # (20, 40, 50, 80, 99)

with torch.no_grad():
    for class_idx in classes:
        norms.append(list())
        angles.append(list())
        for i in range(20):
            model.load_state_dict(torch.load(f'resnet_after_{i}.pth'))
            weights = model.fc.weight
            vector = weights[class_idx]
            norm = torch.norm(vector, p=2).item()
            # print(f'norm after task {i} =', norm)
            norms[-1].append(norm)
            versor = torch.eye(len(vector), 1).flatten()
            angle = torch.arccos(torch.dot(vector, versor) / norm)
            angles[-1].append(angle)

plt.subplot(2, 1, 1)
for angle, class_idx in zip(angles, classes):
    plt.plot(angle, label=f'class {class_idx}')
plt.xlabel('task')
plt.ylabel('weights vector angle')
plt.legend()

plt.subplot(2, 1, 2)
for class_norms, class_idx in zip(norms, classes):
    plt.plot(class_norms, label=f'class {class_idx}')
plt.xlabel('task')
plt.ylabel('L2 norm of last class weights')
plt.legend()

plt.show()
