from torchvision.models.segmentation import deeplabv3_resnet101
model=deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)

bus=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /Github_bachelor/Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/1000352404.jpg')
bus=np.array(bus)

image_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
image=image_transform(bus)

model.eval()
output=model(image.unsqueeze(0).float())
output=output['out'][0]
output_predictions = output.argmax(0)
input_image=bus
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
plt.imshow(r)