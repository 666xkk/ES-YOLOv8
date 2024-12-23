from ultralytics import YOLO, nn
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect

# Load a model
yolo = YOLO("runs/detect/YOLOv8/weights/last.pt")
model = yolo.model


# Define a function to perform pruning on a convolution layer
def prune_conv(conv_module, amount=0.2):
    # Calculate the number of channels to prune
    num_channels = conv_module.conv.weight.shape[0]
    num_prune = int(num_channels * amount)

    # Get the sorted absolute values of the weights
    weight_abs = torch.abs(conv_module.conv.weight).clone().detach()
    sorted_weights, _ = torch.sort(weight_abs, dim=0, descending=True)

    # Determine the threshold for pruning
    threshold = sorted_weights[num_prune]

    # Create a mask for pruning
    mask = weight_abs > threshold

    # Apply the mask to the weights and biases
    conv_module.conv.weight.data[mask] = 0
    if conv_module.conv.bias is not None:
        conv_module.conv.bias.data[~mask] = 0

    # Update the number of features in the batch normalization layer
    conv_module.bn.num_features = num_channels - num_prune


# Define a function to prune the model
def prune_model(model,amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune_conv(module)
        elif isinstance(module, nn.Sequential):
            # Sequential 模块需要进一步遍历
            for i, sub_module in enumerate(module):
                if isinstance(sub_module, nn.Conv2d):
                    prune_conv(sub_module)

# 执行剪枝
prune_model(model,amount=0.5)

# Save the pruned model
torch.save(model.state_dict(), "runs/detect/YOLOv8/weights/pruned_model.pt")

# Validate the pruned model
yolo.val()

# Export the pruned model to ONNX format
yolo.export(format="onnx")

# Optionally, retrain the pruned model
# yolo.train(data="./data/data_nc5/data_nc5.yaml", epochs=100)
