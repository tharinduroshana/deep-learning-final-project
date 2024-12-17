import os
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from torchvision import models
from part_a.dual_image_model import DualImageModel
from part_a.single_image_model import SingleImageModel
from shared.shared import transform_test
import torch.nn as nn

pre_trained_models = {
    "resnet18": models.resnet18(pretrained=True),
    "resnet34": models.resnet34(pretrained=True),
    "vgg16": models.vgg16(pretrained=True),
    "efficientnet_b0": models.efficientnet_b0(pretrained=True),
    "densenet121": models.densenet121(pretrained=True),
}


class SecondStageModel(nn.Module):
    def __init__(self, num_classes=5, model="resnet18", dropout_rate=0.5):
        super().__init__()

        backbone = pre_trained_models[model]

        backbone.fc = nn.Identity()

        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)
        self.backbone3 = copy.deepcopy(backbone)
        self.backbone4 = copy.deepcopy(backbone)

        if model not in ["resnet18", "resnet34"]:
            input_features = 1000
        else:
            input_features = 512

        self.fc = nn.Sequential(
            nn.Linear(input_features * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, images):
        image1, image2, image3, image4 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)
        x3 = self.backbone3(image3)
        x4 = self.backbone4(image4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc(x)
        return x


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output

    def forward(self, input_image, target_class=None):
        self.model.eval()

        input_image = input_image.unsqueeze(0)
        output = self.model(input_image)

        if target_class is None:
            target_class = torch.argmax(output)

        heatmap = self.generate_heatmap(output)

        return heatmap, target_class

    def forward_dual_image(self, input_image1, input_image2, target_class=None):
        self.model.eval()

        input_image = input_image1.unsqueeze(0)
        input_image2 = input_image2.unsqueeze(0)
        output = self.model([input_image, input_image2])

        if target_class is None:
            target_class = torch.argmax(output)

        heatmap = self.generate_heatmap(output)

        return heatmap, target_class

    def forward_patient_level(self, input_image1, input_image2, input_image3, input_image4, target_class=None):
        self.model.eval()

        input_image = input_image1.unsqueeze(0)
        input_image2 = input_image2.unsqueeze(0)
        input_image3 = input_image3.unsqueeze(0)
        input_image4 = input_image4.unsqueeze(0)

        output = self.model([input_image, input_image2, input_image3, input_image4])

        if target_class is None:
            target_class = torch.argmax(output)

        heatmap = self.generate_heatmap(output)

        return heatmap, target_class

    def generate_heatmap(self, model_output):
        self.model.zero_grad()

        pred_class = model_output.argmax(dim=1).item()
        model_output[:, pred_class].backward()

        gradients = self.gradients[0]
        feature_maps = self.feature_maps
        weights = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(feature_maps.size()[1]):
            feature_maps[:, i, :, :] *= weights[i]

        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap

    @staticmethod
    def overlay_heatmap(img, grad_cam):
        grad_cam = grad_cam.detach().cpu().numpy()

        if grad_cam.ndim > 2:
            grad_cam = np.mean(grad_cam, axis=0)

        grad_cam_resized = cv2.resize(grad_cam, (img.shape[1], img.shape[0]))
        grad_cam_resized = np.uint8(255 * grad_cam_resized)
        heatmap = cv2.applyColorMap(grad_cam_resized, cv2.COLORMAP_JET)
        overlayed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        return overlayed_img


def plot_overlayed_image(img, predicted_class, save_path='./output'):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f"Grad-CAM for Predicted Class: {predicted_class.item()}")
    plt.axis('off')

    checkpoint_dir = save_path
    os.makedirs(checkpoint_dir, exist_ok=True)

    plt.savefig(f'{save_path}/gradcam.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def generate_gradcam_heatmap(model_name, model, training_mode, trained_model_path):
    model.load_state_dict(torch.load(trained_model_path, map_location='cpu',
                                     weights_only=False))

    if training_mode == "single":
        if model_name == "resnet18":
            target_layer = model.backbone.layer4
        if model_name == "resnet34":
            target_layer = model.backbone.layer4
        elif model_name == "densenet121":
            target_layer = model.backbone.features
        elif model_name == "vgg16":
            target_layer = model.backbone.features[28]
        elif model_name == "efficientnet_b0":
            target_layer = model.backbone.features
    else:
        if model_name == "resnet18":
            target_layer = model.backbone1.layer4
        if model_name == "resnet34":
            target_layer = model.backbone1.layer4
        elif model_name == "densenet121":
            target_layer = model.backbone1.features
        elif model_name == "vgg16":
            target_layer = model.backbone1.features[28]
        elif model_name == "efficientnet_b0":
            target_layer = model.backbone1.features

    grad_cam = GradCAM(model, target_layer)

    img_path = '../DeepDRiD/val/277/277_l1.jpg'
    input_img = Image.open(img_path).convert('RGB')
    image = np.array(input_img)
    image_tensor = transform_test(input_img)

    if training_mode == "single":
        grad_cam_heatmap, predicted_class = grad_cam.forward(image_tensor)
    if training_mode == "dual":
        img_path2 = '../DeepDRiD/val/277/277_l2.jpg'
        input_img2 = Image.open(img_path2).convert('RGB')
        image2 = np.array(input_img2)
        image_tensor2 = transform_test(input_img2)
        grad_cam_heatmap, predicted_class = grad_cam.forward_dual_image(image_tensor, image_tensor2)
    elif training_mode == "patient_level":
        img_path2 = '../DeepDRiD/val/277/277_l2.jpg'
        input_img2 = Image.open(img_path2).convert('RGB')
        image2 = np.array(input_img2)
        image_tensor2 = transform_test(input_img2)

        img_path3 = '../DeepDRiD/val/277/277_r1.jpg'
        input_img3 = Image.open(img_path3).convert('RGB')
        image3 = np.array(input_img3)
        image_tensor3 = transform_test(input_img3)

        img_path4 = '../DeepDRiD/val/277/277_r2.jpg'
        input_img4 = Image.open(img_path4).convert('RGB')
        image4 = np.array(input_img2)
        image_tensor4 = transform_test(input_img4)
        grad_cam_heatmap, predicted_class = grad_cam.forward_patient_level(image_tensor, image_tensor2, image_tensor3,
                                                                           image_tensor4)

    overlayed_img = grad_cam.overlay_heatmap(image, grad_cam_heatmap)

    plot_overlayed_image(overlayed_img, predicted_class,
                         save_path=f'output/{model_name.lower()}_{training_mode.lower()}')


if __name__ == '__main__':
    model = DualImageModel("resnet18")
    generate_gradcam_heatmap("resnet18", model, "dual", '../trained_models_task_a/resnet18/dual/resnet13_dual.pth')

    # model = DualImageModel("densenet121")
    # generate_gradcam_heatmap("densenet121", model, "dual", './trained_models_task_a/densenet121/dual/densenet121_dual.pth')

    # model = SingleImageModel("vgg16")
    # generate_gradcam_heatmap("vgg16", model, "single",
    #                          './trained_models_task_a/vgg16/single/vgg16.pth')

    # model = SingleImageModel("resnet34")
    # generate_gradcam_heatmap("resnet34", model, "single",
    #                          './trained_models_task_a/resnet34/single/resnet34.pth')

    # model = SingleImageModel("resnet18")
    # generate_gradcam_heatmap("resnet18", model, "single", './trained_models_task_a/resnet18/single/resnet13.pth')

    # model = SecondStageModel(model="densenet121")
    # generate_gradcam_heatmap("resnet18", model, "patient_level", './trained_models_b/resnet18/second_stage_pt_level/second_stage_resnet18_pt_level.pth')

    # model = SecondStageModel(model="efficientnet_b0")
    # generate_gradcam_heatmap("efficientnet_b0", model, "patient_level", './trained_models_b/efficientnet_b0/second_stage_pt_level/second_stage_eff_b0_pt_level.pth')
