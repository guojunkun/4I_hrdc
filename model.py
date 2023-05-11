import os
import cv2
import torch
from torch import nn
import torchvision.models as models


class model:
    def __init__(self):
        self.checkpoint = "model3_2.pt"
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        make sure these files are in the same directory as the model.py file.
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        resnet = models.resnet34(pretrained=True)
        resnet.add_module("final_fc1", nn.Linear(1000, 50))
        resnet.add_module("final_fc2", nn.Linear(50, 2))
        self.model = resnet
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        image = cv2.resize(input_image, (512, 512))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image / 255
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, torch.float)

        with torch.no_grad():
            score = self.model(image)
        pred_class = score.argmax(dim=1)

        pred_class = pred_class.detach().cpu()
        pred_class = int(pred_class)
        return pred_class




