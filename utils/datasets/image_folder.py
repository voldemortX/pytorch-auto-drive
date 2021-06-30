import torchvision
import os
from PIL import Image


# Load a directory of images for inference
class ImageFolderDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, output_dir, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.output_dir = output_dir
        self.filenames = []
        self.images = []
        for filename in sorted(os.listdir(root)):
            self.filenames.append(filename)
            self.images.append(os.path.join(root, filename))

    def __getitem__(self, index):
        # Return x (input image) & y (filename for prediction results)
        img = Image.open(self.images[index]).convert('RGB')
        filename = os.path.join(self.output_dir, self.filenames[index])

        # Transforms
        if self.transforms is not None:
            img = self.transforms(img)

        return img, filename

    def __len__(self):
        return len(self.images)
