# Importamos las librerías necesarias
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pytorch_lightning as pl
from label_studio_sdk import Client
from PIL import Image

# Definimos un dataset personalizado para cargar las imágenes y anotaciones
class CustomDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        """
        annotations: lista de anotaciones (provenientes de Label Studio)
        img_dir: directorio donde se encuentran las imágenes
        transform: transformaciones a aplicar a las imágenes
        """
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.label_mapping = self.create_label_mapping()

    def create_label_mapping(self):
        """
        Creates a mapping from string labels to numeric values.
        """
        unique_labels = set(annotation['label'] for annotation in self.annotations)
        return {label: idx for idx, label in enumerate(unique_labels)}

    def label_to_numeric(self, label):
        """
        Converts a string label to its numeric value.
        """
        return self.label_mapping[label]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load the image and annotations
        annotation = self.annotations[idx]
        # Extract the image path
        image_name = os.path.basename(annotation['image'].split("-")[-1])
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        
        # Ensure bounding boxes are 2D tensors of shape [N, 4]
        boxes = torch.tensor(annotation['bbox'], dtype=torch.float32)
        if boxes.ndimension() == 1:  # If it's a single bounding box (1D), reshape to 2D
            boxes = boxes.unsqueeze(0)
        
        # Convert label to tensor
        labels = torch.tensor([self.label_to_numeric(annotation['label'])], dtype=torch.int64)  # Wrap label in a list

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Return image and targets as a dictionary
        targets = {"boxes": boxes, "labels": labels}
        return image, targets

# Definimos el modelo Lightning
class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # Cargamos un modelo preentrenado de torchvision (Faster R-CNN)
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # Ajustamos el número de clases
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        # Forward para entrenamiento o inferencia
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss)
        for key, value in loss_dict.items():
            self.log(f"train_{key}", value)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", loss)
        for key, value in loss_dict.items():
            self.log(f"val_{key}", value)
        return loss
    
    def on_train_end(self):
        print("I'm done training")
    
    def configure_optimizers(self):
        # Optimizador
        return torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Función para cargar anotaciones desde Label Studio
def load_annotations_from_label_studio(api_url, api_key, project_id):
    client = Client(api_url, api_key)
    project = client.get_project(project_id)
    tasks = project.get_labeled_tasks()
    annotations = []
    for task in tasks:
        # Extraemos las anotaciones relevantes
        bbox_data = task['annotations'][0]['result'][0]['value']
        x_min = bbox_data['x']
        y_min = bbox_data['y']
        x_max = x_min + bbox_data['width']
        y_max = y_min + bbox_data['height']
        # Equivocacion intentional para practicar
        #"labels":task['annotations'][0]['result'][0]['value']['labels']
        label = task['annotations'][0]['result'][0]['value']['rectanglelabels'][0]

        annotations.append({
            "image": task['data']['image'],
            "bbox": [x_min, y_min, x_max, y_max],
            "label":label
        })

    return annotations

def collate_fn(batch):
    images, targets = zip(*batch)  # Unzip the batch
    return list(images), list(targets)  # Ensure targets is a list of dicts

# Configuración principal
if __name__ == "__main__":
    # Parámetros
    api_url = "http://localhost:8080"  # URL de Label Studio

    # Cargamos la clave de API desde un archivo en la misma carpeta
    with open(os.path.join(os.path.dirname(__file__), "label_studio_key.txt"), "r") as key_file:
        api_key = key_file.read().strip()

    project_id = 10  # ID del proyecto en Label Studio, cambia esto según tu configuración

    ## Cambia esto a tu directorio de imágenes
    img_dir = "/Users/benweinstein/Downloads/example_airborne_birds"  # Directorio de imágenes
    num_classes = 2  # Número de clases (incluyendo fondo)

    # Cargamos las anotaciones
    annotations = load_annotations_from_label_studio(api_url, api_key,
                                                     project_id)

    # Definimos las transformaciones
    transform = transforms.Compose([transforms.ToTensor()])

    # Creamos el dataset y el dataloader
    dataset = CustomDataset(annotations, img_dir, transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=collate_fn)

    # Inicializamos el modelo
    model = ObjectDetectionModel(num_classes=num_classes)

    # Entrenamos el modelo con PyTorch Lightning
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, dataloader)

    # Evaluate
    # Predict the rest of the images
    # Upload new images to label-studio based on confidence and pre-annotate them