"""
CABLE DETECTION - OPTIMIZED VERSION (DETECTRON2)
================================================
Questo script implementa una pipeline completa per la segmentazione di cavi usando Detectron2.
Include:
1. Setup e Installazione automatica.
2. Configurazione personalizzata per oggetti sottili (Hyperparameter Tuning).
3. Data Loaders custom con Augmentation avanzata (Albumentations).
4. Training Loop con monitoraggio e checkpointing.
5. Inferenza con Post-Processing (RANSAC Line Fitting).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Seleziona la GPU 0

import warnings
warnings.filterwarnings('ignore')

import subprocess
import sys

# ============================================================================
# SECTION 0: INSTALLATION
# ============================================================================
# Verifica se Detectron2 è installato, altrimenti lo installa al volo.
# Fondamentale per portabilità su Colab/Kaggle.
print("Checking dependencies...")
try:
    import detectron2
    import sklearn
    print("✓ Detectron2 already installed")
    NEEDS_INSTALL = False
except (ImportError, ValueError):
    print("Installing dependencies...")
    NEEDS_INSTALL = True

if NEEDS_INSTALL:
    # Installa dipendenze base
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                          "numpy>=1.26,<2.0", "pyyaml>=6.0", "scikit-learn", 
                          "matplotlib", "opencv-python-headless", "pycocotools", "albumentations"])
    # Installa Detectron2 da GitHub (attenzione alle compatibilità CUDA)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                          "git+https://github.com/facebookresearch/detectron2.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "segment-anything"])
    print("\n✅ Installation complete. RESTART KERNEL NOW!")
    sys.exit(0)

# ============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ============================================================================

import os
import json
import numpy as np
import torch
import cv2
import time
from pathlib import Path
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils

try:
    import albumentations as A
except ImportError:
    A = None

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import BoxMode, Instances, Boxes, BitMasks
from detectron2.data import detection_utils as utils
import copy
import detectron2.data.transforms as T


print("✓ Imports successful")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU detected")

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class Config:
    # PERCORSI DATI
    DATA_ROOT = "/kaggle/input/cableimages/progetto_cv_2025_2026"
    TRAIN_JSON = f"{DATA_ROOT}/train/train.json"
    TEST_JSON = f"{DATA_ROOT}/test/test.json"
    TRAIN_IMAGES = f"{DATA_ROOT}/train"
    TEST_IMAGES = f"{DATA_ROOT}/test"
    OUTPUT_DIR = "/kaggle/working/output"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    
    # TRAINING SETTINGS
    QUICK_TEST = False       # Set True per debug veloce (poche epoche/img)
    SKIP_TRAINING = False    # Set True se vuoi fare solo inferenza
    
    BATCH_SIZE = 12          # Batch size alto per stabilizzare BatchNorm sui cavi
    NUM_WORKERS = 2
    EPOCHS = 10 if QUICK_TEST else 30 
    LEARNING_RATE = 0.00075  # LR basso per Fine-Tuning
    MAX_TRAIN_SAMPLES = 100 if QUICK_TEST else None
    CHECKPOINT_PERIOD = 10
    
    # MODELLO SCELTO: Mask R-CNN con ResNeXt-101
    # 32x8d_FPN_3x: Backbone molto larga e profonda, ottima per feature complesse.
    MODEL_NAME = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    NUM_CLASSES = 1 # Classe 'cable'
    
    # Mixed Precision Training (FP16)
    USE_AMP = True
    
    # INFERENZA & POST-PROC
    SCORE_THRESHOLD = 0.15  # Accettiamo predizioni anche poco certe, poi filtriamo geometricamente
    MIN_MASK_AREA = 120     # Filtro via macchie troppo piccole
    LINE_FITTING = "ransac" # Algoritmo per estrarre la linea
    
    MATRICOLA = "263780"
    RESUME_FROM = None
    
    # PREPROCESSING FLAGS
    USE_AUGMENTATION = True  # Usa Albumentations
    USE_PREPROCESSING = True # Usa CLAHE + Sharpening

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

print(f"\n⚙️ Config: {cfg.EPOCHS} epochs, Batch {cfg.BATCH_SIZE}, Augmentation {'ON' if cfg.USE_AUGMENTATION else 'OFF'}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_detectron2_dicts(json_path: str, img_dir: str):
    """
    Carica annotazioni COCO e le converte nel formato interno di Detectron2.
    Gestisce sia annotazioni a poligono che RLE.
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    dataset_dicts = []
    for img_info in coco_data['images']:
        record = {
            "file_name": os.path.join(img_dir, img_info['file_name']),
            "image_id": img_info['id'],
            "height": img_info['height'],
            "width": img_info['width'],
        }
        
        anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_info['id']]
        
        objs = []
        for ann in anns:
            if isinstance(ann['segmentation'], list):
                segmentation = ann['segmentation']
            elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                rle = ann['segmentation']
                mask = mask_utils.decode(rle)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = [c.flatten().tolist() for c in contours]
            else:
                continue
            
            objs.append({
                "bbox": ann['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS, # Detectron2 vuole sapere il formato
                "segmentation": segmentation,
                "category_id": 0,
                "iscrowd": 0,
            })
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


def fit_line_ransac(mask: np.ndarray, iterations: int = 100, threshold: float = 3.0):
    """
    Calcola la linea mediana (Rho, Theta) usando RANSAC.
    Input: Maschera binaria (numpy array)
    Output: rho, theta
    """
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) < 2: return 0.0, 0.0
    
    points = np.column_stack([x_coords, y_coords])
    
    # Se ci sono troppi pochi punti, RANSAC fallisce -> usa minimi quadrati semplici
    if len(points) < 100:
        return fit_line_leastsquares(mask)
    
    best_inliers = 0
    best_rho, best_theta = 0, 0
    
    # RANSAC manuale semplificato (o si usa sklearn.linear_model.RANSACRegressor)
    for _ in range(iterations):
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]
        
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if dx == 0 and dy == 0: continue
        
        length = np.sqrt(dx**2 + dy**2)
        normal = np.array([-dy/length, dx/length])
        
        rho = np.abs(np.dot(normal, p1))
        theta = np.arctan2(normal[1], normal[0]) % np.pi
        
        # Conta quanti punti sono vicini a questa linea
        a, b = np.cos(theta), np.sin(theta)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] - rho)
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_rho, best_theta = rho, theta
    
    return float(best_rho), float(best_theta)


def fit_line_leastsquares(mask: np.ndarray):
    """Fallback: PCA/SVD (Least Squares) per trovare la linea principale."""
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) < 2: return 0.0, 0.0
    
    points = np.column_stack([x_coords, y_coords])
    mean = points.mean(axis=0)
    centered = points - mean
    
    _, _, Vt = np.linalg.svd(centered)
    direction = Vt[0]
    normal = np.array([-direction[1], direction[0]])
    
    if np.dot(normal, mean) < 0: normal = -normal
    
    rho = np.abs(np.dot(normal, mean))
    theta = np.arctan2(normal[1], normal[0]) % np.pi
    
    return float(rho), float(theta)


def preprocess_image(img):
    """
    Preprocessing per risaltare cavi: CLAHE + Sharpening.
    """
    # 1. CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. Sharpening (Kernel laplaciano)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)
    
    return np.clip(img, 0, 255).astype(np.uint8)

def get_cable_augmentation():
    """Augmentation pipeline con Albumentations."""
    if A is None: return None
    
    return A.Compose([
        A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_REFLECT), # Rotazioni importanti
        A.Perspective(scale=(0.05, 0.1), p=0.3),                 # Prospettiva
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),            # Deformazioni elastiche per cavi curvi
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['class_labels']))


# CUSTOM MAPPER
# Serve per iniettare il nostro preprocessing e la nostra augmentation nella pipeline di Detectron2
class AlbumentationsMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, augmentations=None):
        super().__init__(cfg, is_train=is_train)
        self.augmentations = augmentations
    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        utils.check_image_size(dataset_dict, image)
        
        # Preprocessing sempre attivo
        if cfg.USE_PREPROCESSING:
            image = preprocess_image(image)
        
        # Augmentation solo in training
        if self.augmentations and self.is_train:
            annos = dataset_dict.get("annotations", [])
            bboxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
            category_ids = [obj["category_id"] for obj in annos]
            masks = [utils.polygons_to_bitmask(obj["segmentation"], image.shape[0], image.shape[1]) for obj in annos if "segmentation" in obj]
            
            # Gestione sicura nel caso manchino maschere
            if len(masks) != len(bboxes):
                masks = [np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) for _ in bboxes]
            
            try:
                transformed = self.augmentations(image=image, bboxes=bboxes, masks=masks, class_labels=category_ids)
                image = transformed["image"]
                bboxes = transformed["bboxes"]
                masks = transformed["masks"]
                category_ids = transformed["class_labels"]
            except Exception:
                pass  # Fallback: keep original if aug fails
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        if "annotations" in dataset_dict:
            image_shape = image.shape[:2]
            instances = Instances(image_shape)
            
            if self.augmentations and self.is_train:
                if len(bboxes) > 0:
                    instances.gt_boxes = Boxes(bboxes)
                    instances.gt_classes = torch.tensor(category_ids, dtype=torch.int64)
                    instances.gt_masks = BitMasks(torch.stack([torch.from_numpy(m) for m in masks]))
                else:
                    instances.gt_boxes = Boxes([])
                    instances.gt_classes = torch.tensor([], dtype=torch.int64)
                    instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            else:
                # Augmentation standard di Detectron2 se la nostra fallisce o è disabilitata
                instances = utils.annotations_to_instances(dataset_dict["annotations"], image_shape, mask_format="bitmask")
            
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        return dataset_dict

print("✓ Utility functions loaded")

# ============================================================================
# SECTION 2: DATASET REGISTRATION
# ============================================================================

if not NEEDS_INSTALL:
    print("\n📂 Loading datasets...")
    
    for split in ["train", "test"]:
        dataset_name = f"cable_{split}"
        json_path = cfg.TRAIN_JSON if split == "train" else cfg.TEST_JSON
        img_dir = cfg.TRAIN_IMAGES if split == "train" else cfg.TEST_IMAGES
        
        # Rimuovi se esiste già (per evitare errori se rilanci lo script)
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)
        
        DatasetCatalog.register(dataset_name, lambda jp=json_path, id=img_dir: get_detectron2_dicts(jp, id))
        MetadataCatalog.get(dataset_name).set(thing_classes=["cable"])
    
    dataset_dicts = get_detectron2_dicts(cfg.TRAIN_JSON, cfg.TRAIN_IMAGES)
    print(f"✓ Loaded {len(dataset_dicts)} training samples")
    
    if cfg.MAX_TRAIN_SAMPLES:
        dataset_dicts = dataset_dicts[:cfg.MAX_TRAIN_SAMPLES]
        print(f"⚠️ Quick test: using {len(dataset_dicts)} samples")

# ============================================================================
# SECTION 3: TRAINING CONFIGURATION
# ============================================================================

if not NEEDS_INSTALL and not cfg.SKIP_TRAINING:
    print("\n⚙️ Setting up training...")
    
    def setup_cfg_training():
        cfg_d2 = get_cfg()
        cfg_d2.merge_from_file(model_zoo.get_config_file(cfg.MODEL_NAME))
        cfg_d2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.MODEL_NAME)
        
        if cfg.RESUME_FROM and os.path.exists(cfg.RESUME_FROM):
            cfg_d2.MODEL.WEIGHTS = cfg.RESUME_FROM
            print(f"✓ Resuming from: {cfg.RESUME_FROM}")
        
        cfg_d2.DATASETS.TRAIN = ("cable_train",)
        cfg_d2.DATASETS.TEST = ("cable_test",)
        cfg_d2.DATALOADER.NUM_WORKERS = cfg.NUM_WORKERS
        
        cfg_d2.SOLVER.IMS_PER_BATCH = cfg.BATCH_SIZE
        
        # SCALA IMMAGINI: Training multi-scala + Risoluzione alta
        cfg_d2.INPUT.MIN_SIZE_TRAIN = (640, 700, 800, )
        cfg_d2.INPUT.MAX_SIZE_TRAIN = 1024
        cfg_d2.INPUT.MIN_SIZE_TEST = 700
        cfg_d2.INPUT.MAX_SIZE_TEST = 1024
        
        cfg_d2.SOLVER.BASE_LR = cfg.LEARNING_RATE
        max_iter = int((len(dataset_dicts) / cfg.BATCH_SIZE) * cfg.EPOCHS)
        cfg_d2.SOLVER.MAX_ITER = max_iter
        
        # STEPPED LEARNING RATE DECAY
        cfg_d2.SOLVER.STEPS = [int(max_iter * 0.6), int(max_iter * 0.8)]
        cfg_d2.SOLVER.GAMMA = 0.1
        
        # Warmup
        cfg_d2.SOLVER.WARMUP_ITERS = 1000
        cfg_d2.SOLVER.WARMUP_METHOD = "linear"
        cfg_d2.SOLVER.WARMUP_FACTOR = 0.001 / cfg.LEARNING_RATE 
        
        cfg_d2.SOLVER.CHECKPOINT_PERIOD = int((len(dataset_dicts) / cfg.BATCH_SIZE) * cfg.CHECKPOINT_PERIOD)
        cfg_d2.MODEL.ROI_HEADS.NUM_CLASSES = cfg.NUM_CLASSES
        cfg_d2.OUTPUT_DIR = cfg.OUTPUT_DIR
        
        if cfg.USE_AMP:
            # Parametri RPN spinti per catturare oggetti difficili
            cfg_d2.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
            cfg_d2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
            cfg_d2.MODEL.RPN.POSITIVE_FRACTION = 0.5
            cfg_d2.SOLVER.AMP.ENABLED = True
        
        # Setup Mapper
        if cfg.USE_AUGMENTATION:
            # Albumentations
            mapper = DatasetMapper(cfg_d2, is_train=True) # Placeholder, overwritten in Trainer
        else:
             # Default
            mapper = DatasetMapper(cfg_d2, is_train=True)
            
        return cfg_d2
    
    detectron_cfg = setup_cfg_training()
    print(f"✓ Config ready: {detectron_cfg.SOLVER.MAX_ITER} iterations, LR decay at {detectron_cfg.SOLVER.STEPS}")


# ============================================================================
# SECTION 4: TRAINING EXECUTION
# ============================================================================

if not NEEDS_INSTALL and not cfg.SKIP_TRAINING:
    print(f"\n🚀 Starting training ({cfg.EPOCHS} epochs)...")
    time.sleep(2)
    
    class CableTrainer(DefaultTrainer):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.start_time = time.time()
            self.epoch = 0
            
        # Sovrascriviamo build_train_loader per usare il nostro mapper custom con Albumentations
        @classmethod
        def build_train_loader(cls, cfg):
            if Config.USE_AUGMENTATION:
                aug = get_cable_augmentation()
                return build_detection_train_loader(cfg, mapper=AlbumentationsMapper(cfg, is_train=True, augmentations=aug))
            return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True))
        
        def run_step(self):
            super().run_step()
            
            # Logging manuale "per epoca"
            iters_per_epoch = len(dataset_dicts) // detectron_cfg.SOLVER.IMS_PER_BATCH
            current_epoch = self.iter // iters_per_epoch
            
            if current_epoch != self.epoch:
                self.epoch = current_epoch
                elapsed = (time.time() - self.start_time) / 3600
                print(f"Epoch {current_epoch}/{cfg.EPOCHS} | {elapsed:.2f}h")
                
                # Salvataggio manuale periodico
                if (current_epoch % cfg.CHECKPOINT_PERIOD == 0) and current_epoch > 0:
                    checkpoint_path = f"{cfg.CHECKPOINT_DIR}/checkpoint_epoch_{current_epoch}.pth"
                    torch.save({
                        'iteration': self.iter,
                        'epoch': current_epoch,
                        'model': self.model.state_dict(),
                    }, checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    try:
        trainer = CableTrainer(detectron_cfg)
        trainer.resume_or_load(resume=bool(cfg.RESUME_FROM))
        trainer.train()
        
        training_time = (time.time() - trainer.start_time) / 3600
        print(f"\n✅ Training complete ({training_time:.2f}h)")
        print(f"Model: {cfg.OUTPUT_DIR}/model_final.pth")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

# ============================================================================
# SECTION 5: INFERENCE
# ============================================================================

if not NEEDS_INSTALL and os.path.exists(f"{cfg.OUTPUT_DIR}/model_final.pth"):
    print("\n🔮 Running inference...")
    
    def setup_predictor():
        cfg_pred = get_cfg()
        cfg_pred.merge_from_file(model_zoo.get_config_file(cfg.MODEL_NAME))
        cfg_pred.MODEL.WEIGHTS = f"{cfg.OUTPUT_DIR}/model_final.pth"
        
        # Test Resolution Increase (TTA-like)
        cfg_pred.INPUT.MIN_SIZE_TEST = 1024
        cfg_pred.INPUT.MAX_SIZE_TEST = 2048
        
        cfg_pred.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.SCORE_THRESHOLD
        cfg_pred.MODEL.ROI_HEADS.NUM_CLASSES = cfg.NUM_CLASSES
        return DefaultPredictor(cfg_pred)
    
    predictor = setup_predictor()
    fit_line = fit_line_ransac if cfg.LINE_FITTING == "ransac" else fit_line_leastsquares
    
    with open(cfg.TEST_JSON, 'r') as f:
        test_data = json.load(f)
    
    results = []
    result_id = 0
    
    print(f"Processing {len(test_data['images'])} images...")
    
    for img_info in tqdm(test_data['images']):
        img_path = Path(cfg.TEST_IMAGES) / img_info['file_name']
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        
        # Preprocessing anche in test!
        if cfg.USE_PREPROCESSING:
            img = preprocess_image(img)
        
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        for i in range(len(instances)):
            mask = instances.pred_masks[i].numpy()
            
            # Post-Filter: Area
            if mask.sum() < cfg.MIN_MASK_AREA: continue
            
            score = instances.scores[i].item()
            bbox = instances.pred_boxes[i].tensor.numpy()[0]
            bbox_coco = [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
            
            mask_rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
            
            rho, theta = fit_line(mask)
            
            results.append({
                "image_id": img_info['id'],
                "category_id": 0,
                "bbox": bbox_coco,
                "segmentation": mask_rle,
                "score": float(score),
                "lines": [float(rho), float(theta)],
                "area": float(mask.sum()),
                "height": img_info['height'],
                "width": img_info['width'],
                "id": result_id
            })
            result_id += 1
    
    output_file = f"{cfg.MATRICOLA}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Predictions saved: {output_file}")
    print(f"Total detections: {len(results)}")
    
    # Stats
    scores = [p['score'] for p in results]
    print(f"Mean confidence: {np.mean(scores):.3f}")

print("\n✅ Done! Remember to download the JSON file.")
