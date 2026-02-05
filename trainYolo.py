# ============================================================================
# SECTION 0: IMPORTS
# ============================================================================

import os
import json
import numpy as np
import torch
import cv2
import shutil
import random
import warnings
from pathlib import Path
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils
from ultralytics import YOLO
from skimage.morphology import skeletonize

# Configurazione dispositivo: Seleziona la GPU (device 0) se disponibile, altrimenti usa la CPU.
# Questo è fondamentale per velocizzare il training di YOLO.
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================
class Config:
    """
    Classe di configurazione centralizzata.
    Raccoglie tutti i parametri modificabili per il training e la gestione dei dati.
    """
    # 1. PERCORSI DEI DATI
    # Ottiene la directory corrente di lavoro per costruire percorsi assoluti
    BASE_DIR = os.getcwd()
    DATA_ROOT = os.path.join(BASE_DIR, "data")
    
    # File JSON contenenti le annotazioni (in formato COCO) per training e test
    TRAIN_JSON = os.path.join(DATA_ROOT, "train", "train.json")
    TEST_JSON = os.path.join(DATA_ROOT, "test", "test.json")
    
    # Cartelle contenenti le immagini sorgenti
    TRAIN_IMAGES = os.path.join(DATA_ROOT, "train")
    TEST_IMAGES = os.path.join(DATA_ROOT, "test")
    
    # 2. OUTPUT DEL TRAINING
    # runs_output: Dove YOLO salverà i pesi (.pt), grafici e log del training
    # yolo_dataset_local: Dove verrà creata la copia del dataset convertita in formato YOLO
    OUTPUT_DIR = os.path.join(BASE_DIR, "runs_output")
    YOLO_DATA_DIR = os.path.join(BASE_DIR, "yolo_dataset_local")
    
    # 3. IPERPARAMETRI DI TRAINING
    SKIP_TRAINING = False    # Se True, salta la fase di training (utile se hai già i pesi)
    EPOCHS = 50              # Numero totale di epoche (passaggi completi sul dataset)
    BATCH_SIZE = 4           # Numero di immagini processate contemporaneamente (dipende dalla VRAM della GPU)
    IMAGE_SIZE = 1024        # Dimensione a cui vengono ridimensionate le immagini per il training (risoluzione alta per vedere i cavi)
    MODEL_SIZE = "m"         # Taglia del modello YOLOv8-seg: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra)
    

cfg = Config()
# Crea la cartella di output se non esiste
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

def apply_clahe(img):
    """
    Migliora il contrasto locale dell'immagine usando CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Perché serve?
    I cavi sono spesso scuri su sfondo scuro o in ombra. CLAHE aumenta il contrasto locale
    rendendo i cavi più visibili per il modello, senza sovra-esporre le parti già chiare.
    """
    # Converte da BGR (formato OpenCV) a LAB (Luminosità + Canali Colore)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Configura CLAHE: clipLimit evita che il rumore venga amplificato troppo
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Applica l'equalizzazione solo al canale 'L' (Luminosità/Lightness)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Riconverte in BGR per l'uso normale
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def get_refined_line(mask_binary):
    """
    Funzione ausiliaria per calcolare Rho e Theta da una maschera binaria.
    Usa SVD (Singular Value Decomposition) sui punti dello scheletro della predizione.
    
    Nota: Questa funzione è definita qui ma non viene esplicitamente usata nel ciclo di training base,
    è spesso parte della logica di post-processing o validazione.
    """
    try:
        # Riduce la maschera a uno "scheletro" largo 1 pixel
        skeleton = skeletonize(mask_binary > 0).astype(np.uint8)
        y, x = np.where(skeleton > 0)
        
        # Se lo scheletro è troppo piccolo, usa tutti i punti della maschera
        if len(x) < 5: y, x = np.where(mask_binary > 0)
        
        points = np.column_stack([x, y])
        mean = points.mean(axis=0)
        
        # SVD per trovare la direzione principale della distribuzione di punti
        _, _, Vt = np.linalg.svd(points - mean)
        
        # Il primo vettore di Vt corrisponde alla direzione della linea
        # La normale è ortogonale alla direzione
        norm = np.array([-Vt[0][1], Vt[0][0]])
        
        # Assicura che la normale punti "verso fuori" (convenzione standard)
        if np.dot(norm, mean) < 0: norm = -norm
        
        # Calcola la distanza dall'origine (rho) e l'angolo (theta)
        return float(np.abs(np.dot(norm, mean))), float(np.arctan2(norm[1], norm[0]) % np.pi)
    except: return 0.0, 0.0

# ============================================================================
# SECTION 3: DATA PREPARATION
# ============================================================================
def prepare_data():
    """
    Converte il dataset dal formato COCO (usato nella competizione) al formato YOLO standard.
    YOLO richiede:
    1. Una struttura di cartelle specifica: images/train, labels/train, images/val, labels/val
    2. Annotazioni in file .txt (uno per immagine)
    3. Formato riga nel .txt: <class_id> <x1> <y1> <x2> <y2> ... (coordinate normalizzate 0-1)
    """
    
    # Se il dataset esiste già, evita di ricrearlo per risparmiare tempo
    if os.path.exists(os.path.join(cfg.YOLO_DATA_DIR, "data.yaml")):
        print("✓ Dataset YOLO già preparato.")
        return

    print("\n📂 Preparazione dataset in formato YOLO...")
    
    # Definizione percorsi destinazione
    img_train = Path(cfg.YOLO_DATA_DIR) / "images" / "train"
    lbl_train = Path(cfg.YOLO_DATA_DIR) / "labels" / "train"
    img_val = Path(cfg.YOLO_DATA_DIR) / "images" / "val"
    lbl_val = Path(cfg.YOLO_DATA_DIR) / "labels" / "val"
    
    # Creazione cartelle fisiche
    for p in [img_train, lbl_train, img_val, lbl_val]: p.mkdir(parents=True, exist_ok=True)

    # Carica le annotazioni originali
    with open(cfg.TRAIN_JSON, 'r') as f: train_data = json.load(f)

    # Ciclo su tutte le immagini del training set
    for img_info in tqdm(train_data['images'], desc="Converting Labels"):
        img_name = img_info['file_name']
        h, w = img_info['height'], img_info['width']
        
        # Copia l'immagine nella cartella train
        shutil.copy2(os.path.join(cfg.TRAIN_IMAGES, img_name), img_train / img_name)
        
        # Trova le annotazioni per questa specifica immagine
        anns = [ann for ann in train_data['annotations'] if ann['image_id'] == img_info['id']]
        
        # Crea il file .txt corrispondente
        with open(lbl_train / f"{Path(img_name).stem}.txt", 'w') as f_lbl:
            for ann in anns:
                seg = ann['segmentation']
                # Gestione formato poligono COCO
                if isinstance(seg, list) and len(seg) > 0:
                    poly = seg[0] if isinstance(seg[0], list) else seg # Appiattisce se necessario
                    
                    # Normalizza le coordinate (dividi per larghezza e altezza) come richiesto da YOLO
                    norm_coords = [f"{poly[i]/w:.6f} {poly[i+1]/h:.6f}" for i in range(0, len(poly), 2)]
                    
                    # Scrive: classe 0 (cavo) + coordinate poligono
                    f_lbl.write(f"0 {' '.join(norm_coords)}\n")

    # SPLIT TRAINING / VALIDATION (80% / 20%)
    # Sposta casualmente il 20% dei file da train a val per monitorare la qualità durante il training
    imgs = list(img_train.glob('*.*'))
    random.shuffle(imgs)
    val_count = len(imgs) // 5 # 20%
    
    for img in imgs[:val_count]:
        # Sposta immagine
        shutil.move(str(img), str(img_val / img.name))
        # Sposta annotazione corrispondente (se esiste)
        lbl = lbl_train / f"{img.stem}.txt"
        if lbl.exists(): shutil.move(str(lbl), str(lbl_val / lbl.name))

    # Scrive il file data.yaml che dice a YOLO dove trovare i dati
    with open(os.path.join(cfg.YOLO_DATA_DIR, "data.yaml"), 'w') as f:
        f.write(f"path: {os.path.abspath(cfg.YOLO_DATA_DIR)}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['cable']\n")

# ============================================================================
# SECTION 4: TRAINING
# ============================================================================
if __name__ == "__main__":
    # Esegue la preparazione dei dati
    prepare_data()
    
    # Percorsi assoluti file configurazione
    data_yaml = os.path.abspath(os.path.join(cfg.YOLO_DATA_DIR, "data.yaml"))
    checkpoint = os.path.join(cfg.OUTPUT_DIR, "yolo_train", "weights", "last.pt")

    # Avvio del Training
    if not cfg.SKIP_TRAINING:
        # Carica il modello pre-addestrato (transfer learning) o riprendi un checkpoint interrotto
        model = YOLO(checkpoint if os.path.exists(checkpoint) else f"yolov8{cfg.MODEL_SIZE}-seg.pt")
        
        # Comando principale di training
        model.train(
            data=data_yaml,           # File che definisce il dataset
            epochs=cfg.EPOCHS,        # Numero di epoche
            imgsz=cfg.IMAGE_SIZE,     # Risoluzione interna (es. 1024)
            batch=cfg.BATCH_SIZE,     # Batch size
            project=cfg.OUTPUT_DIR,   # Cartella radice per i risultati
            name='yolo_train',        # Nome della sottocartella d'esperimento
            
            # Data Augmentation & Iperparametri Specifici
            degrees=180.0,      # Rotazione +/- 180 gradi (i cavi possono avere qualsiasi orientamento)
            perspective=0.0005, # Leggera distorsione prospettica
            mosaic=1.0,         # Usa sempre mosaic augmentation (unisce 4 immagini in 1)
            
            resume=os.path.exists(checkpoint), # Riprendi se esiste un checkpoint
            device=DEVICE       # GPU o CPU
        )