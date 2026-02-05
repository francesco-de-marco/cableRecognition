
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')
import sys
import subprocess
import shutil

# ============================================================================
# INSTALLAZIONE DIPENDENZE (KAGGLE / COLAB)
# ============================================================================
# Questo blocco serve per assicurarsi che le librerie necessarie siano installate
# anche in ambienti cloud volatili come Kaggle o Colab.
print("Checking dependencies...")
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    print("✓ Dependencies OK")
    NEEDS_INSTALL = False
except ImportError:
    NEEDS_INSTALL = True

if NEEDS_INSTALL:
    # Installa ultralytics (YOLOv8) e pycocotools per gestire le maschere
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "numpy<2", "ultralytics", "pycocotools"])
    print("✅ Installed. RESTART KERNEL!")
    sys.exit(0)

# ============================================================================
# IMPORTS
# ============================================================================

import json
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm.auto import tqdm
from ultralytics import YOLO

print("✓ Imports OK")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Device: {device}")

# ============================================================================
# CONFIGURAZIONE GENERALE (Semi-Supervised Learning)
# ============================================================================

class Config:
    # PERCORSI (Adattati per ambiente Kaggle)
    DATA_ROOT = "/kaggle/input/cableimages/progetto_cv_2025_2026"
    TRAIN_JSON = f"{DATA_ROOT}/train/train.json"
    TRAIN_IMAGES = f"{DATA_ROOT}/train"
    TEST_JSON = f"{DATA_ROOT}/test/test.json"
    TEST_IMAGES = f"{DATA_ROOT}/test"
    
    # MODELLO DI PARTENZA (Teacher Model)
    # È fondamentale usare il PROPRIO modello migliore ottenuto finora (es. quello da 2.35 LDS).
    # Questo modello agirà da "insegnante" per generare le etichette per il set di test.
    MODEL_PATH = "/kaggle/input/yolo-weights/best.pt"  
    
    # CARTELLE DI OUTPUT
    OUTPUT_DIR = "/kaggle/working"
    PSEUDO_DATA_DIR = "/kaggle/working/pseudo_dataset" # Dove creeremo il nuovo dataset espanso
    
    # PARAMETRI PSEUDO-LABELING
    # PSEUDO_CONF_THRESHOLD: Soglia di fiducia molto alta (0.5).
    # Accettiamo come "verità" solo le predizioni di cui il modello è molto sicuro.
    # Se usassimo una soglia bassa (es. 0.05), introdurremmo rumore nel training set.
    PSEUDO_CONF_THRESHOLD = 0.5   
    MIN_MASK_AREA = 100           # Ignoriamo maschere troppo piccole (probabile rumore) anche se confidenza alta
    
    # PARAMETRI DI RE-TRAINING (Student Model)
    RETRAIN_EPOCHS = 30           # Numero moderato di epoche per raffinare il modello
    BATCH_SIZE = 4
    IMAGE_SIZE = 1024

cfg = Config()

print(f"⚙️ Pseudo-label threshold: {cfg.PSEUDO_CONF_THRESHOLD}")

# ============================================================================
# STEP 1: CARICAMENTO DATI DI TRAINING ORIGINALI
# ============================================================================
# In questa fase copiamo i dati di training originali (manuali, affidabili al 100%)
# nella nuova cartella del dataset ibrido.

print("\n📂 Step 1: Loading original training data...")

# Creazione struttura cartelle dataset YOLO
pseudo_train_imgs = Path(cfg.PSEUDO_DATA_DIR) / "images" / "train"
pseudo_train_labels = Path(cfg.PSEUDO_DATA_DIR) / "labels" / "train"
pseudo_val_imgs = Path(cfg.PSEUDO_DATA_DIR) / "images" / "val"
pseudo_val_labels = Path(cfg.PSEUDO_DATA_DIR) / "labels" / "val"

for p in [pseudo_train_imgs, pseudo_train_labels, pseudo_val_imgs, pseudo_val_labels]:
    p.mkdir(parents=True, exist_ok=True)

with open(cfg.TRAIN_JSON, 'r') as f:
    train_data = json.load(f)

print("Converting original training data to YOLO format...")
for img_info in tqdm(train_data['images']):
    img_name = img_info['file_name']
    img_id = img_info['id']
    h, w = img_info['height'], img_info['width']
    
    # Copia immagine fisica
    src = Path(cfg.TRAIN_IMAGES) / img_name
    dst = pseudo_train_imgs / img_name
    if src.exists():
        shutil.copy2(src, dst)
    
    # Recupera annotazioni e converte in YOLO txt
    anns = [a for a in train_data['annotations'] if a['image_id'] == img_id]
    
    label_file = pseudo_train_labels / f"{Path(img_name).stem}.txt"
    with open(label_file, 'w') as f:
        for ann in anns:
            seg = ann['segmentation']
            if isinstance(seg, list) and len(seg) > 0:
                polygon = seg[0] if isinstance(seg[0], list) else seg
                if len(polygon) < 6: continue # Ignora poligoni degeneri
                
                # Normalizza coordinate rispetto a W e H
                norm_coords = []
                for i in range(0, len(polygon), 2):
                    if i+1 < len(polygon):
                        norm_coords.append(f"{polygon[i]/w:.6f}")
                        norm_coords.append(f"{polygon[i+1]/h:.6f}")
                
                if norm_coords:
                    f.write(f"0 {' '.join(norm_coords)}\n")

original_count = len(list(pseudo_train_imgs.glob("*")))
print(f"✓ Converted {original_count} original training images")

# ============================================================================
# STEP 2: GENERAZIONE PSEUDO-LABELS DAL TEST SET
# ============================================================================
# Il cuore del Self-Training:
# 1. Prendiamo le immagini NON etichettate (Test Set)
# 2. Le passiamo al nostro modello attuale (Teacher)
# 3. Salviamo le predizioni AD ALTA CONFIDENZA come se fossero etichette vere
# 4. Aggiungiamo queste immagini + etichette al training set.

print("\n🔮 Step 2: Generating pseudo-labels from test images...")

if not os.path.exists(cfg.MODEL_PATH):
    print(f"❌ Model not found: {cfg.MODEL_PATH}")
    print("Upload your best.pt to Kaggle and update cfg.MODEL_PATH")
    sys.exit(1)

model = YOLO(cfg.MODEL_PATH)

with open(cfg.TEST_JSON, 'r') as f:
    test_data = json.load(f)

pseudo_count = 0
skipped_low_conf = 0
skipped_small = 0

for img_info in tqdm(test_data['images'], desc="Generating pseudo-labels"):
    img_path = Path(cfg.TEST_IMAGES) / img_info['file_name']
    if not img_path.exists(): continue
    
    img = cv2.imread(str(img_path))
    h, w = img_info['height'], img_info['width']
    
    # Inferenza sul test set
    # Usiamo conf=0.1 qui per estrarre candidati, filtreremo dopo con PSEUDO_CONF_THRESHOLD (0.5)
    results = model(img, imgsz=cfg.IMAGE_SIZE, conf=0.1, verbose=False, retina_masks=True)
    
    if results[0].masks is None: continue
    
    masks = results[0].masks.data.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    
    high_conf_masks = []
    for mask, conf in zip(masks, confs):
        # FILTRO DI QUALITÀ CRUCIALE
        if conf < cfg.PSEUDO_CONF_THRESHOLD: # Solo se > 0.5
            skipped_low_conf += 1
            continue
        
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        if mask_binary.sum() < cfg.MIN_MASK_AREA:
            skipped_small += 1
            continue
        
        high_conf_masks.append(mask_binary)
    
    if len(high_conf_masks) == 0: continue
    
    # Se l'immagine ha predizioni valide, la aggiungiamo al training set
    pseudo_count += 1
    dst_img = pseudo_train_imgs / img_info['file_name']
    shutil.copy2(img_path, dst_img)
    
    # Scrittura etichette generate (Pseudo-Ground Truth)
    label_file = pseudo_train_labels / f"{Path(img_info['file_name']).stem}.txt"
    with open(label_file, 'w') as f:
        for mask in high_conf_masks:
            # Estrazione contorni dalla maschera binaria per avere il poligono YOLO
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3: continue
                
                # Semplificazione poligono per ridurre il numero di punti
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3: continue
                
                # Normalizzazione
                points = approx.reshape(-1, 2)
                norm_coords = []
                for x, y in points:
                    norm_coords.append(f"{x/w:.6f}")
                    norm_coords.append(f"{y/h:.6f}")
                
                if len(norm_coords) >= 6:
                    f.write(f"0 {' '.join(norm_coords)}\n")

print(f"✓ Generated pseudo-labels for {pseudo_count} test images")
print(f"  Skipped (low conf): {skipped_low_conf}")
print(f"  Skipped (small area): {skipped_small}")

# ============================================================================
# STEP 3: CREAZIONE SPLIT VALIDATION
# ============================================================================
# Creiamo un validation set anche per i dati pseudo, per monitorare che il modello
# non stia imparando rumore.

print("\n📊 Step 3: Creating train/val split...")

import random
all_imgs = list(pseudo_train_imgs.glob("*"))
random.shuffle(all_imgs)

val_count = max(1, len(all_imgs) // 10)  # 10% validazione
val_imgs = all_imgs[:val_count]

for img in val_imgs:
    shutil.move(str(img), str(pseudo_val_imgs / img.name))
    label = pseudo_train_labels / f"{img.stem}.txt"
    if label.exists():
        shutil.move(str(label), str(pseudo_val_labels / f"{img.stem}.txt"))

final_train = len(list(pseudo_train_imgs.glob("*")))
final_val = len(list(pseudo_val_imgs.glob("*")))
print(f"✓ Final split: {final_train} train, {final_val} val")

# ============================================================================
# STEP 4: CREATE DATA.YAML
# ============================================================================

data_yaml = cfg.PSEUDO_DATA_DIR + "/data.yaml"
with open(data_yaml, 'w') as f:
    f.write(f"path: {cfg.PSEUDO_DATA_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("nc: 1\n")
    f.write("names: ['cable']\n")

print(f"✓ Created {data_yaml}")

# ============================================================================
# STEP 5: RE-TRAINING CON PSEUDO-LABELS
# ============================================================================
# Addestriamo un nuovo modello (Student) usando sia i dati veri che quelli generati.
# Questo permette al modello di imparare dai suoi stessi feedback ad alta confidenza
# su nuovi dati (il test set).

print(f"\n🚀 Step 5: Re-training with pseudo-labels ({cfg.RETRAIN_EPOCHS} epochs)...")

# Partiamo comunque dai pesi del modello precedente (Transfer Learning continuo)
model = YOLO(cfg.MODEL_PATH)

results = model.train(
    data=data_yaml,
    epochs=cfg.RETRAIN_EPOCHS,
    batch=cfg.BATCH_SIZE,
    imgsz=cfg.IMAGE_SIZE,
    project=cfg.OUTPUT_DIR,
    name='pseudo_train',
    exist_ok=True,
    patience=15,
    device=0 if device == "cuda" else "cpu",
    workers=2,
    verbose=True,
    # Augmentation: devono essere robuste per generalizzare bene
    degrees=45,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)

print("\n✅ Re-training complete!")
new_model_path = f"{cfg.OUTPUT_DIR}/pseudo_train/weights/best.pt"
print(f"New model: {new_model_path}")

# ============================================================================
# STEP 6: INFERENZA FINALE CON IL NUOVO MODELLO
# ============================================================================
# Ora usiamo il modello "Student" ri-addestrato per generare le predizioni finali
# da sottomettere. Qui torniamo alla strategia "LDS boost" aggressiva.

print("\n🔮 Step 6: Running final inference with new model...")

from pycocotools import mask as mask_utils
from skimage.morphology import medial_axis
from sklearn.linear_model import RANSACRegressor

# UTILITY LOCALI (uguali a bestYolo.py)
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def get_ransac_line(mask_binary):
    # Logica robusta per estrare la linea dalla maschera
    try:
        mask_smoothed = cv2.GaussianBlur(mask_binary.astype(np.float32), (3, 3), 0)
        mask_binary = (mask_smoothed > 0.5).astype(np.uint8)
        skel, distance = medial_axis(mask_binary > 0, return_distance=True)
        y, x = np.where(skel > 0)
        
        if len(x) < 5: return 0.0, 0.0 # Troppo poco per una linea
        
        X = x.reshape(-1, 1)
        ransac = RANSACRegressor(residual_threshold=1.5, max_trials=500)
        ransac.fit(X, y)
        m = ransac.estimator_.coef_[0]
        q = ransac.estimator_.intercept_
        theta = np.arctan2(-1, m) % np.pi
        rho = np.abs(q) / np.sqrt(1 + m**2)
        return float(rho), float(theta)
    except:
        return 0.0, 0.0

def apply_nms_masks(results, iou_threshold=0.5):
    # NMS post-process sulle maschere
    if not results: return []
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    keep = []
    used_indices = set()
    for i, res_i in enumerate(results):
        if i in used_indices: continue
        keep.append(res_i)
        m_i = mask_utils.decode(res_i['segmentation'])
        for j in range(i + 1, len(results)):
            if j in used_indices: continue
            m_j = mask_utils.decode(results[j]['segmentation'])
            intersection = np.logical_and(m_i, m_j).sum()
            union = np.logical_or(m_i, m_j).sum()
            iou = intersection / union if union > 0 else 0
            if iou > iou_threshold: used_indices.add(j)
    return keep

# PARAMETRI INFERENZA FINALE (Aggressivi)
CONF_THRESHOLD = 0.05   # Recupero massimo
BIN_THRESHOLD = 0.35    # Maschere spesse
MIN_AREA = 60           

final_model = YOLO(new_model_path)

with open(cfg.TEST_JSON, 'r') as f:
    test_data = json.load(f)

all_results = []

for img_info in tqdm(test_data['images'], desc="Final Inference"):
    img_path = Path(cfg.TEST_IMAGES) / img_info['file_name']
    if not img_path.exists(): continue
    
    img_raw = cv2.imread(str(img_path))
    img_enh = apply_clahe(img_raw)
    H, W = img_info['height'], img_info['width']
    
    res = final_model(img_enh, conf=CONF_THRESHOLD, imgsz=cfg.IMAGE_SIZE, retina_masks=True, verbose=False)
    
    if res[0].masks is None: continue
    
    img_predictions = []
    for m, b, s in zip(res[0].masks.data.cpu().numpy(), 
                       res[0].boxes.xyxy.cpu().numpy(), 
                       res[0].boxes.conf.cpu().numpy()):
        
        m_res = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        m_bin = (m_res > BIN_THRESHOLD).astype(np.uint8)
        
        area = m_bin.sum()
        if area < MIN_AREA: continue
        
        rle = mask_utils.encode(np.asfortranarray(m_bin))
        rle['counts'] = rle['counts'].decode('utf-8')
        rho, theta = get_ransac_line(m_bin)
        
        img_predictions.append({
            "image_id": img_info['id'], "category_id": 0, "score": float(s),
            "bbox": [float(b), for b in b], # (semplificato per leggibilità codice)
            "segmentation": rle, "lines": [rho, theta], "area": float(area),
            "height": H, "width": W
        })
        # Nota: La riga bbox sopra nel commento è simbolica, nel codice reale va estratta corretta
        # Correggo qui sotto per essere eseguibile
        img_predictions[-1]["bbox"] = [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])]

    clean_predictions = apply_nms_masks(img_predictions, iou_threshold=0.5)
    for pred in clean_predictions:
        pred["id"] = len(all_results)
        all_results.append(pred)

output_file = "263780.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Saved {len(all_results)} predictions to {output_file}")
print("🎯 DONE! Download 263780.json and submit!")
