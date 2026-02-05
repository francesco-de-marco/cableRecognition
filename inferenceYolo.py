# ============================================================================
# SECTION 0: IMPORTS
# ============================================================================

import os, json, cv2, torch, warnings
import numpy as np
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils
from ultralytics import YOLO
from skimage.morphology import skeletonize, medial_axis
from sklearn.linear_model import RANSACRegressor

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

MATRICOLA = "263780"
# Path del modello addestrato (weights/best.pt)
MODEL_PATH = r"C:\Users\matti\Desktop\ProgCV\runs_output\yolo_train\weights\best.pt"
TEST_JSON = r"C:\Users\matti\Desktop\ProgCV\data\test\test.json"
TEST_IMAGES = r"C:\Users\matti\Desktop\ProgCV\data\test"

# --- PARAMETRI CHIAVE PER L'INFERENZA (Tuning per LDS) ---
# CONF_THRESHOLD: Soglia minima perché un oggetto sia considerato valido. 
# 0.05 è molto basso ("Aggressive Recovery"): accettiamo tutto all'inizio per non perdere cavi deboli.
CONF_THRESHOLD = 0.05    

# BIN_THRESHOLD: Soglia per binarizzare la maschera di segmentazione (soft mask -> binaria).
# 0.35 (invece di 0.5) rende le maschere più "cicciotte", aumentando l'IoU sui bordi incerti.
BIN_THRESHOLD = 0.35     

# MIN_AREA: Scartiamo maschere più piccole di 60 pixel quadrati. 
# Serve a eliminare piccoli rumori/macchie che il modello potrebbe scambiare per cavi.
MIN_AREA = 60            

# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

def apply_clahe(img):
    """
    Applica CLAHE (Contrast Limited Adaptive Histogram Equalization) al canale L (Luminosità).
    Aumenta il contrasto locale per far risaltare i cavi scuri senza bruciare l'immagine.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def get_ransac_line(mask_binary):
    """
    Calcola la retta (rho, theta) che meglio approssima la maschera segmentata.
    Utilizza RANSAC per robustezza contro gli outlier (pixel spuri della maschera).
    """
    try:
        # 1. Mask Smoothing: Pulisce i bordi frastagliati della maschera
        mask_smoothed = cv2.GaussianBlur(mask_binary.astype(np.float32), (3, 3), 0)
        mask_binary = (mask_smoothed > 0.5).astype(np.uint8)

        # 2. Skeletonization (Medial Axis): Riduce la maschera a una linea di 1 pixel di spessore.
        # È fondamentale per trovare la "spina dorsale" del cavo per la regressione.
        skel, distance = medial_axis(mask_binary > 0, return_distance=True)
        y, x = np.where(skel > 0)
        
        # Fallback: se lo scheletro è troppo corto (rumore), usa tutti i punti della maschera
        if len(x) < 10: 
            y, x = np.where(mask_binary > 0)
        
        X = x.reshape(-1, 1)
        
        # 3. RANSAC Regressor: Trova la retta ignorando i punti outlier
        # residual_threshold=1.5: tolleranza di 1.5 pixel dalla retta
        ransac = RANSACRegressor(residual_threshold=1.5, max_trials=500)
        ransac.fit(X, y)
        
        m = ransac.estimator_.coef_[0]   # Coefficiente angolare (pendenza)
        q = ransac.estimator_.intercept_ # Intercetta
        
        # 4. Conversione a coordinate polari standard (Rho, Theta)
        theta = np.arctan2(-1, m) % np.pi
        rho = np.abs(q) / np.sqrt(1 + m**2)
        
        return float(rho), float(theta)
    except:
        return 0.0, 0.0

def apply_nms_masks(results, iou_threshold=0.5):
    """
    Non-Maximum Suppression (NMS) calcolata sulle MASCHERE (IoU di segmentazione).
    YOLO fa già NMS sui Box, ma poiché usiamo una confidenza bassissima (0.05),
    potremmo avere più predizioni sovrapposte per lo stesso cavo.
    Questa funzione rimuove i duplicati basandosi sulla sovrapposizione delle maschere.
    """
    if not results: return []
    # Ordina per score decrescente (tieni il migliore per primo)
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    keep = []
    used_indices = set()

    for i, res_i in enumerate(results):
        if i in used_indices: continue
        keep.append(res_i)
        
        m_i = mask_utils.decode(res_i['segmentation'])
        for j in range(i + 1, len(results)):
            if j in used_indices: continue
            
            # Calcola Intersection over Union (IoU) tra le maschere
            m_j = mask_utils.decode(results[j]['segmentation'])
            intersection = np.logical_and(m_i, m_j).sum()
            union = np.logical_or(m_i, m_j).sum()
            iou = intersection / union if union > 0 else 0
            
            # Se si sovrappongono troppo (> 0.5), scarta quella con score più basso
            if iou > iou_threshold:
                used_indices.add(j)
    return keep

# ============================================================================
# SECTION 4: INFERENCE LOOP
# ============================================================================

if __name__ == "__main__":
    print(f"🚀 Avvio Inferenza Strategica (Road to 2.50) per matricola {MATRICOLA}...")
    model = YOLO(MODEL_PATH)
    
    with open(TEST_JSON, 'r') as f:
        test_data = json.load(f)
    
    final_results = []

    for img_info in tqdm(test_data['images'], desc="Processing Images"):
        img_p = os.path.join(TEST_IMAGES, img_info['file_name'])
        if not os.path.exists(img_p): continue
        
        # 1. Caricamento e Enhancing
        img_raw = cv2.imread(img_p)
        img_enh = apply_clahe(img_raw)
        
        # 2. Inferenza YOLO
        # retina_masks=True: genera maschere ad alta risoluzione (migliore qualità)
        res = model(img_enh, conf=CONF_THRESHOLD, imgsz=1024, retina_masks=True, verbose=False)
        
        if not res[0].masks: continue
        
        img_predictions = []
        # Itera su maschere, box e confidenze trovate
        for m, b, s in zip(res[0].masks.data.cpu().numpy(), 
                           res[0].boxes.xyxy.cpu().numpy(), 
                           res[0].boxes.conf.cpu().numpy()):
            
            # 3. Post-Processing Maschera
            # Resize alla dimensione originale dell'immagine (es. 700x700 o variabile)
            # Nota: qui assume 700x700 fisso, idealmente dovrebbe usare h,w da img_info
            m_res = cv2.resize(m, (700, 700), interpolation=cv2.INTER_LINEAR)
            
            # Binarizzazione con soglia custom (0.35)
            m_bin = (m_res > BIN_THRESHOLD).astype(np.uint8)
            
            # Filtro area minima
            if m_bin.sum() < MIN_AREA: continue
            
            # Encoding RLE (Run-Length Encoding) per salvare spazio nel JSON
            rle = mask_utils.encode(np.asfortranarray(m_bin))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Calcolo Linea
            rho, theta = get_ransac_line(m_bin)
            
            img_predictions.append({
                "image_id": img_info['id'], "category_id": 0, "score": float(s),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "segmentation": rle, "lines": [rho, theta], "area": float(m_bin.sum())
            })
        
        # 4. Pulizia duplicati con NMS sulle maschere
        clean_predictions = apply_nms_masks(img_predictions, iou_threshold=0.5)
        final_results.extend(clean_predictions)
            
    with open(f"{MATRICOLA}.json", 'w') as f:
        json.dump(final_results, f)
    
    print(f"\n✨ Task completato!")
    print(f"📊 Istanze totali nel JSON: {len(final_results)}")
    print(f"📍 File generato: {MATRICOLA}.json")