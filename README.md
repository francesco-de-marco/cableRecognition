# Cable Detection & Segmentation

Questo repository contiene il codice sorgente per il progetto di **Computer Vision** "Cable Detection & Segmentation", sviluppato per identificare e segmentare cavi elettrici in scenari complessi (urbani ed extra-urbani).

Il progetto si focalizza sulla massimizzazione della metrica **LDS (Line Detection Score)**, combinando architetture di Deep Learning all'avanguardia con tecniche avanzate di post-processing geometrico.

## 📄 Descrizione del Progetto

La sfida principale consiste nel rilevare oggetti estremamente sottili e filiformi (cavi) spesso confusi con il background o occlusi. Per affrontare il problema, sono state esplorate due strade principali:
1.  **Detectron2 (Mask R-CNN):** Approccio Two-Stage solido e preciso.
2.  **Ultralytics YOLOv8-seg:** Approccio One-Stage Real-Time che, combinato con inferenza aggressiva, ha prodotto i risultati migliori.

### Risultati Ottenuti

| Modello | Metodo | LDS Score |
|---|---|---|
| **YOLO-Pseudo v2** | **Self-Training (Iterative)** | **2.450** 🏆 |
| YOLOv8m-seg | RANSAC + CLAHE | 2.360 |
| Mask R-CNN | ResNeXt-101 Backbone | 2.127 |

## 📂 Contenuto del Repository

I file presenti nella root del repository implementano le diverse fasi della pipeline:

-   **`trainYolo.py`**:
    Script per l'addestramento del modello **YOLOv8-seg**. Gestisce la preparazione del dataset, l'augmentations e il training loop con configurazioni ottimizzate per segmentare linee sottili.

-   **`inferenceYolo.py`**:
    Script di inferenza avanzata ("Road to 2.50"). Implementa la pipeline di **High-Recall Inference**:
    -   **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) sul canale di luminanza.
    -   **Inference**: Soglia di confidenza bassissima (0.05) per catturare ogni cavo.
    -   **Post-Processing**: Skeletonization + **RANSAC Regressor** per fittare linee precise sulle maschere predette.
    -   **NMS Custom**: Rimozione duplicati basata su IoU delle maschere.

-   **`progPseudoLabeling.py`**:
    Implementazione del **Self-Training (Semi-Supervised Learning)**.
    -   Genera "Pseudo-Labels" sul Test Set usando un modello "Teacher".
    -   Riestende il dataset di training includendo le nuove etichette.
    -   Riadestra un modello "Student" per migliorare le performance iterativamente.

-   **`progMaskCVBest.py`**:
    Script completo per l'approccio **Detectron2**. Include:
    -   Configurazione di Mask R-CNN (ResNeXt-101).
    -   Data Augmentation personalizzata con *Albumentations*.
    -   Training loop custom e inferenza con post-processing RANSAC.

## 🚀 Tecnologie Utilizzate

-   **Python 3.x**
-   **PyTorch**
-   **Ultralytics YOLOv8**
-   **Detectron2 (FAIR)**
-   **OpenCV** & **Albumentations**

## 👥 Autori
-   **Francesco De Marco**
-   **Mattia D'agostino**
