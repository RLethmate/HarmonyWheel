# Live Chord Detector – State (kurz)

## Ziel
Live Akkorderkennung (Pop/Jazz) via Browser-Mikrofon → WebSocket → FastAPI.
Erkennt Triads (maj/min/dim) und optional dom7 (z.B. G7). Im Jazz Modus 4 Klänge.
Key-Detection zeigt Top-2 Kandidaten. 
Es gibt 2 Detektionsmodelle zum Testen: 1. DSP und 2. crema (default)

## Defaults (wichtig)
- Engine: crema
- Mode: pop
- DEFAULT_HOP_SEC = 0.10
- DEFAULT_WINDOW_SEC = 0.55
- DEFAULT_HOLD_UPDATES = 3
- DEFAULT_GATE = 0.010


### Besonderheiten:
✔ 7ths → Triads nur für Key-Finding (Pop)
✔ Relative-Major-Tie-Breaker (C-Dur gewinnt gegen A-moll bei I-vi-IV-V)
✔ DSP-Chord-Detector (FFT + Chroma)
✔ Keine Abhängigkeit von Crema → DSP funktioniert immer
✔ Kompatibel mit dem aktuellen Frontend

## Start
### Backend
#(Bei PowerShell ggf. vorher conda init powershell nötig)
touch backend/__init__.py
cd backend
conda activate chord_tf 
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8002
# start in Browser with URL:
http://127.0.0.1:8002/

### Frontend
conda activate chord_tf 
cd frontend
python -m http.server 5173

# start in Browser with URL:
http://127.0.0.1:5173/


# Ordnerstruktur
+ backend
--_pycache__
-- main.py
--requirements.txt
+ frontend
--index.html
