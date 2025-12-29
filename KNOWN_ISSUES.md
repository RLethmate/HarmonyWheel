
---

# C) KNOWN_ISSUES.md (fertige Zeilen)

```md
# Known Issues / Beobachtungen

## 1) Key-Detection wird durch Extensions verzerrt
Crema erkennt bei Gitarre gelegentlich "C7" statt "C".
Wenn Key-Estimation Dominant-7 wie echte V7 behandelt, entsteht ein Bias Richtung F-Dur/D-moll.

Fix: Im POP-Modus werden 7ths für Key-Finding zu Triads normalisiert:
- dom7/maj7 -> maj
- min7 -> min
- m7b5 -> dim

## 2) Relative Dur/Moll (C-Dur vs A-moll) ist oft ein Unentschieden
Bei Progressionen wie C–Am–F–G sind C-Dur und A-moll beide plausibel.
Heuristik nötig: Pop bevorzugt Major, wenn IV/V-Akkorde auftauchen.

## 3) Gitarre: Powerchords / schwache Terz → maj/min flip
Wenn die Terz schwach ist, kann die Erkennung zwischen G und Gm flackern.
Im Pop-Modus wird bei schwacher Terz MAJ bevorzugt.

## 4) Gate/Window/Hold beeinflussen Latenz vs Stabilität
- mehr Hold/Window = stabiler, aber mehr Delay
- weniger Hold/Window = schneller, aber mehr Flattern
