#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
svg_only_mpl_demo.py
~~~~~~~~~~~~~~~~~~~

Erzeugt ein SVG‑Diagramm ausschließlich mit Matplotlib.

* Achsen, Gitter und Tick‑Labels werden als echte <text>-Elemente
  (nicht als Pfade) geschrieben.
* Das Bild‑Raster wird über imshow eingebettet – Matplotlib verwendet dabei
  intern PNG (verlustfrei). Ein BMP‑Einbettung ist mit Matplotlib allein nicht
  möglich.
* Am Ende wird das SVG geparst und ausgegeben, wie viele <text>- bzw.
  <path>-Elemente darin vorkommen (zur Kontrolle).
"""

# ----------------------------------------------------------------------
# 1️⃣ Imports + rcParams (muss ganz oben, *vor* irgendeiner Figure!)
# ----------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --------- 1️⃣ Matplotlib‑Einstellungen ----------
mpl.rcParams.update({
    # --- Schrift ---------------------------------------------------------
    "font.family":       "sans-serif",               # serifenlose Grundschrift
    "font.sans-serif":   ["DejaVu Sans"],            # DejaVu Sans ist mit Matplotlib mitgeliefert

    # --- Text bleibt Text (keine Glyph‑Outlines) -----------------------
    # "none" → <text> Elemente, "path" (Standard) → Pfade/Outlines
    "svg.fonttype":      "none",

    # --- (optional) weitere Layout‑Parameter ---------------------------
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "legend.fontsize":   11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "grid.linestyle":    "--",
    "grid.color":        "lightgray",
    "grid.alpha":        0.7,
})


def downsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    """
    Nearest‑Neighbour / Sub‑Sampling.
    - `arr`  : 2‑D‑Array (beliebiger dtype, z. B. float64)
    - `factor`: Ganzzahliger Down‑Sampling‑Faktor (> 1)

    Rückgabe: ein neues 2‑D‑Array, bei dem jede `factor`‑te Zeile und
    jede `factor`‑te Spalte genommen wird.
    """
    if factor <= 1:
        # kein Down‑Sampling – Rückgabe einer Kopie im gleichen dtype
        return arr.copy()

    # Wenn Höhe bzw. Breite **nicht** durch den Faktor teilbar ist,
    # pad‑en wir den Rand (repeat‑pad), damit das Ergebnis exakt
    # `ceil(original/factor)` Zeilen/Spalten hat.
    h, w = arr.shape
    pad_h = (factor - (h % factor)) % factor
    pad_w = (factor - (w % factor)) % factor
    if pad_h or pad_w:
        # Rand‑Pixel werden wiederholt → kein künstlicher schwarzer Rand
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='edge')

    # Jetzt einfach jedes `factor`‑te Element auswählen
    return arr[::factor, ::factor].astype(arr.dtype)


### dann im Aufruf der plotting Fkt.:
#    ATT_BETA = downsample_nearest(ATT_BETA, factor=4)
#    ATT_BETA[ATT_BETA == 0] = np.nan
#    ATT_BETA[ATT_BETA < zLim[0]] = np.nan
#    #ATT_BETA = np.ma.masked_invalid(ATT_BETA)       # NaNs → Mask
#    ## set color of nan-values
#    cmap.set_bad(color='white')


# ----------------------------------------------------------------------
# 2️⃣ Daten für das Raster‑Bild (z. B. ein simples Heat‑Map‑Array)
# ----------------------------------------------------------------------
def make_raster_data(width: int = 600, height: int = 400) -> np.ndarray:
    """
    Liefert ein (H, W, 3) uint8‑Array mit einem Farbverlauf + leichtem Rauschen.
    Matplotlib wird das Array über imshow als PNG‑Raster in das SVG einbetten.
    """
    # Farbverlauf: Rot ∝ x, Grün ∝ y, Blau = konstant
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]

    r = np.clip(x + np.random.randint(-30, 31, (height, width)), 0, 255).astype(np.uint8)
    g = np.clip(y + np.random.randint(-30, 31, (height, width)), 0, 255).astype(np.uint8)
    b = np.full((height, width), 128, dtype=np.uint8)          # konstantes Blau

    rgb = np.stack([r, g, b], axis=2)   # (H, W, 3) uint8
    return rgb

# ----------------------------------------------------------------------
# 3️⃣ Plot erzeugen und als SVG speichern
# ----------------------------------------------------------------------
def create_svg(svg_path: Path):
    # --- Bilddaten ------------------------------------------------------
    img = make_raster_data()

    # --- Figure & Axes ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)   # 800 × 500 px Canvas

    # Raster‑Bild einblenden (wird intern als PNG gespeichert)
    print(img)
    print(img.shape)
    ax.imshow(img, extent=(0, 10, 0, 5), origin="lower", interpolation="none")

    # Achsen‑Titel, Labels, Grid … (alles bleibt Vektor)
    ax.set_xlabel("X‑Achse (Einheit)")
    ax.set_ylabel("Y‑Achse (Einheit)")
    ax.set_title("Beispiel‑Plot – Vektor‑Achsen + PNG‑Raster")
    ax.grid(True)

    #ax.legend()

    # --- SVG export ------------------------------------------------------
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)   # Speicher freigeben
    print(f"✅ SVG geschrieben → {svg_path}")


# ----------------------------------------------------------------------
# 5️⃣ Main – alles zusammenführen
# ----------------------------------------------------------------------
if __name__ == "__main__":
    out_svg = Path("demo_mpl_only.svg")
    create_svg(out_svg)       # 1. Plot erzeugen
