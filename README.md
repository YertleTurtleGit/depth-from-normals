[![Map Pipeline](https://github.com/YertleTurtleGit/depth-from-normals/actions/workflows/map_pipeline.yml/badge.svg)](https://github.com/YertleTurtleGit/depth-from-normals/actions/workflows/map_pipeline.yml)
[![Lint](https://github.com/YertleTurtleGit/depth-from-normals/actions/workflows/lint.yml/badge.svg)](https://github.com/YertleTurtleGit/depth-from-normals/actions/workflows/lint.yml)
<a target="_blank" href="https://colab.research.google.com/github/YertleTurtleGit/depth-from-normals/blob/main/README.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Introduction


This algorithm utilizes the normal mapping to approximate a 3D integral by means of surface integrals of vector fields. Initially, the directional gradients of the normals are determined along the x- and y-directions. These gradients are then used to compute the integrated values by employing a cumulative sum (Riemann sum). To enhance the accuracy of the estimated values, this process is repeated multiple times with the gradient mapping rotated in different orientations, and the results are averaged.

# Quickstart

Just run the following in your CLI install all necessary packages on your machine, before you continue.


```python
!git clone https://github.com/YertleTurtleGit/depth-from-normals.git
%cd depth-from-normals/
%pip install -q -r requirements.txt
```

    Cloning into 'depth-from-normals'...


    remote: Enumerating objects: 1850, done.[K
    remote: Counting objects:   0% (1/448)[Kremote: Counting objects:   1% (5/448)[Kremote: Counting objects:   2% (9/448)[Kremote: Counting objects:   3% (14/448)[Kremote: Counting objects:   4% (18/448)[Kremote: Counting objects:   5% (23/448)[Kremote: Counting objects:   6% (27/448)[Kremote: Counting objects:   7% (32/448)[Kremote: Counting objects:   8% (36/448)[Kremote: Counting objects:   9% (41/448)[Kremote: Counting objects:  10% (45/448)[Kremote: Counting objects:  11% (50/448)[Kremote: Counting objects:  12% (54/448)[Kremote: Counting objects:  13% (59/448)[Kremote: Counting objects:  14% (63/448)[Kremote: Counting objects:  15% (68/448)[Kremote: Counting objects:  16% (72/448)[Kremote: Counting objects:  17% (77/448)[Kremote: Counting objects:  18% (81/448)[Kremote: Counting objects:  19% (86/448)[Kremote: Counting objects:  20% (90/448)[Kremote: Counting objects:  21% (95/448)[Kremote: Counting objects:  22% (99/448)[Kremote: Counting objects:  23% (104/448)[Kremote: Counting objects:  24% (108/448)[Kremote: Counting objects:  25% (112/448)[Kremote: Counting objects:  26% (117/448)[Kremote: Counting objects:  27% (121/448)[Kremote: Counting objects:  28% (126/448)[Kremote: Counting objects:  29% (130/448)[Kremote: Counting objects:  30% (135/448)[Kremote: Counting objects:  31% (139/448)[Kremote: Counting objects:  32% (144/448)[Kremote: Counting objects:  33% (148/448)[Kremote: Counting objects:  34% (153/448)[Kremote: Counting objects:  35% (157/448)[Kremote: Counting objects:  36% (162/448)[Kremote: Counting objects:  37% (166/448)[Kremote: Counting objects:  38% (171/448)[Kremote: Counting objects:  39% (175/448)[Kremote: Counting objects:  40% (180/448)[Kremote: Counting objects:  41% (184/448)[Kremote: Counting objects:  42% (189/448)[Kremote: Counting objects:  43% (193/448)[Kremote: Counting objects:  44% (198/448)[Kremote: Counting objects:  45% (202/448)[Kremote: Counting objects:  46% (207/448)[Kremote: Counting objects:  47% (211/448)[Kremote: Counting objects:  48% (216/448)[Kremote: Counting objects:  49% (220/448)[Kremote: Counting objects:  50% (224/448)[Kremote: Counting objects:  51% (229/448)[Kremote: Counting objects:  52% (233/448)[Kremote: Counting objects:  53% (238/448)[Kremote: Counting objects:  54% (242/448)[Kremote: Counting objects:  55% (247/448)[Kremote: Counting objects:  56% (251/448)[Kremote: Counting objects:  57% (256/448)[Kremote: Counting objects:  58% (260/448)[Kremote: Counting objects:  59% (265/448)[Kremote: Counting objects:  60% (269/448)[Kremote: Counting objects:  61% (274/448)[Kremote: Counting objects:  62% (278/448)[Kremote: Counting objects:  63% (283/448)[Kremote: Counting objects:  64% (287/448)[Kremote: Counting objects:  65% (292/448)[Kremote: Counting objects:  66% (296/448)[Kremote: Counting objects:  67% (301/448)[Kremote: Counting objects:  68% (305/448)[Kremote: Counting objects:  69% (310/448)[Kremote: Counting objects:  70% (314/448)[Kremote: Counting objects:  71% (319/448)[Kremote: Counting objects:  72% (323/448)[Kremote: Counting objects:  73% (328/448)[Kremote: Counting objects:  74% (332/448)[Kremote: Counting objects:  75% (336/448)[Kremote: Counting objects:  76% (341/448)[Kremote: Counting objects:  77% (345/448)[Kremote: Counting objects:  78% (350/448)[Kremote: Counting objects:  79% (354/448)[Kremote: Counting objects:  80% (359/448)[Kremote: Counting objects:  81% (363/448)[Kremote: Counting objects:  82% (368/448)[Kremote: Counting objects:  83% (372/448)[Kremote: Counting objects:  84% (377/448)[Kremote: Counting objects:  85% (381/448)[K

    remote: Counting objects:  86% (386/448)[Kremote: Counting objects:  87% (390/448)[Kremote: Counting objects:  88% (395/448)[Kremote: Counting objects:  89% (399/448)[Kremote: Counting objects:  90% (404/448)[Kremote: Counting objects:  91% (408/448)[Kremote: Counting objects:  92% (413/448)[Kremote: Counting objects:  93% (417/448)[Kremote: Counting objects:  94% (422/448)[Kremote: Counting objects:  95% (426/448)[Kremote: Counting objects:  96% (431/448)[Kremote: Counting objects:  97% (435/448)[Kremote: Counting objects:  98% (440/448)[Kremote: Counting objects:  99% (444/448)[Kremote: Counting objects: 100% (448/448)[Kremote: Counting objects: 100% (448/448), done.[K
    remote: Compressing objects:   0% (1/207)[Kremote: Compressing objects:   1% (3/207)[Kremote: Compressing objects:   2% (5/207)[Kremote: Compressing objects:   3% (7/207)[Kremote: Compressing objects:   4% (9/207)[Kremote: Compressing objects:   5% (11/207)[Kremote: Compressing objects:   6% (13/207)[Kremote: Compressing objects:   7% (15/207)[Kremote: Compressing objects:   8% (17/207)[Kremote: Compressing objects:   9% (19/207)[Kremote: Compressing objects:  10% (21/207)[Kremote: Compressing objects:  11% (23/207)[Kremote: Compressing objects:  12% (25/207)[Kremote: Compressing objects:  13% (27/207)[Kremote: Compressing objects:  14% (29/207)[Kremote: Compressing objects:  15% (32/207)[Kremote: Compressing objects:  16% (34/207)[Kremote: Compressing objects:  17% (36/207)[Kremote: Compressing objects:  18% (38/207)[Kremote: Compressing objects:  19% (40/207)[Kremote: Compressing objects:  20% (42/207)[Kremote: Compressing objects:  21% (44/207)[Kremote: Compressing objects:  22% (46/207)[Kremote: Compressing objects:  23% (48/207)[Kremote: Compressing objects:  24% (50/207)[Kremote: Compressing objects:  25% (52/207)[Kremote: Compressing objects:  26% (54/207)[Kremote: Compressing objects:  27% (56/207)[Kremote: Compressing objects:  28% (58/207)[Kremote: Compressing objects:  29% (61/207)[Kremote: Compressing objects:  30% (63/207)[Kremote: Compressing objects:  31% (65/207)[Kremote: Compressing objects:  32% (67/207)[Kremote: Compressing objects:  33% (69/207)[Kremote: Compressing objects:  34% (71/207)[Kremote: Compressing objects:  35% (73/207)[Kremote: Compressing objects:  36% (75/207)[Kremote: Compressing objects:  37% (77/207)[Kremote: Compressing objects:  38% (79/207)[Kremote: Compressing objects:  39% (81/207)[Kremote: Compressing objects:  40% (83/207)[Kremote: Compressing objects:  41% (85/207)[Kremote: Compressing objects:  42% (87/207)[Kremote: Compressing objects:  43% (90/207)[Kremote: Compressing objects:  44% (92/207)[Kremote: Compressing objects:  45% (94/207)[Kremote: Compressing objects:  46% (96/207)[Kremote: Compressing objects:  47% (98/207)[Kremote: Compressing objects:  48% (100/207)[Kremote: Compressing objects:  49% (102/207)[Kremote: Compressing objects:  50% (104/207)[Kremote: Compressing objects:  51% (106/207)[Kremote: Compressing objects:  52% (108/207)[Kremote: Compressing objects:  53% (110/207)[Kremote: Compressing objects:  54% (112/207)[Kremote: Compressing objects:  55% (114/207)[Kremote: Compressing objects:  56% (116/207)[Kremote: Compressing objects:  57% (118/207)[Kremote: Compressing objects:  58% (121/207)[Kremote: Compressing objects:  59% (123/207)[Kremote: Compressing objects:  60% (125/207)[Kremote: Compressing objects:  61% (127/207)[Kremote: Compressing objects:  62% (129/207)[Kremote: Compressing objects:  63% (131/207)[Kremote: Compressing objects:  64% (133/207)[Kremote: Compressing objects:  65% (135/207)[Kremote: Compressing objects:  66% (137/207)[Kremote: Compressing objects:  67% (139/207)[Kremote: Compressing objects:  68% (141/207)[Kremote: Compressing objects:  69% (143/207)[Kremote: Compressing objects:  70% (145/207)[Kremote: Compressing objects:  71% (147/207)[Kremote: Compressing objects:  72% (150/207)[Kremote: Compressing objects:  73% (152/207)[Kremote: Compressing objects:  74% (154/207)[Kremote: Compressing objects:  75% (156/207)[Kremote: Compressing objects:  76% (158/207)[Kremote: Compressing objects:  77% (160/207)[Kremote: Compressing objects:  78% (162/207)[Kremote: Compressing objects:  79% (164/207)[Kremote: Compressing objects:  80% (166/207)[Kremote: Compressing objects:  81% (168/207)[Kremote: Compressing objects:  82% (170/207)[Kremote: Compressing objects:  83% (172/207)[Kremote: Compressing objects:  84% (174/207)[Kremote: Compressing objects:  85% (176/207)[Kremote: Compressing objects:  86% (179/207)[Kremote: Compressing objects:  87% (181/207)[Kremote: Compressing objects:  88% (183/207)[Kremote: Compressing objects:  89% (185/207)[Kremote: Compressing objects:  90% (187/207)[Kremote: Compressing objects:  91% (189/207)[Kremote: Compressing objects:  92% (191/207)[Kremote: Compressing objects:  93% (193/207)[Kremote: Compressing objects:  94% (195/207)[Kremote: Compressing objects:  95% (197/207)[Kremote: Compressing objects:  96% (199/207)[Kremote: Compressing objects:  97% (201/207)[Kremote: Compressing objects:  98% (203/207)[Kremote: Compressing objects:  99% (205/207)[Kremote: Compressing objects: 100% (207/207)[Kremote: Compressing objects: 100% (207/207), done.[K
    Receiving objects:   0% (1/1850)Receiving objects:   1% (19/1850)

    Receiving objects:   2% (37/1850)Receiving objects:   3% (56/1850)Receiving objects:   4% (74/1850)Receiving objects:   5% (93/1850)Receiving objects:   6% (111/1850)Receiving objects:   7% (130/1850)Receiving objects:   8% (148/1850)

    Receiving objects:   9% (167/1850)Receiving objects:  10% (185/1850)Receiving objects:  11% (204/1850)Receiving objects:  12% (222/1850)Receiving objects:  13% (241/1850)Receiving objects:  14% (259/1850)Receiving objects:  15% (278/1850)Receiving objects:  16% (296/1850)Receiving objects:  17% (315/1850)Receiving objects:  18% (333/1850)

    Receiving objects:  19% (352/1850)Receiving objects:  20% (370/1850)Receiving objects:  21% (389/1850)Receiving objects:  22% (407/1850)Receiving objects:  23% (426/1850)Receiving objects:  24% (444/1850)Receiving objects:  25% (463/1850)Receiving objects:  26% (481/1850)Receiving objects:  27% (500/1850)Receiving objects:  28% (518/1850)Receiving objects:  29% (537/1850)Receiving objects:  30% (555/1850)Receiving objects:  31% (574/1850)Receiving objects:  32% (592/1850)Receiving objects:  33% (611/1850)Receiving objects:  34% (629/1850)Receiving objects:  35% (648/1850)Receiving objects:  36% (666/1850)Receiving objects:  37% (685/1850)Receiving objects:  38% (703/1850)Receiving objects:  39% (722/1850)Receiving objects:  40% (740/1850)

    Receiving objects:  41% (759/1850)Receiving objects:  42% (777/1850)Receiving objects:  43% (796/1850)Receiving objects:  44% (814/1850)Receiving objects:  45% (833/1850)Receiving objects:  46% (851/1850)Receiving objects:  47% (870/1850)Receiving objects:  48% (888/1850)Receiving objects:  49% (907/1850)Receiving objects:  50% (925/1850)Receiving objects:  51% (944/1850)Receiving objects:  52% (962/1850)Receiving objects:  53% (981/1850)

    Receiving objects:  54% (999/1850)Receiving objects:  55% (1018/1850)Receiving objects:  56% (1036/1850)Receiving objects:  57% (1055/1850)Receiving objects:  58% (1073/1850)Receiving objects:  59% (1092/1850)Receiving objects:  60% (1110/1850)Receiving objects:  61% (1129/1850)Receiving objects:  62% (1147/1850)Receiving objects:  63% (1166/1850)Receiving objects:  64% (1184/1850)Receiving objects:  65% (1203/1850)Receiving objects:  66% (1221/1850)Receiving objects:  67% (1240/1850)

    Receiving objects:  68% (1258/1850)Receiving objects:  69% (1277/1850)Receiving objects:  70% (1295/1850)Receiving objects:  71% (1314/1850)

    Receiving objects:  72% (1332/1850)

    Receiving objects:  73% (1351/1850)Receiving objects:  74% (1369/1850)Receiving objects:  75% (1388/1850)Receiving objects:  76% (1406/1850)Receiving objects:  77% (1425/1850)Receiving objects:  78% (1443/1850)Receiving objects:  79% (1462/1850)Receiving objects:  80% (1480/1850)Receiving objects:  81% (1499/1850)Receiving objects:  82% (1517/1850)Receiving objects:  83% (1536/1850)Receiving objects:  84% (1554/1850)Receiving objects:  85% (1573/1850)Receiving objects:  86% (1591/1850)Receiving objects:  87% (1610/1850)Receiving objects:  88% (1628/1850)Receiving objects:  89% (1647/1850)

    Receiving objects:  90% (1665/1850), 31.59 MiB | 63.16 MiB/sReceiving objects:  91% (1684/1850), 31.59 MiB | 63.16 MiB/sReceiving objects:  92% (1702/1850), 31.59 MiB | 63.16 MiB/sReceiving objects:  93% (1721/1850), 31.59 MiB | 63.16 MiB/s

    Receiving objects:  94% (1739/1850), 31.59 MiB | 63.16 MiB/sReceiving objects:  95% (1758/1850), 31.59 MiB | 63.16 MiB/sReceiving objects:  96% (1776/1850), 31.59 MiB | 63.16 MiB/s

    Receiving objects:  97% (1795/1850), 31.59 MiB | 63.16 MiB/sReceiving objects:  98% (1813/1850), 31.59 MiB | 63.16 MiB/sReceiving objects:  99% (1832/1850), 31.59 MiB | 63.16 MiB/sremote: Total 1850 (delta 244), reused 435 (delta 237), pack-reused 1402[K
    Receiving objects: 100% (1850/1850), 31.59 MiB | 63.16 MiB/sReceiving objects: 100% (1850/1850), 39.90 MiB | 64.14 MiB/s, done.
    Resolving deltas:   0% (0/1013)Resolving deltas:   1% (11/1013)Resolving deltas:   2% (21/1013)Resolving deltas:   3% (31/1013)Resolving deltas:   4% (41/1013)Resolving deltas:   5% (51/1013)Resolving deltas:   6% (61/1013)Resolving deltas:   7% (71/1013)Resolving deltas:   8% (82/1013)Resolving deltas:   9% (92/1013)Resolving deltas:  10% (102/1013)Resolving deltas:  11% (112/1013)Resolving deltas:  12% (122/1013)Resolving deltas:  13% (132/1013)Resolving deltas:  14% (142/1013)Resolving deltas:  15% (152/1013)Resolving deltas:  16% (163/1013)Resolving deltas:  17% (173/1013)Resolving deltas:  18% (183/1013)Resolving deltas:  19% (193/1013)Resolving deltas:  20% (203/1013)Resolving deltas:  21% (213/1013)Resolving deltas:  22% (223/1013)Resolving deltas:  23% (233/1013)Resolving deltas:  24% (244/1013)Resolving deltas:  25% (254/1013)Resolving deltas:  26% (264/1013)Resolving deltas:  27% (274/1013)Resolving deltas:  28% (284/1013)Resolving deltas:  29% (294/1013)Resolving deltas:  30% (304/1013)Resolving deltas:  31% (315/1013)Resolving deltas:  32% (325/1013)Resolving deltas:  33% (335/1013)Resolving deltas:  34% (345/1013)Resolving deltas:  35% (355/1013)Resolving deltas:  36% (365/1013)Resolving deltas:  37% (375/1013)Resolving deltas:  38% (385/1013)Resolving deltas:  39% (396/1013)Resolving deltas:  40% (406/1013)Resolving deltas:  41% (416/1013)Resolving deltas:  42% (427/1013)Resolving deltas:  43% (436/1013)Resolving deltas:  44% (446/1013)Resolving deltas:  45% (456/1013)Resolving deltas:  46% (466/1013)Resolving deltas:  47% (477/1013)Resolving deltas:  48% (487/1013)Resolving deltas:  49% (497/1013)

    Resolving deltas:  50% (507/1013)Resolving deltas:  51% (517/1013)Resolving deltas:  52% (527/1013)Resolving deltas:  53% (537/1013)Resolving deltas:  54% (548/1013)Resolving deltas:  55% (558/1013)Resolving deltas:  56% (568/1013)Resolving deltas:  57% (578/1013)Resolving deltas:  58% (588/1013)Resolving deltas:  59% (598/1013)Resolving deltas:  60% (608/1013)Resolving deltas:  61% (618/1013)Resolving deltas:  62% (629/1013)Resolving deltas:  63% (639/1013)Resolving deltas:  64% (649/1013)Resolving deltas:  65% (659/1013)Resolving deltas:  66% (669/1013)Resolving deltas:  67% (679/1013)Resolving deltas:  68% (689/1013)Resolving deltas:  69% (699/1013)Resolving deltas:  70% (710/1013)Resolving deltas:  71% (720/1013)Resolving deltas:  72% (730/1013)Resolving deltas:  73% (740/1013)Resolving deltas:  74% (750/1013)Resolving deltas:  75% (760/1013)Resolving deltas:  76% (770/1013)Resolving deltas:  77% (781/1013)Resolving deltas:  78% (791/1013)Resolving deltas:  79% (801/1013)Resolving deltas:  80% (811/1013)Resolving deltas:  81% (821/1013)Resolving deltas:  82% (831/1013)Resolving deltas:  83% (841/1013)Resolving deltas:  84% (851/1013)Resolving deltas:  85% (862/1013)Resolving deltas:  86% (872/1013)Resolving deltas:  87% (882/1013)Resolving deltas:  88% (892/1013)Resolving deltas:  89% (902/1013)Resolving deltas:  90% (912/1013)Resolving deltas:  91% (922/1013)Resolving deltas:  92% (932/1013)Resolving deltas:  93% (943/1013)Resolving deltas:  94% (953/1013)Resolving deltas:  95% (963/1013)Resolving deltas:  96% (973/1013)Resolving deltas:  97% (983/1013)Resolving deltas:  98% (993/1013)Resolving deltas:  99% (1003/1013)

    Resolving deltas: 100% (1013/1013)Resolving deltas: 100% (1013/1013), done.


    /home/runner/work/depth-from-normals/depth-from-normals/depth-from-normals


    /home/runner/.local/lib/python3.10/site-packages/IPython/core/magics/osm.py:417: UserWarning: This is now an optional IPython functionality, setting dhist requires you to install the `pickleshare` library.
      self.shell.db['dhist'] = compress_dhist(dhist)[-100:]


    Note: you may need to restart the kernel to use updated packages.



```python
from height_map import (
    estimate_height_map,
)  # local file 'height_map.py' in this repository
from matplotlib import pyplot as plt
import numpy as np
from skimage import io

NORMAL_MAP_A_PATH: str = (
    "https://raw.githubusercontent.com/YertleTurtleGit/depth-from-normals/main/normal_mapping_a.png"
)
NORMAL_MAP_B_PATH: str = (
    "https://raw.githubusercontent.com/YertleTurtleGit/depth-from-normals/main/normal_mapping_b.png"
)
NORMAL_MAP_A_IMAGE: np.ndarray = io.imread(NORMAL_MAP_A_PATH)
NORMAL_MAP_B_IMAGE: np.ndarray = io.imread(NORMAL_MAP_B_PATH)
```


```python
heights = estimate_height_map(NORMAL_MAP_A_IMAGE, raw_values=True)

figure, axes = plt.subplots(1, 2, figsize=(7, 3))
_ = axes[0].imshow(NORMAL_MAP_A_IMAGE)
_ = axes[1].imshow(heights)

x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
_, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
_ = axes.scatter(x, y, heights, c=heights)
```


    
![png](README_files/README_7_0.png)
    



    
![png](README_files/README_7_1.png)
    


# Explanation



```python
import cv2 as cv
from scipy.integrate import cumulative_trapezoid, simpson
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count
from typing import List, Tuple
from matplotlib.colors import TwoSlopeNorm
```

## Gradients

By deriving gradients in the x- and y-directions from the normal mapping, a mapping of angles is generated, which can be utilized to compute the directional gradients.

Given the normal vector $\vec{n} \in \mathbb{R}^{3}$ and a rotation value $r \in \mathbb{R}[0,2\pi]$, the anisotropic gradients are calculated:

$$
a_h = \arccos{\vec{n_x}}, \hspace{5px} g_l = (1 - \sin{a_h}) * sgn(a_h - \frac{\pi}{2})
$$

$$
a_v = \arccos{\vec{n_y}}, \hspace{5px} g_t = (1 - \sin{a_v}) * sgn(a_v - \frac{\pi}{2})
$$



```python
def calculate_gradients(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    horizontal_angle_map = np.arccos(np.clip(normals[:, :, 0], -1, 1))
    left_gradients = (1 - np.sin(horizontal_angle_map)) * np.sign(
        horizontal_angle_map - np.pi / 2
    )

    vertical_angle_map = np.arccos(np.clip(normals[:, :, 1], -1, 1))
    top_gradients = -(1 - np.sin(vertical_angle_map)) * np.sign(
        vertical_angle_map - np.pi / 2
    )

    return left_gradients, top_gradients


normals = ((NORMAL_MAP_A_IMAGE[:, :, :3] / 255) - 0.5) * 2
left_gradients, top_gradients = calculate_gradients(normals)


figsize = (14, 14)
figure, axes = plt.subplots(1, 3, figsize=figsize)
axes[0].set_title("anisotropic left gradients (left to right)")
_ = axes[0].imshow(left_gradients, cmap="RdBu", norm=TwoSlopeNorm(0))
axes[1].set_title("anisotropic top gradients (top to bottom)")
_ = axes[1].imshow(top_gradients, cmap="RdBu", norm=TwoSlopeNorm(0))
axes[2].set_title("normals (clipped)")
_ = axes[2].imshow(np.clip(normals, 0, 255))
```


    
![png](README_files/README_11_0.png)
    


## Heights

The height values $h(x,y) \in \mathbb{R}^{2}, \ \ x,y \in \mathbb{N}^{0}$ can be obtained by performing a cumulative sum over the gradients, which eventually approaches an integral over $g(x,y)$:

$$
h(x_t,y_t) = \iint g(x,y) dydx \ \ (x_t,y_t) \approx \sum_{x_i=0}^{x_t} g(x_i,y_t)
$$

The isotropic (non-directional) heights are determined by a combination of all the anisotropic heights.



```python
def integrate_gradient_field(gradient_field: np.ndarray, axis: int) -> np.ndarray:
    return np.cumsum(gradient_field, axis=axis)


def calculate_heights(
    left_gradients: np.ndarray, top_gradients: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_heights = integrate_gradient_field(left_gradients, axis=1)
    right_heights = np.fliplr(
        integrate_gradient_field(np.fliplr(-left_gradients), axis=1)
    )
    top_heights = integrate_gradient_field(top_gradients, axis=0)
    bottom_heights = np.flipud(
        integrate_gradient_field(np.flipud(-top_gradients), axis=0)
    )
    return left_heights, right_heights, top_heights, bottom_heights


left_heights, right_heights, top_heights, bottom_heights = calculate_heights(
    left_gradients, top_gradients
)


def combine_heights(*heights: np.ndarray) -> np.ndarray:
    return np.mean(np.stack(heights, axis=0), axis=0)


isotropic_heights = combine_heights(
    left_heights, right_heights, top_heights, bottom_heights
)


def visualize_heights(heights_list: List[np.ndarray], labels: List[str]):
    if len(heights_list) == 1:
        heights = heights_list[0]
        plt.title(labels[0])
        _ = plt.imshow(heights)
        x, y = np.meshgrid(range(heights.shape[1]), range(heights.shape[0]))
        _, axes = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        _ = axes.scatter(x, y, heights, c=heights)
        return

    figure, axes = plt.subplots(1, len(heights_list), figsize=(19, 5))
    for index, heights in enumerate(heights_list):
        axes[index].set_title(labels[index])
        _ = axes[index].imshow(heights, norm=TwoSlopeNorm(0.5))

    x, y = np.meshgrid(range(left_heights.shape[0]), range(left_heights.shape[1]))
    figure, axes = plt.subplots(
        1, len(heights_list), subplot_kw={"projection": "3d"}, figsize=(19, 5)
    )
    for index, heights in enumerate(heights_list):
        _ = axes[index].scatter(x, y, heights, c=heights)


visualize_heights(
    [left_heights, right_heights, top_heights, bottom_heights, isotropic_heights],
    [
        "anisotropic left heights",
        "anisotropic right heights",
        "anisotropic top heights",
        "anisotropic bottom heights",
        "isotropic heights",
    ],
)
```


    
![png](README_files/README_13_0.png)
    



    
![png](README_files/README_13_1.png)
    


## Rotation

While using the cumulative sum (Riemann sum) to calculate the height map is a straightforward and efficient method, it may result in errors, particularly when the gradient mapping contains abrupt changes in direction. In order to mitigate such errors, the estimate_height_map function utilizes multiple rotated versions of the gradient mapping and computes their averages to generate the height map. This approach aids in the reduction of errors and enhances the precision of the height map estimates.

$$
h(x_t,y_t) = \sum_{r=0}^{2\pi} \sum_{x_i=0}^{x_t} g R_\theta (x_i,y_t)
$$

To refer to the height maps in polar coordinates representing the left, right, top, and bottom, it would be more appropriate to name them as 180°, 0°, 90°, and 270° height maps, respectively.



```python
plt.polar()
_ = plt.yticks([1])
```


    
![png](README_files/README_15_0.png)
    


When computing an anisotropic height map for a 225° direction, it is necessary to first rotate the normal map. However, a standard image rotation technique may lead to incorrect normal vectors, hence it is also essential to perform a corresponding rotation of the normal vectors.


```python
ANGLE = 225


def rotate(matrix: np.ndarray, angle: float) -> np.ndarray:
    h, w = matrix.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    corners = cv.transform(
        np.array([[[0, 0], [w, 0], [w, h], [0, h]]]), rotation_matrix
    )[0]

    _, _, w, h = cv.boundingRect(corners)

    rotation_matrix[0, 2] += w / 2 - center[0]
    rotation_matrix[1, 2] += h / 2 - center[1]
    result = cv.warpAffine(matrix, rotation_matrix, (w, h), flags=cv.INTER_LINEAR)

    return result


rotated_normal_map_wrong = rotate(NORMAL_MAP_A_IMAGE, ANGLE)

wrong_normals = (
    (rotated_normal_map_wrong[:, :, :3].astype(np.float64) / 255) - 0.5
) * 2


def rotate_vector_field_normals(normals: np.ndarray, angle: float) -> np.ndarray:
    angle = np.radians(angle)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotated_normals = np.empty_like(normals)
    rotated_normals[:, :, 0] = (
        normals[:, :, 0] * cos_angle - normals[:, :, 1] * sin_angle
    )
    rotated_normals[:, :, 1] = (
        normals[:, :, 0] * sin_angle + normals[:, :, 1] * cos_angle
    )

    return rotated_normals


rotated_normals = rotate_vector_field_normals(wrong_normals, ANGLE)
rotated_normal_map = (((rotated_normals + 1) / 2) * 255).astype(np.uint8)

figure, axes = plt.subplots(1, 3, figsize=figsize)
axes[0].set_title("normal map")
_ = axes[0].imshow(NORMAL_MAP_A_IMAGE)
axes[1].set_title("rotated normal map (wrong)")
_ = axes[1].imshow(rotated_normal_map_wrong)
axes[2].set_title("rotated normal map (correct)")
_ = axes[2].imshow(rotated_normal_map)
```


    
![png](README_files/README_17_0.png)
    



```python
def centered_crop(image: np.ndarray, target_resolution: Tuple[int, int]) -> np.ndarray:
    return image[
        (image.shape[0] - target_resolution[0])
        // 2 : (image.shape[0] - target_resolution[0])
        // 2
        + target_resolution[0],
        (image.shape[1] - target_resolution[1])
        // 2 : (image.shape[1] - target_resolution[1])
        // 2
        + target_resolution[1],
    ]
```


```python
def integrate_vector_field(
    vector_field: np.ndarray,
    target_iteration_count: int,
    thread_count: int = cpu_count(),
) -> np.ndarray:
    shape = vector_field.shape[:2]
    angles = np.linspace(0, 90, target_iteration_count, endpoint=False)

    def integrate_vector_field_angles(angles: List[float]) -> np.ndarray:
        all_combined_heights = np.zeros(shape)

        for angle in angles:
            rotated_vector_field = rotate_vector_field_normals(
                rotate(vector_field, angle), angle
            )

            left_gradients, top_gradients = calculate_gradients(rotated_vector_field)
            (
                left_heights,
                right_heights,
                top_heights,
                bottom_heights,
            ) = calculate_heights(left_gradients, top_gradients)

            combined_heights = combine_heights(
                left_heights, right_heights, top_heights, bottom_heights
            )
            combined_heights = centered_crop(rotate(combined_heights, -angle), shape)
            all_combined_heights += combined_heights / len(angles)

        return all_combined_heights

    with Pool(processes=thread_count) as pool:
        heights = pool.map(
            integrate_vector_field_angles,
            np.array(
                np.array_split(angles, thread_count),
                dtype=object,
            ),
        )
        pool.close()
        pool.join()

    isotropic_height = np.zeros(shape)
    for height in heights:
        isotropic_height += height / thread_count

    return isotropic_height


def estimate_height_map(
    normal_map: np.ndarray, target_iteration_count: int = 250
) -> np.ndarray:
    normals = ((normal_map[:, :, :3] / 255) - 0.5) * 2
    heights = integrate_vector_field(normals, target_iteration_count)
    return heights


figure, axes = plt.subplots(1, 4, figsize=(14, 6))

for index in range(4):
    target_iteration_count = max(1, index * 5)
    heights = estimate_height_map(NORMAL_MAP_B_IMAGE, target_iteration_count)
    x, y = np.meshgrid(range(heights.shape[0]), range(heights.shape[1]))

    axes[index].set_title(f"target iteration count: {target_iteration_count}")
    _ = axes[index].imshow(heights)
```


    
![png](README_files/README_19_0.png)
    


# Discussion

## Integration

The cumulative sum method is a rudimentary approach for computing integrals. In the following, we have implemented the trapezoid and Simpson's method to provide additional options for integral computation.


```python
from enum import auto, Enum


class INTEGRATION_METHODS(Enum):
    SUM = auto()
    TRAPEZOID = auto()
    SIMSPSON = auto()


def integrate_gradient_field(gradient_field: np.ndarray, axis: int) -> np.ndarray:
    if INTEGRATION_METHOD == INTEGRATION_METHOD.SUM:
        return np.cumsum(gradient_field, axis=axis)

    if INTEGRATION_METHOD == INTEGRATION_METHOD.TRAPEZOID:
        return cumulative_trapezoid(gradient_field, axis=axis, initial=0)

    if INTEGRATION_METHOD == INTEGRATION_METHOD.SIMSPSON:
        integral = np.zeros(gradient_field.shape[:2])

        if axis == 1:
            for y in range(gradient_field.shape[0]):
                for x in range(1, gradient_field.shape[1]):
                    integral[y, x] = simpson(gradient_field[y, :x])

        elif axis == 0:
            for x in range(gradient_field.shape[1]):
                for y in range(1, gradient_field.shape[0]):
                    integral[y, x] = simpson(gradient_field[:y, x])

        return integral

    raise NotImplementedError(
        f"Integration method '{INTEGRATION_METHOD}' not implemented."
    )


target_iteration_count = 1


INTEGRATION_METHOD = INTEGRATION_METHODS.TRAPEZOID
trapezoid_heights = estimate_height_map(NORMAL_MAP_A_IMAGE, target_iteration_count)

INTEGRATION_METHOD = INTEGRATION_METHODS.SIMSPSON
simpson_heights = estimate_height_map(NORMAL_MAP_A_IMAGE, target_iteration_count)

INTEGRATION_METHOD = INTEGRATION_METHODS.SUM
sum_heights = estimate_height_map(NORMAL_MAP_A_IMAGE, target_iteration_count)

visualize_heights(
    [sum_heights, trapezoid_heights, simpson_heights], ["sum", "trapezoid", "Simpson"]
)
```


    
![png](README_files/README_21_0.png)
    



    
![png](README_files/README_21_1.png)
    


Although they may appear similar and effectively provide equivalent results, the Simpson method demands considerably more computational time in this implementation than the sum and trapezoid methods. This outcome can be disheartening regarding the use of polynomial approximation to enhance the resulting heights in general.

## Confidence

A straightforward technique for computing the confidence of a pixel is to modify the heights combination function to return the negative standard deviation rather than the mean.


```python
def combine_heights(*heights: np.ndarray) -> np.ndarray:
    return -np.std(np.stack(heights, axis=0), axis=0)


confidences = estimate_height_map(NORMAL_MAP_A_IMAGE)
plt.title("confidences")
plt.xlabel(f"mean: {np.mean(confidences)}")
_ = plt.imshow(confidences)
_ = plt.colorbar()
```


    
![png](README_files/README_24_0.png)
    

