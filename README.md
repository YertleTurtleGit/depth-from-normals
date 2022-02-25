[![Jupyter to Python](https://github.com/YertleTurtleGit/photometric-stereo-mappings/actions/workflows/jupyter_to_python.yml/badge.svg?branch=main)](https://github.com/YertleTurtleGit/photometric-stereo-mappings/actions/workflows/jupyter_to_python.yml)

# Photometric Stereo Mappings (WIP)

|Mapping Type|Example Image|
|---|---|
|**Opacity Mapping** from Normal Mapping.<br><a href="https://colab.research.google.com/github/YertleTurtleGit/photometric-stereo-mappings/blob/main/1_opacity_map.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|<img title="Opacity Mapping" src="./test_dataset/output/opacity.png" width="200">|
|**Albedo Mapping**  from Differently Lit Images witch Exposure Fusion. [1]<br><a href="https://colab.research.google.com/github/YertleTurtleGit/photometric-stereo-mappings/blob/main/2_albedo_map.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|<img title="Albedo Mapping" src="./test_dataset/output/albedo.png" width="200">|
|**Translucency Mapping** from Differently Lit Images with Exposure Fusion. [^1]<br><a href="https://colab.research.google.com/github/YertleTurtleGit/photometric-stereo-mappings/blob/main/2_translucency_map.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|<img title="Translucency Mapping" src="./test_dataset/output/translucency.png" width="200">|
|**Normal Mapping** from Differently Lit Images with Photometric Stereo. [^2] [^3]<br><a href="https://colab.research.google.com/github/YertleTurtleGit/photometric-stereo-mappings/blob/main/2_normal_map.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|<img title="Normal Mapping" src="./test_dataset/output/normal.png" width="200">|
|**Height Mapping** from Normal Mapping with Averaged Integrals from Rotated Discrete Origin Functions.<br><a href="https://colab.research.google.com/github/YertleTurtleGit/photometric-stereo-mappings/blob/main/3_height_map.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|<img title="Height Mapping" src="./test_dataset/output/height.png" width="200">|
|**Ambient Occlusion Mapping** from Height Mapping with Normalized Blurred Difference<br><a href="https://colab.research.google.com/github/YertleTurtleGit/photometric-stereo-mappings/blob/main/4_ambient_occlusion_map.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|<img title="Ambient Occlusion Mapping" src="./test_dataset/output/ambient_occlusion.png" width="200">|
|**Roughness Mapping** from Normal Mapping with Inverse of Normalized Blurred .<br><a href="https://colab.research.google.com/github/YertleTurtleGit/photometric-stereo-mappings/blob/main/roughness_map.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|<img title="Roughness Mapping" src="./test_dataset/output/roughness.png" width="200">|

---
[^1]: Mertens, Tom, Jan Kautz, and Frank Van Reeth. "Exposure fusion." 15th Pacific Conference on Computer Graphics and Applications (PG'07). IEEE, 2007.

[^2]: Woodham, Robert J. "Photometric method for determining surface orientation from multiple images." Optical engineering 19.1 (1980): 139-144.

[^3]: Wu, Lun, et al. "Robust photometric stereo via low-rank matrix completion and recovery." Asian Conference on Computer Vision. Springer, Berlin, Heidelberg, 2010.