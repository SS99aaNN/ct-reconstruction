# CT Reconstruction

## 环境安装

```bash
conda create -n gpu-recon  -c conda-forge libparallelproj parallelproj pytorch cupy cudatoolkit=11.8 tqdm python=3.10
```



## 已实现重建方法

1. FDK

2. OSEM

3. TFDK



## 其余工具

1. Water Pre-Correction
   - Sourbelle, Katia, M. Kachelrieb, and Willi A. Kalender. "Empirical water precorrection for cone-beam computed tomography." *IEEE Nuclear Science Symposium Conference Record, 2005*. Vol. 4. IEEE, 2005.
2. extended z-range
   - Grimmer, Rainer, et al. "Cone‐beam CT image reconstruction with extended range." *Medical physics* 36.7 (2009): 3363-3370.



## 未实现

1. Motion Correction
   - Rit, Simon, et al. "On‐the‐fly motion‐compensated cone‐beam CT using an a priori model of the respiratory motion." *Medical physics* 36.6Part1 (2009): 2283-2296.
