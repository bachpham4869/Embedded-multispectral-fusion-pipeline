# References and Links — Smart Binocular (IEEE Style)

**Document version:** 2026-04-28 (rev. 6 — §IV/§V retargeted to `docs/PIPELINE_EVIDENCE_REGISTER.md`; `ALGORITHMS_DETAILED.md` and `ALGORITHM_INTEGRATION_TIERS.md` consolidated into register)  
**Scope:** Sources cited or implied in `docs/PIPELINE_EVIDENCE_REGISTER.md` (Parts B–D; consolidates the former `ALGORITHMS_DETAILED.md` and `ALGORITHM_INTEGRATION_TIERS.md`), `smart_binocular/fusion_advanced.py`, and `smart_binocular/MODES_ALGORITHMS.md`, plus the LearnOpenCV tutorial requested by the project; **additionally**, Cursor research threads and `weekly_report/` (Apr. 2026) on NIRGB–GS (3D Gaussian Splatting + NIR), lightweight IR enhancement (CLAHE, RGF, FPGA implementations), Retinex surveys, IR–visible fusion, the LLVIP dataset, IMX290 / NIR-channel labeling for training, and `docs/DATASET_LOCAL_STATUS.md` (including the optional Kaggle *Landscape image colorization* dataset [35]); **additionally**, `docs/research_report_20260415_env_policy_branching_optical_v4.md` (ENV optical buckets A–F, DCP/dehaze, rain median, multi-label CC, temporal stabilization) — bibliography entries [36]–[65]; **additionally**, `docs/research_report_20260421_thesis_improvement.md` (RF calibration, ECE/reliability, BRISQUE/NIQE, SMOTE/imbalance, KAIST benchmark, Kalman tutorial, RPi OpenCV notes) — entries [66]–[77] below.  
**Citation style:** IEEE Transactions reference list conventions — initials + surname, *italic* journal/book, `vol.`, `no.`, `pp.`, month abbrev., year, `doi:` or arXiv ID; online sources use *[Online]* and *Available:*.

---

## I. How to cite in text (IEEE)

- Numeric brackets in line with text: thermal likelihood follows δ-GLMB ideas from Li and Wang [2].  
- For arXiv: include version if quoting a specific PDF (optional).  
- For web/tutorials: cite [18] and the **underlying paper** [12] when the tutorial implements that method.

---

## II. IEEE-formatted reference list

[1] W. Gao *et al.*, “Dim Small Target Detection and Tracking: A Novel Method Based on Temporal Energy Selective Scaling and Trajectory Association,” *arXiv preprint arXiv:2405.09054*, May 2024. [Online]. Available: https://arxiv.org/abs/2405.09054

[2] C. Li and W. Wang, “Detection and tracking of moving targets for thermal infrared video sequences,” *Sensors*, vol. 18, no. 11, p. 3944, Nov. 2018, doi: 10.3390/s18113944. [Online]. Available: https://www.mdpi.com/1424-8220/18/11/3944

[3] Y. Fan *et al.*, “IRSDT: A framework for infrared small target tracking with enhanced detection,” *Sensors*, vol. 23, no. 9, p. 4240, 2023, doi: 10.3390/s23094240. [Online]. Available: https://www.mdpi.com/1424-8220/23/9/4240

[4] G. Zhang *et al.*, “You Only Look Omni Gradient Backpropagation for Moving Infrared Small Target Detection,” *arXiv preprint arXiv:2511.13013*, Nov. 2025. [Online]. Available: https://arxiv.org/abs/2511.13013

[5] F. Wu *et al.*, “Neural Spatial-Temporal Tensor Representation for Infrared Small Target Detection,” *arXiv preprint arXiv:2412.17302*, Dec. 2024. [Online]. Available: https://arxiv.org/abs/2412.17302

[6] J. Liang *et al.*, “SwinIR: Image restoration using swin transformer,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW)*, 2021, pp. 1833–1844, doi: 10.1109/ICCVW54120.2021.00201. Preprint: *arXiv:2108.10257*. [Online]. Available: https://arxiv.org/abs/2108.10257

[7] Z. Liu *et al.*, “Swin transformer: Hierarchical vision transformer using shifted windows,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 10012–10022, doi: 10.1109/ICCV48922.2021.00986. (Architecture background for Swin-based denoising; see `docs/PIPELINE_EVIDENCE_REGISTER.md` §C for implemented algorithms.)

[8] J. Veraart *et al.*, “Denoising of diffusion MRI using random matrix theory,” *Neuroimage*, vol. 142, pp. 394–406, Nov. 2016, doi: 10.1016/j.neuroimage.2016.08.016.

[9] O. Barnich and M. Van Droogenbroeck, “ViBe: A universal background subtraction algorithm for video sequences,” *IEEE Trans. Image Process.*, vol. 20, no. 6, pp. 1709–1724, Jun. 2011, doi: 10.1109/TIP.2010.2101613.

[10] J. F. Henriques, R. Caseiro, P. Martins, and J. Batista, “High-speed tracking with kernelized correlation filters,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 37, no. 3, pp. 583–596, Mar. 2015, doi: 10.1109/TPAMI.2014.2345390. (KCF; cited as full tracking alternative; not implemented in the RPi4 pipeline — see `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.)

[11] R. E. Kalman, “A new approach to linear filtering and prediction problems,” *J. Basic Eng.*, vol. 82, no. 1, pp. 35–45, Mar. 1960, doi: 10.1115/1.3662552. (Standard Kalman filter; §9.3.)

[12] Z. Shi *et al.*, “Nighttime low illumination image enhancement with single image using bright/dark channel prior,” *EURASIP J. Image Video Process.*, vol. 2018, no. 1, p. 13, Feb. 2018, doi: 10.1186/s13640-018-0251-4. [Online]. Available: https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-018-0251-4

[13] K. He, J. Sun, and X. Tang, “Single image haze removal using dark channel prior,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2011, pp. 1956–1963, doi: 10.1109/CVPR.2011.5995751.

[14] K. He, J. Sun, and X. Tang, “Guided image filtering,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2010, pp. 1–14, doi: 10.1007/978-3-642-15561-1_6. (Used in many bright/dark-channel pipelines including tutorial [18].)

[15] K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in *Graphics Gems IV*, P. S. Heckbert, Ed. San Diego, CA, USA: Academic Press Professional, Inc., 1994, pp. 474–485.

[16] E. H. Land and J. J. McCann, “Lightness and Retinex theory,” *J. Opt. Soc. Amer.*, vol. 61, no. 1, pp. 1–11, Jan. 1971, doi: 10.1364/JOSA.61.000001.

[17] A. Samal and H. R. Lone, “Thermal vision: Pioneering non-invasive temperature tracking in congested spaces,” *arXiv preprint arXiv:2412.00863*, 2024. [Online]. Available: https://arxiv.org/abs/2412.00863

[18] V. Agarwal and L. Patnaik, “Improving illumination in night time images,” LearnOpenCV, Mar. 15, 2021. [Online]. Available: https://learnopencv.com/improving-illumination-in-night-time-images/ (Tutorial implementing bright/dark channel ideas from [12]; see also [13], [14].)

[19] *GST-IR* (generic signal-processing chain for uncooled IR: NUC, AGC, DRC, temporal/spatial NR, edge enhancement) — **vendor-specific**; cite camera/IP-core datasheet or white paper from the manufacturer used in deployment. Referenced in code comments as “GST-IR 3DNR.”

[20] A. Vandone, “[Title and venue to be completed from original 2011 source],” 2011. (Cited in `fusion_advanced.py` for cold-frame / correlation-style thermal processing; **full bibliographic data not present in-repo** — complete before thesis submission.)

[27] C. Yang *et al.*, “NIRGB-GS: Near-Infrared Assisted Low-Light Scene Reconstruction and Enhancement via Gaussian Splatting,” *Adv. Intell. Syst.*, Early Access, Apr. 2026, doi: 10.1002/aisy.202501258. [Online]. Available: https://doi.org/10.1002/aisy.202501258

[28] J. Liu *et al.*, “Multi-Scale FPGA-Based Infrared Image Enhancement by Using RGF and CLAHE,” *Sensors*, vol. 23, no. 19, p. 8101, Sep. 2023, doi: 10.3390/s23198101. [Online]. Available: https://www.mdpi.com/1424-8220/23/19/8101

[29] X. Wang *et al.*, “Lightweight and Real-Time Infrared Image Processor Based on FPGA,” *Sensors*, vol. 24, no. 4, p. 1333, Feb. 2024, doi: 10.3390/s24041333. [Online]. Available: https://www.mdpi.com/1424-8220/24/4/1333 (PMC mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC10893426/)

[30] R. Soundrapandiyan *et al.*, “A comprehensive survey on image enhancement techniques with special emphasis on infrared images,” *Multimedia Tools Appl.*, vol. 81, no. 7, pp. 9045–9077, 2022, doi: 10.1007/s11042-021-11250-y. [Online]. Available: https://doi.org/10.1007/s11042-021-11250-y

[31] G. Simone, M. Lecca, G. Gianini, and A. Rizzi, “Survey of methods and evaluation of Retinex-inspired image enhancers,” *J. Electron. Imag.*, vol. 31, no. 6, Dec. 2022, Art. no. 063055, doi: 10.1117/1.JEI.31.6.063055. [Online]. Available: https://doi.org/10.1117/1.JEI.31.6.063055

[32] S. Bansal *et al.*, “A novel low complexity retinex-based algorithm for enhancing low-light images,” *Multimedia Tools Appl.*, vol. 83, no. 10, pp. 29485–29504, May 2024, doi: 10.1007/s11042-023-16610-4. [Online]. Available: https://doi.org/10.1007/s11042-023-16610-4

[33] Y. Zou *et al.*, “Infrared and visible image fusion based on multi-scale transform and sparse low-rank representation,” *Frontiers Phys.*, vol. 13, Jul. 2025, Art. no. 1514476, doi: 10.3389/fphy.2025.1514476. [Online]. Available: https://www.frontiersin.org/articles/10.3389/fphy.2025.1514476/full

[34] X. Jia *et al.*, “LLVIP: A Visible-infrared Paired Dataset for Low-light Vision,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW)*, Oct. 2021, pp. 3489–3497, doi: 10.1109/ICCVW54120.2021.00389. *arXiv preprint arXiv:2108.10831*, Aug. 2021. [Online]. Available: https://doi.org/10.1109/ICCVW54120.2021.00389 ; https://arxiv.org/abs/2108.10831 ; project page: https://bupt-ai-cz.github.io/LLVIP/

[36] K. He, J. Sun, and X. Tang, “Single Image Haze Removal Using Dark Channel Prior,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009. [Online]. Available: https://doi.org/10.1109/CVPR.2009.5206515

[37] K. He, J. Sun, and X. Tang, “Single Image Haze Removal Using Dark Channel Prior,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 33, no. 12, pp. 2341–2353, Dec. 2011, doi: 10.1109/TPAMI.2011.195. (TPAMI journal version of DCP; complements [13] CVPR 2011 duplicate venue.) [Online]. Available: https://doi.org/10.1109/TPAMI.2011.195

[38] J.-P. Tarel and N. Hautière, “Fast Visibility Restoration from a Single Color or Gray Level Image,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2009, pp. 2201–2208, doi: 10.1109/ICCV.2009.5459251. [Online]. Available: https://doi.org/10.1109/ICCV.2009.5459251

[39] H. Bai *et al.*, “A new approach to develop optimal CLAHE algorithm,” *Optik*, vol. 133, pp. 52–63, 2017, doi: 10.1016/j.ijleo.2016.10.104. [Online]. Available: https://doi.org/10.1016/j.ijleo.2016.10.104

[40] A. M. Reza, “Realization of the contrast limited adaptive histogram equalization (CLAHE) for real-time image enhancement,” *J. VLSI Signal Process. Syst.*, vol. 38, no. 1, pp. 35–44, 2004, doi: 10.1023/B:VLSI.0000028532.53896.2b. [Online]. Available: https://doi.org/10.1023/B:VLSI.0000028532.53896.2b

[41] J. Y. Kim, L. S. Kim, and S. H. Hwang, “An advanced contrast enhancement using partially overlapped sub-block histogram equalization,” *IEEE Trans. Circuits Syst. Video Technol.*, vol. 11, no. 4, pp. 475–484, Apr. 2001, doi: 10.1109/76.915423. [Online]. Available: https://doi.org/10.1109/76.915423

[42] S. K. Nayar and V. Branzoi, “Adaptive Dynamic Range Imaging: Optical Control of Pixel Exposures Over Space and Time,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2003, pp. 1168–1175, doi: 10.1109/ICCV.2003.1238326. [Online]. Available: https://doi.org/10.1109/ICCV.2003.1238326

[43] E. Reinhard, M. Stark, P. Shirley, and J. Ferwerda, “Photographic tone reproduction for digital images,” *ACM Trans. Graph.*, vol. 21, no. 3, pp. 267–276, Jul. 2002, doi: 10.1145/566654.566575. [Online]. Available: https://doi.org/10.1145/566654.566575

[44] E. Reinhard *et al.*, *High Dynamic Range Imaging: Acquisition, Display, and Image-Based Lighting*, 2nd ed. San Francisco, CA, USA: Morgan Kaufmann, 2010.

[45] J. Read, B. Pfahringer, G. Holmes, and E. Frank, “Classifier chains for multi-label classification,” *Mach. Learn.*, vol. 85, no. 3, pp. 333–359, Dec. 2011, doi: 10.1007/s10994-011-5257-5. [Online]. Available: https://doi.org/10.1007/s10994-011-5257-5

[46] G. Tsoumakas and I. Katakis, “Multi-label classification: An overview,” *Int. J. Data Warehousing Mining*, vol. 3, no. 3, pp. 1–13, 2007, doi: 10.4018/jdwm.2007070101. [Online]. Available: https://doi.org/10.4018/jdwm.2007070101

[47] M. S. Sorower, “A literature survey on algorithms for multi-label learning,” M.S. thesis, Oregon State Univ., Corvallis, OR, USA, 2010.

[48] L.-W. Kang, C.-W. Lin, and Y.-H. Fu, “Automatic Single-Image-Based Rain Streaks Removal via Image Decomposition,” *IEEE Trans. Image Process.*, vol. 21, no. 4, pp. 1742–1755, Apr. 2012, doi: 10.1109/TIP.2011.2179070. [Online]. Available: https://doi.org/10.1109/TIP.2011.2179070

[49] A. K. Tripathi and S. Mukhopadhyay, “Removal of rain from videos: A review,” *Signal, Image Video Process.*, vol. 8, no. 8, pp. 1421–1430, Nov. 2014, doi: 10.1007/s11760-012-0397-6. (Report cites 2012; journal volume 8(8) often listed as 2014.) [Online]. Available: https://doi.org/10.1007/s11760-012-0397-6

[50] S. Gu *et al.*, “Joint Convolutional Analysis and Synthesis Sparse Representation for Single Image Layer Separation,” in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 1708–1716, doi: 10.1109/ICCV.2017.186. [Online]. Available: https://doi.org/10.1109/ICCV.2017.186

[51] Y. Li *et al.*, “Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 1633–1642, doi: 10.1109/CVPR.2019.00171. [Online]. Available: https://doi.org/10.1109/CVPR.2019.00171

[52] R. T. Tan, “Visibility in Bad Weather from a Single Image,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2008, pp. 1–8, doi: 10.1109/CVPR.2008.4587643. [Online]. Available: https://doi.org/10.1109/CVPR.2008.4587643

[53] R. Fattal, “Single Image Dehazing,” *ACM Trans. Graph.*, vol. 27, no. 3, Art. no. 72, Aug. 2008, doi: 10.1145/1360612.1360671. [Online]. Available: https://doi.org/10.1145/1360612.1360671

[54] Q. Zhu, J. Mai, and L. Shao, “A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior,” *IEEE Trans. Image Process.*, vol. 24, no. 11, pp. 3522–3533, Nov. 2015, doi: 10.1109/TIP.2015.2469960. [Online]. Available: https://doi.org/10.1109/TIP.2015.2469960

[55] D. Berman, T. Treibitz, and S. Avidan, “Non-local Image Dehazing,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 1674–1682, doi: 10.1109/CVPR.2016.183. [Online]. Available: https://doi.org/10.1109/CVPR.2016.183

[56] B. Li *et al.*, “Benchmarking Single-Image Dehazing and Beyond,” *IEEE Trans. Image Process.*, vol. 28, no. 1, pp. 492–505, Jan. 2019, doi: 10.1109/TIP.2018.2867051. [Online]. Available: https://doi.org/10.1109/TIP.2018.2867051

[57] E. D. Pisano *et al.*, “Contrast Limited Adaptive Histogram Equalization Image Processing to Improve the Detection of Simulated Spiculations in Dense Mammograms,” *J. Digit. Imaging*, vol. 11, no. 4, pp. 193–200, 1998, doi: 10.1007/BF03178082. [Online]. Available: https://doi.org/10.1007/BF03178082

[58] J. A. Stark, “Adaptive image contrast enhancement using generalizations of histogram equalization,” *IEEE Trans. Image Process.*, vol. 9, no. 5, pp. 889–896, May 2000, doi: 10.1109/83.841534. [Online]. Available: https://doi.org/10.1109/83.841534

[59] B. D. Lucas and T. Kanade, “An iterative image registration technique with an application to stereo vision,” in *Proc. Imaging Understanding Workshop*, 1981, pp. 121–130. (Often cited as IJCAI’81; DARPA IU Workshop.) [Online]. Available: https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_1981_2/lucas_bruce_1981_2.pdf

[60] V. Lepetit and P. Fua, “Keypoint recognition using randomized trees,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 28, no. 9, pp. 1465–1479, Sep. 2006, doi: 10.1109/TPAMI.2006.188. [Online]. Available: https://doi.org/10.1109/TPAMI.2006.188

[61] O. H. Schmitt, “A thermionic trigger,” *J. Sci. Instrum.*, vol. 15, no. 1, pp. 24–26, Jan. 1938, doi: 10.1088/0950-7671/15/1/305. (Historical “Schmitt trigger”; cited in report for hysteresis analogy.) [Online]. Available: https://doi.org/10.1088/0950-7671/15/1/305

[62] P. J. Werbos, “Backpropagation through time: What it does and how to do it,” *Proc. IEEE*, vol. 78, no. 10, pp. 1550–1560, Oct. 1990, doi: 10.1109/5.58337. [Online]. Available: https://doi.org/10.1109/5.58337

[63] L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, Oct. 2001, doi: 10.1023/A:1010933404324. [Online]. Available: https://doi.org/10.1023/A:1010933404324

[64] F. Pedregosa *et al.*, “Scikit-learn: Machine Learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html

[65] R. Raskar, A. Ilie, and J. Yu, “Image fusion for context enhancement and video surrealism,” in *Proc. Int. Symp. Non-Photorealistic Animation Rendering (NPAR)*, 2004, pp. 85–152, doi: 10.1145/987657.987669. [Online]. Available: https://doi.org/10.1145/987657.987669

[66] A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” in *Proc. 22nd Int. Conf. Mach. Learn. (ICML)*, Bonn, Germany, 2005, pp. 625–632.

[67] C. Leys *et al.*, “Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median,” *J. Experimental Social Psychology*, vol. 49, no. 4, pp. 764–766, Jul. 2013, doi: 10.1016/j.jesp.2013.03.013.

[68] S. M. Pizer *et al.*, “Adaptive histogram equalization and its variations,” *Computer Vision, Graphics, and Image Processing*, vol. 39, no. 3, pp. 355–368, Sep. 1987, doi: 10.1016/0734-189X(87)90116-X.

[69] A. Mittal, A. K. Moorthy, and A. C. Bovik, “No-reference image quality assessment in the spatial domain,” *IEEE Trans. Image Process.*, vol. 21, no. 12, pp. 4695–4708, Dec. 2012, doi: 10.1109/TIP.2012.2214050.

[70] A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a ‘completely blind’ image quality analyzer,” *IEEE Signal Process. Lett.*, vol. 20, no. 3, pp. 209–212, Mar. 2013, doi: 10.1109/LSP.2012.2227726.

[71] N. V. Chawla *et al.*, “SMOTE: synthetic minority over-sampling technique,” *J. Artificial Intelligence Res.*, vol. 16, pp. 321–357, 2002. [Online]. Available: https://www.jair.org/index.php/jair/article/view/10300

[72] S. Hwang *et al.*, “Multispectral pedestrian detection: Benchmark dataset and baseline,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2015, pp. 1037–1045, doi: 10.1109/CVPR.2015.7299082.

[73] G. Welch and G. Bishop, “An introduction to the Kalman filter,” Tech. Rep. TR 95-041, Dept. Computer Sci., Univ. North Carolina, Chapel Hill, NC, USA, Jul. 2006 (rev. of 1995). [Online]. Available: https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

[74] scikit-learn developers, “Probability calibration,” in *scikit-learn User Guide*, Accessed: Apr. 28, 2026. [Online]. Available: https://scikit-learn.org/stable/modules/calibration.html

[75] A. Rosebrock, “Optimizing OpenCV on the Raspberry Pi,” *PyImageSearch*, Oct. 9, 2017. [Online]. Available: https://pyimagesearch.com/2017/10/09/optimizing-opencv-on-the-raspberry-pi/

[76] C. Feng *et al.*, “IQA-PyTorch: PyTorch Toolbox for Image Quality Assessment,” GitHub repository, Accessed: Apr. 28, 2026. [Online]. Available: https://github.com/chaofengc/IQA-PyTorch

[77] Y. Zhang *et al.*, “Advances and challenges in infrared-visible image fusion: A comprehensive review,” *Artificial Intelligence Review*, vol. 58, no. 3, 2025, Art. no. 75, doi: 10.1007/s10462-025-11426-0. [Online]. Available: https://doi.org/10.1007/s10462-025-11426-0

---

## III. Software, APIs, and non-paper sources

These are **not** peer-reviewed articles; cite as manuals, **dataset repositories** (e.g. Kaggle [35]), or online resources.

[21] OpenCV contributors, *OpenCV Library*, Open Source, Accessed: Apr. 10, 2026. [Online]. Available: https://opencv.org/

[22] Raspberry Pi Ltd., *Picamera2* (libcamera-based Python API), Accessed: Apr. 10, 2026. [Online]. Available: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf (or current docs URL used in your environment.)

[23] ONNX Runtime contributors, *ONNX Runtime*, Accessed: Apr. 10, 2026. [Online]. Available: https://onnxruntime.ai/

[24] Google Coral team, *Coral Edge TPU documentation*, Accessed: Apr. 10, 2026. [Online]. Available: https://coral.ai/docs/

[25] Raspberry Pi Ltd., *vcgencmd* and system metrics (throttle, voltage), in *Raspberry Pi Documentation*, Accessed: Apr. 10, 2026. [Online]. Available: https://www.raspberrypi.com/documentation/computers/os.html

[26] Internal project artifact: Cursor chat transcript ID `53b7de6c-0fde-4abd-94a2-7683e6b759a6` (used to reconstruct integration tier priorities; **not** a citable academic source. See `docs/PIPELINE_EVIDENCE_REGISTER.md` §D for current integration parameters.)

[35] theblackmamba31, “Landscape image colorization,” *Kaggle* dataset repository, Accessed: Apr. 15, 2026. [Online]. Available: https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization (Optional offline data; listed in `docs/DATASET_LOCAL_STATUS.md` and `data/README.md`.)

---

## IV. Cross-index: pipeline algorithms → references

> **Note (2026-04-25):** The files `docs/ALGORITHMS_DETAILED.md` and
> `docs/ALGORITHM_INTEGRATION_TIERS.md` referenced by earlier versions of this
> cross-index did not exist in the repository. Their content has been consolidated
> into `docs/PIPELINE_EVIDENCE_REGISTER.md` (Parts B–C for algorithms, Part D for
> integration parameters). The table below retargets all entries to the register.

| Algorithm / topic | Register section | Primary refs |
|-------------------|-----------------|--------------|
| TESS — temporal energy selective scaling | Part C (not implemented; literature survey only) | [1] |
| δ-GLMB / TBD thermal | Part C (not implemented) | [2] |
| IRSDT — CNN + KCF + Kalman | Part C (not implemented) | [3], [10], [11] |
| BP-FPN — gradient-driven FPN | Part C (not implemented) | [4] |
| NeurSTT — neural spatio-temporal tensor | Part C (not implemented) | [5] |
| Swin denoise / U-Net style IR | Part C (not implemented on RPi4) | [6], [7] |
| MP-PCA / NORDIC | Part C (not implemented) | [8] |
| ViBe background subtraction | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.6 (Kalman preferred; ViBe cited as full alternative) | [9] |
| Kalman filter (thermal background) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.6, §D.7 | [11], [73] |
| NIR night enhancement (dark/bright channel) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.1, §B.3 Bucket A | [12]–[14], [18] |
| CLAHE per bucket | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.2, §D.5 | [15], [39], [40], [57], [58], [41], [68] |
| DCP dehaze (Bucket D) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.4, §B.3 Bucket D | [13], [36], [37], [38], [54], [55], [56] |
| Rain temporal median (Bucket E) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.5, §B.3 Bucket E | [48], [49], [50], [51] |
| Fusion blend (addWeighted, explicit gap vs pyramid) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.3, §B.8 | [65], [33] |
| Sparse optical flow (Lucas–Kanade) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.8, §B.5 | [59] |
| Random Forest + isotonic calibration | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.9, §B.7 | [63], [64], [66], [74] |
| MAD anomaly (0.6745 constant) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.7, §D.8 | [67] |
| ENV preset hysteresis (Schmitt-trigger) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.12, §D.10 | [61] |
| Multi-label classification (compositor background) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §C.11 | [45], [46], [47] |
| Lightweight NIR/IR enhancement surveys | Part C §C.1 (survey context) | [28]–[33] |
| NIRGB–GS (3DGS + NIR; not real-time RPi default) | Out of scope for RPi4 budget; cited for contrast | [27] |
| LLVIP dataset (`nir_channel` in `train_classifier.py`) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §D.12 (training data) | [34] |
| Landscape colorization dataset (optional) | `data/README.md` | [35] |

---

## V. Cross-index: integration tiers → references

> **Note (2026-04-25):** `docs/ALGORITHM_INTEGRATION_TIERS.md` did not exist; this
> cross-index has been retargeted to `docs/PIPELINE_EVIDENCE_REGISTER.md` Part D
> (parameters) and Part B (stage-level integration).

| Integration item | Register section | Refs |
|-----------------|-----------------|------|
| Picamera2 / libcamera — NIR capture driver | `docs/PIPELINE_EVIDENCE_REGISTER.md` §B.1, §D.11 | [22] |
| ONNX / Coral edge inference (planned, not wired) | Out of scope for current baseline | [23], [24] |
| `vcgencmd` / power + throttle metrics | `docs/PIPELINE_EVIDENCE_REGISTER.md` §B.11 (session logging) | [25] |
| OpenCV — CLAHE, LK flow, warpAffine, addWeighted | §B.3, §B.5, §B.8 throughout | [21] |
| ENV bucket dispatch (A–F) | `docs/PIPELINE_EVIDENCE_REGISTER.md` §B.3 | DOC [research_report_v4] |
| ML inference thread model | `docs/PIPELINE_EVIDENCE_REGISTER.md` §A.2, §B.7 | Engineering decision |

---

## VI. Notes

- **IEEE spelling:** journal titles abbreviated per IEEE convention in formal submissions; this list uses full journal names for clarity in an internal repo — shorten for camera-ready IEEE papers.  
- **arXiv:** Replace “See arXiv PDF for author list” with full author names from the PDF when preparing the final bibliography.  
- **[12] vs [18]:** [12] is the **peer-reviewed paper**; [18] is the **tutorial** that walks through implementation — cite both when describing NIR night enhancement aligned with `hybrid_night_vision` / dark–bright pipeline narratives.  
- **[27] vs [12], [28]–[32]:** [27] is **3DGS-based scene reconstruction** (high compute); [28]–[32] support **classical / low-complexity** enhancement suitable for embedded exploration — do not conflate with the ≤50 ms/frame HUD path without profiling.  
- **[35]:** *Kaggle* repository (contributor username, not a formal author list); cite URL + access date for reproducibility.  
- **[13], [36], [37]:** Dark channel prior (DCP) — [36] CVPR 2009; [13] CVPR 2011 listing in this file; [37] *IEEE TPAMI* extended journal version; pick one venue per citation need and avoid triple-counting the same claim.  
- **[66]–[77]:** Mapped from `docs/research_report_20260421_thesis_improvement.md` — calibration / IQA / imbalance / benchmarks; see **§IX** for topic routing.

---

## VII. Cross-index: weekly report & chat topics (Apr. 2026) → references

| Topic | Refs |
|--------|------|
| NIRGB–GS (Wiley *Adv. Intell. Syst.*), IMX290 + 940 nm NIR LED discussion | [27] |
| “Deep research” lightweight NIR: CLAHE, RGF, FPGA IR, Retinex surveys | [28]–[32] |
| IR–visible fusion (multiscale + sparse low-rank) | [33] |
| Grayscale / NIR dataset channel (`nir` vs `rgb`), domain gap | [34] |
| Weekly narrative + pipeline metrics (see `weekly_report/week_3.md`) | [27], [69]–[70], [77]; thesis plan (archive): `legacy/md/THESIS_IMPROVEMENT_PLAN.md` |

---

## VIII. Cross-index: `docs/research_report_20260415_env_policy_branching_optical_v4.md` → references

| Topic (report) | Refs |
|----------------|------|
| Dark channel prior / haze (DCP), He CVPR 2009 & TPAMI 2011 | [36], [37]; see also [13] (CVPR 2011 venue in prior list) |
| Fast visibility / alternative dehaze | [38] |
| CLAHE (optimal, real-time VLSI, sub-block HE) | [39]–[41]; foundational CLAHE [15]; medical CLAHE [57]; generalized hist. eq. [58] |
| Glare / high dynamic range, tone mapping | [42], [43]; HDR book [44] |
| Dehaze variants (single-image, CAP, non-local, benchmarks) | [52]–[56] |
| Rain streaks / rain-in-video survey | [48], [49]; layer separation [50]; heavy rain [51] |
| Multi-label classification, classifier chains, surveys | [45]–[47] |
| Random Forest, scikit-learn | [63], [64] |
| Lucas–Kanade (registration; report timing context) | [59] |
| Kalman filter | [11] |
| Schmitt trigger (historical; hysteresis analogy) | [61] |
| BPTT / temporal recurrence | [62] |
| Keypoint / randomized trees | [60] |
| Image fusion (NPAR) | [65] |

**Note:** Report Appendix A also cites “Nayar 2004” for glare — the formal bibliography uses Nayar & Branzoi ICCV 2003 [42]. Reinhard *et al.* TOG 2002 [43] vs photographic tone [12]/[18] narrative: use [43]–[44] for HDR pipeline citations.

---

## IX. Cross-index: `docs/research_report_20260421_thesis_improvement.md` → references

| Topic (report) | Refs |
|----------------|------|
| RF / classifier probability calibration, good probabilities | [66], [74], [63]–[64] |
| Expected calibration error (ECE), reliability diagrams (methodology; train-time) | [66], [74] |
| MAD / median absolute deviation, outlier detection | [67] |
| CLAHE / adaptive histogram equalization (historical AHE paper) | [68]; applied CLAHE practice [15], [39]–[41] |
| No-reference IQA: BRISQUE, NIQE | [69], [70] |
| Class imbalance, SMOTE | [71] |
| Multispectral / KAIST pedestrian benchmark | [72] |
| Kalman filter tutorial (background vs. classical filter [11]) | [73], [11] |
| IR–visible fusion survey | [77]; multiscale fusion [33] |
| DCP / dehaze (codebase buckets) | [36]–[37], [13] |
| RPi OpenCV performance notes | [75] |
| IQA tooling (PyTorch) | [76] |

**Note:** The thesis improvement report’s internal bibliography uses APA-style numbering; this file maps the same sources to IEEE-style [66]–[77]. For ICDM “Calibrating Random Forests” (IEEE Xplore 4724964) cited in the report narrative, treat as supplementary to [66] and sklearn [74].

---

*This file supersedes the prior APA-style table. Update [20] when the Vandone (2011) primary citation is identified.*
