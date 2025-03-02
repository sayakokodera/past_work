"""
test run script for GCNR validity
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from image_quality_analyzer import ImageQualityAnalyzerGCNR

#plt.close('all')
# load data
cimg_ref = np.load('npy_data/ESE/grid/cimg_max_05_oa_0.npy')
fpath = 'H:/2018_Sayako_Kodera_BA_Daten/npy_data/ESE/pos_scan/dim_4_posdef_0/dr'
cimg_np3_005_3 = np.load('{}/dx_np_3/cimg_max_sigma_005_3.npy'.format(fpath))
cimg_np4_002_3 = np.load('{}/dx_np_4/cimg_max_sigma_002_3.npy'.format(fpath))

# =============================================================================
# # plot the data
# plt.figure(1)
# plt.imshow(cimg_ref)
# plt.title('Cimg : reference')
# 
# plt.figure(2)
# plt.imshow(cimg_np3_005_3)
# plt.title('Cimg : Np 3 sigma 005 sam 3')
# 
# plt.figure(3)
# plt.imshow(cimg_np3_200_3)
# plt.title('Cimg : Np 2 sigma 200 sam 3')
# =============================================================================


# ROI setting (here within 3dB of the peak in the cimg_ref) & reference data
defect_map = np.array([[10, 9], [16, 18], [20, 23], [29, 32]])
ref_analyzer = ImageQualityAnalyzerGCNR(cimg_ref)
ref_analyzer.set_roi(cimg_ref, defect_map, 0.5)
roi = ref_analyzer.roi


# =============================================================================
# roi=np.zeros(roi.size())
# roi[10,9] = 1
# roi[16,18] = 1
# roi[20,23] = 1
# roi[29,32] = 1
# =============================================================================

roi[8:13,7:12] = 1
roi[14:19,16:21] = 1
roi[18:23,21:26] = 1
roi[27:32,30:35] = 1

gcnr_ref = ref_analyzer.get_gcnr()
print('Reference')
print('GCNR : {}'.format(gcnr_ref))
print('Pixels inside the ROI : {}'.format(ref_analyzer.pdf_inside))
print('Number of pixels inside the ROI : {}'.format(len(ref_analyzer.pdf_inside)))
print('Number of pixels outside the ROI : {}'.format(len(ref_analyzer.pdf_outside)))
print('Largest value outside the ROI : {}'.format(ref_analyzer.pdf_outside[-1]))
print('Epsilon : {}'.format(ref_analyzer.epsilon))
print('Number of pixels missed : {}'.format(ref_analyzer.pix_missed))
print('Number of pixels falsely detected : {}'.format(ref_analyzer.pix_false))
print('Variance outside of the ROI : {}'.format(ref_analyzer.width_outside))


# Np 3 sogma 005 sam 3
analyzer = ImageQualityAnalyzerGCNR(cimg_np3_005_3, roi)
gcnr = analyzer.get_gcnr()
print('Np 3 sigma 005 sam 3')
print('GCNR : {}'.format(gcnr))
print('Pixels inside the ROI : {}'.format(analyzer.pdf_inside))
print('Number of pixels inside the ROI : {}'.format(len(analyzer.pdf_inside)))
print('Number of pixels outside the ROI : {}'.format(len(analyzer.pdf_outside)))
print('Largest value outside the ROI : {}'.format(analyzer.pdf_outside[-1]))
print('Epsilon : {}'.format(analyzer.epsilon))
print('Number of pixels missed : {}'.format(analyzer.pix_missed))
print('Number of pixels falsely detected : {}'.format(analyzer.pix_false))
print('Variance outside of the ROI : {}'.format(analyzer.width_outside))


# Np 3 sigma 200 sam 3
analyzer2 = ImageQualityAnalyzerGCNR(cimg_np4_002_3, roi)
gcnr2 = analyzer2.get_gcnr()
print('Np 4 sigma 002 sam 3')
print('GCNR : {}'.format(gcnr2))
print('Pixels inside the ROI : {}'.format(analyzer2.pdf_inside))
print('Number of pixels inside the ROI : {}'.format(len(analyzer2.pdf_inside)))
print('Number of pixels outside the ROI : {}'.format(len(analyzer2.pdf_outside)))
print('Largest value outside the ROI : {}'.format(analyzer2.pdf_outside[-1]))
print('Epsilon : {}'.format(analyzer2.epsilon))
print('Number of pixels missed : {}'.format(analyzer2.pix_missed))
print('Number of pixels falsely detected : {}'.format(analyzer2.pix_false))
print('Variance outside of the ROI : {}'.format(analyzer2.width_outside))

plt.figure(4)
n3, bins3, _ = plt.hist(ref_analyzer.pdf_outside, 50)
n4, bins4, _ = plt.hist(ref_analyzer.pdf_inside, 50)
plt.title('Histogram : reference data')
plt.legend(['Outside', 'Inside'])
plt.xlabel('Signal Energy')
plt.ylabel('Number of Pixels')

plt.show()


plt.figure(5)
n1, bins1, _ = plt.hist(analyzer.pdf_outside, 50)
n2, bins2, _ = plt.hist(analyzer.pdf_inside, 50)
plt.title('Histogram : Np 3 sigma 005 sam 3')
plt.legend(['Outside', 'Inside'])
plt.xlabel('Signal Energy')
plt.ylabel('Number of Pixels')

plt.show()


plt.figure(6)
n3, bins3, _ = plt.hist(analyzer2.pdf_outside, 50)
n4, bins4, _ = plt.hist(analyzer2.pdf_inside, 50)
plt.title('Histogram : Np 4 sigma 002 sam 3')
plt.legend(['Outside', 'Inside'])
plt.xlabel('Signal Energy')
plt.ylabel('Number of Pixels')

plt.show()