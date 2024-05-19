import numpy as np
# import matplotlib.pyplot as plt

#data:(9608,1)
def SSA(series, level):
	# series = 0
	# series = series - np.mean(series)
	original_mean = np.mean(series)
	series = series - original_mean


	windowLen = level
	seriesLen = len(series)
	K = seriesLen - windowLen + 1
	series = series.flatten()
	X = np.zeros((windowLen, K))
	for i in range(K):
		X[:, i] = series[i:i + windowLen]

	U, sigma, VT = np.linalg.svd(X, full_matrices=False)

	for i in range(VT.shape[0]):
		VT[i, :] *= sigma[i]
	A = VT


	rec = np.zeros((windowLen, seriesLen))
	for i in range(windowLen):
		for j in range(windowLen - 1):
			for m in range(j + 1):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= (j + 1)
		for j in range(windowLen - 1, seriesLen - windowLen + 1):
			for m in range(windowLen):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= windowLen
		for j in range(seriesLen - windowLen + 1, seriesLen):
			for m in range(j - seriesLen + windowLen, windowLen):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= (seriesLen - j)
	# for i in range(windowLen):
	# 	rec[i, :] += original_mean
	return rec


# rrr = np.sum(rec, axis=0)
#
# plt.figure()
# for i in range(10):
# 	ax = plt.subplot(5, 2, i + 1)
# 	ax.plot(rec[i, :])
#
# plt.figure(2)
# plt.plot(series)
# plt.show()