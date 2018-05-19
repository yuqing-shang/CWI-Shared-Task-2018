import matplotlib.pyplot as plt

X = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
engdev = [0.65,0.69,0.68,0.70,0.71,0.69,0.69,0.70,0.71,0.73]
engtst = [0.65,0.68,0.67,0.69,0.69,0.67,0.66,0.68,0.69,0.70]
spsdev = [0.69,0.69,0.70,0.70,0.71,0.71,0.71,0.71,0.72,0.72]
spstst = [0.67,0.67,0.66,0.68,0.68,0.69,0.68,0.67,0.70,0.72]

plt.figure(figsize=(100, 100))
plt.subplot(221)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(X, engdev, color="red", linewidth=2.0, linestyle="--", marker="D")
plt.xlabel('TrainSet size', fontsize=16)
plt.ylabel('macro-F1', fontsize=16)
plt.title("engdev", fontsize=16)

plt.subplot(222)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(X, engtst, color="red", linewidth=2.0, linestyle="--", marker="D")
plt.xlabel('TrainSet size', fontsize=16)
plt.ylabel('macro-F1', fontsize=16)
plt.title("engtst", fontsize=16)

plt.subplot(223)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(X, spsdev, color="red", linewidth=2.0, linestyle="--", marker="D")
plt.xlabel('TrainSet size', fontsize=16)
plt.ylabel('macro-F1', fontsize=16)
plt.title("spsdev", fontsize=16)

plt.subplot(224)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot(X, spstst, color="red", linewidth=2.0, linestyle="--", marker="D")
plt.xlabel('TrainSet size', fontsize=16)
plt.ylabel('macro-F1', fontsize=16)
plt.title("spstst", fontsize=16)

plt.show()