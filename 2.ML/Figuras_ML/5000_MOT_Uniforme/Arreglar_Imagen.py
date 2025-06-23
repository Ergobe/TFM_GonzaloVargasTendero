import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
modelos = ['PLS', 'LR', 'GPR', 'SVR', 'RF', 'ANN', 'ANN-K']
r2_scores = [0.8231, 0.8232, 0.9811, 0.9838, 0.8771, 0.9855, 0.9837]
mse_scores = [0.1769, 0.1768, 0.0189, 0.0162, 0.1229, 0.0145, 0.0163]

x = np.arange(len(modelos))

# Crear figura y ejes
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de R2
axs[0].bar(x, r2_scores, color='skyblue')
axs[0].set_xticks(x)
axs[0].set_xticklabels(modelos)
axs[0].set_ylim(0, 1.05)
axs[0].set_title('R2 Promedio (Validación Cruzada)')
axs[0].set_xlabel('Modelo')
axs[0].set_ylabel('R2 Promedio')
for i, v in enumerate(r2_scores):
    axs[0].text(i, v + 0.005, f"{v:.3f}", ha='center')

# Gráfico de MSE
axs[1].bar(x, mse_scores, color='salmon')
axs[1].set_xticks(x)
axs[1].set_xticklabels(modelos)
axs[1].set_title('Resumen con las métricas promedio')
axs[1].set_xlabel('Modelo')
axs[1].set_ylabel('MSE Promedio')
for i, v in enumerate(mse_scores):
    axs[1].text(i, v + 0.0005, f"{v:.3f}", ha='center')

# Ajustar el layout
plt.tight_layout()

# También podrías guardar la imagen sin cortar contenido:
plt.savefig("metricas_promedio_hiper.png", dpi = 1080, bbox_inches='tight')

# Mostrar la imagen
plt.show()

