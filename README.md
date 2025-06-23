# TFM_GonzaloVargasTendero

## Descripción

**Motor Design Data Driven** es una herramienta de prediseño inteligente de motores síncronos de imán permanente (PMSM) orientada a aplicaciones aeronáuticas. Integra simulaciones FEA, modelos predictivos basados en IA y algoritmos metaheurísticos para:

- **Optimizar** densidad de potencia, eficiencia energética y peso de motores eléctricos.  
- **Automatizar** el flujo de trabajo desde la generación de datos hasta la selección de candidatos óptimos.  
- **Reducir** drásticamente el coste computacional frente a métodos tradicionales de diseño iterativo.  

## Estructura de carpetas
```
TFM_GonzaloVargasTendero/
├─ requirements.txt
├─ README.md
├─ 1.EDA/
│ ├─ EDA_V1.ipynb
│ ├─ DB_EDA/
│ └─ Figuras_EDA/
├─ 2.ML/
│ ├─ ML_V7.ipynb
│ ├─ DB_ML/
│ ├─ Figuras_ML/
│ └─ Modelos_ML/
├─ 3.MOP/
│ ├─ MOP_V13.ipynb
│ ├─ DB_MOP/
│ ├─ Figuras_MOP/
│ └─ Modelos_MOP/
├─ 4.DBG/
│ ├─ DBG_V5.ipynb
│ ├─ DB_DBG/
│ ├─ Figuras_DBG/
│ └─ Modelos_DBG/
```

- Cada subcarpeta `DB_*` contiene los CSV de entrada.  
- Cada subcarpeta `Figuras_*` almacena gráficos generados.  
- Cada subcarpeta `Modelos_*` persiste los pipelines y modelos entrenados.  
- El archivo `hiperparametros_MOP.json` define la búsqueda de hiperparámetros para MOP.

## Notebooks y flujo de ejecución

El proceso total consta de **4 notebooks** que deben ejecutarse en orden:

1. **`1.EDA/EDA_V1.ipynb`**  
   - **Propósito:** Limpieza y análisis exploratorio del dataset FEA.  
   - **Funcionalidades:**  
     1. Carga y limpieza de datos.  
     2. Estadística descriptiva de variables de diseño y rendimiento.  
     3. Visualizaciones (histogramas, KDE).  
     4. Mapas de calor de correlación para orientar selección de variables. :contentReference[oaicite:5]{index=5}

2. **`2.ML/ML_V7.ipynb`**  
   - **Propósito:** Desarrollo y validación de metamodelos predictivos.  
   - **Funcionalidades:**  
     1. Partición de datos (`train/test`).  
     2. Preprocesado (escalado, pipelines).  
     3. Definición de 7 regresores (PLS, LR, GPR, SVR, RF, ANN, ANN-K).  
     4. Evaluación inicial mediante validación cruzada y cálculo de R², MSE. :contentReference[oaicite:6]{index=6}

3. **`3.MOP/MOP_V13.ipynb`**  
   - **Propósito:** Afinamiento de hiperparámetros y ensamblaje del modelo unificado.  
   - **Funcionalidades:**  
     1. Carga de resultados de `GridSearchCV` y `BayesSearchCV`.  
     2. Reconstrucción de pipelines con parámetros óptimos.  
     3. Creación del `UnifiedDescaledRegressor` que agrupa todas las salidas.  
     4. Persistencia en disco del modelo desescalado listo para optimización. :contentReference[oaicite:7]{index=7}

4. **`4.DBG/DBG_V5.ipynb`**  
   - **Propósito:** Orquestador maestro del flujo completo.  
   - **Funcionalidades:**  
     1. Ejecución secuencial de los notebooks EDA, ML y MOP.  
     2. Gestión de artefactos intermedios (datos limpios, hiperparámetros, modelo).  
     3. Generación de la **BD final** (10 000 diseños) para optimización multiobjetivo.  
     4. Selección del mejor candidato según criterios definidos. :contentReference[oaicite:8]{index=8}

## Orden de ejecución

    1. **EDA_V1.ipynb** → 2. **ML_V7.ipynb** → 3. **MOP_V13.ipynb** → 4. **DBG_V5.ipynb**


