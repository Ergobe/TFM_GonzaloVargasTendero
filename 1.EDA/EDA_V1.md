# *Motor Design Data Driven*

## 1. Análisis EDA

### 1.1. Librerías


```python
# Librerías necesarias
import os
import re  # Import the regular expression module

import pandas as pd
import numpy as np
import math

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
```

### 1.2. Lectura de fichero


```python
# Definir las rutas base y de las carpetas
base_path = os.getcwd()  # Se asume que el notebook se ejecuta desde la carpeta 'EDA'
db_path = os.path.join(base_path, "DB_EDA")
fig_path = os.path.join(base_path, "Figuras_EDA")

# Ruta al archivo de la base de datos
data_file = os.path.join(db_path, "design_DB_5000_Uniforme.csv")
print(data_file)

# Ruta al archivo de las figuras
figure_path = os.path.join(fig_path, "5000_MOT_Uniforme")
```

    C:\Users\s00244\Documents\GitHub\MotorDesignDataDriven\Notebooks_TFM\1.EDA\DB_EDA\design_DB_5000_Uniforme.csv
    


```python
# Lectura del archivo CSV
try:
    df = pd.read_csv(data_file)
    print("Archivo cargado exitosamente.")
except FileNotFoundError:
    print("Error: Archivo no encontrado. Revisa la ruta del archivo.")
except pd.errors.ParserError:
    print("Error: Problema al analizar el archivo CSV. Revisa el formato del archivo.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")
```

    Archivo cargado exitosamente.
    

### 1.3. Exploración inicial de datos


```python
# Exploración inicial de datos

# Mostrar las primeras filas del DataFrame
print("\nPrimeras filas del DataFrame:")
display(df.head())
```

    
    Primeras filas del DataFrame:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1::OSD</th>
      <th>x2::Dint</th>
      <th>x3::L</th>
      <th>x4::tm</th>
      <th>x5::hs2</th>
      <th>x6::wt</th>
      <th>x7::Nt</th>
      <th>x8::Nh</th>
      <th>m1::Drot</th>
      <th>m2::Dsh</th>
      <th>...</th>
      <th>p2::Tnom</th>
      <th>p3::nnom</th>
      <th>p4::GFF</th>
      <th>p5::BSP_T</th>
      <th>p6::BSP_n</th>
      <th>p7::BSP_Pm</th>
      <th>p8::BSP_Mu</th>
      <th>p9::BSP_Irms</th>
      <th>p10::MSP_n</th>
      <th>p11::UWP_Mu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48.60</td>
      <td>27.8640</td>
      <td>14.800000</td>
      <td>2.780311</td>
      <td>6.312467</td>
      <td>4.392325</td>
      <td>6</td>
      <td>4</td>
      <td>26.8640</td>
      <td>13.342235</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>40.082719</td>
      <td>0.170606</td>
      <td>17113.2350</td>
      <td>305.74251</td>
      <td>90.763857</td>
      <td>10.070335</td>
      <td>18223.3200</td>
      <td>86.138152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54.60</td>
      <td>23.1040</td>
      <td>32.800001</td>
      <td>3.080830</td>
      <td>11.833245</td>
      <td>2.379534</td>
      <td>18</td>
      <td>5</td>
      <td>22.1040</td>
      <td>9.341198</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>49.664102</td>
      <td>0.990486</td>
      <td>2684.3461</td>
      <td>278.42958</td>
      <td>79.546525</td>
      <td>12.589184</td>
      <td>3576.9857</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59.40</td>
      <td>24.0560</td>
      <td>29.200001</td>
      <td>2.121244</td>
      <td>10.249868</td>
      <td>2.569301</td>
      <td>12</td>
      <td>3</td>
      <td>23.0560</td>
      <td>11.940368</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>24.675780</td>
      <td>0.412852</td>
      <td>4913.5479</td>
      <td>212.43125</td>
      <td>87.076820</td>
      <td>7.558136</td>
      <td>5737.1407</td>
      <td>88.799881</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54.72</td>
      <td>32.0528</td>
      <td>22.960001</td>
      <td>2.456926</td>
      <td>7.797124</td>
      <td>2.123813</td>
      <td>18</td>
      <td>3</td>
      <td>31.0528</td>
      <td>16.981004</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>42.652370</td>
      <td>0.538189</td>
      <td>3806.5372</td>
      <td>214.53262</td>
      <td>83.929471</td>
      <td>7.553457</td>
      <td>4325.1237</td>
      <td>83.402341</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48.84</td>
      <td>21.9616</td>
      <td>25.120000</td>
      <td>3.032073</td>
      <td>6.972909</td>
      <td>2.557345</td>
      <td>14</td>
      <td>3</td>
      <td>20.9616</td>
      <td>8.622712</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>57.017278</td>
      <td>0.380920</td>
      <td>5161.0967</td>
      <td>205.87507</td>
      <td>87.040314</td>
      <td>7.554095</td>
      <td>6293.4336</td>
      <td>91.343493</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



```python
# Información general del DataFrame
print("\nInformación general del DataFrame:")
df.info()
```

    
    Información general del DataFrame:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5242 entries, 0 to 5241
    Data columns (total 25 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   x1::OSD       5242 non-null   float64
     1   x2::Dint      5242 non-null   float64
     2   x3::L         5242 non-null   float64
     3   x4::tm        5242 non-null   float64
     4   x5::hs2       5242 non-null   float64
     5   x6::wt        5242 non-null   float64
     6   x7::Nt        5242 non-null   int64  
     7   x8::Nh        5242 non-null   int64  
     8   m1::Drot      5242 non-null   float64
     9   m2::Dsh       5242 non-null   float64
     10  m3::he        5242 non-null   float64
     11  m4::Rmag      5242 non-null   float64
     12  m5::Rs        5242 non-null   float64
     13  m6::GFF       5242 non-null   float64
     14  p1::W         4447 non-null   float64
     15  p2::Tnom      5242 non-null   float64
     16  p3::nnom      5242 non-null   float64
     17  p4::GFF       4447 non-null   float64
     18  p5::BSP_T     4447 non-null   float64
     19  p6::BSP_n     4447 non-null   float64
     20  p7::BSP_Pm    4447 non-null   float64
     21  p8::BSP_Mu    4447 non-null   float64
     22  p9::BSP_Irms  4447 non-null   float64
     23  p10::MSP_n    4447 non-null   float64
     24  p11::UWP_Mu   3761 non-null   float64
    dtypes: float64(23), int64(2)
    memory usage: 1024.0 KB
    


```python
# Estadísticas descriptivas del DataFrame
print("\nEstadísticas descriptivas:")
display(df.describe())
```

    
    Estadísticas descriptivas:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1::OSD</th>
      <th>x2::Dint</th>
      <th>x3::L</th>
      <th>x4::tm</th>
      <th>x5::hs2</th>
      <th>x6::wt</th>
      <th>x7::Nt</th>
      <th>x8::Nh</th>
      <th>m1::Drot</th>
      <th>m2::Dsh</th>
      <th>...</th>
      <th>p2::Tnom</th>
      <th>p3::nnom</th>
      <th>p4::GFF</th>
      <th>p5::BSP_T</th>
      <th>p6::BSP_n</th>
      <th>p7::BSP_Pm</th>
      <th>p8::BSP_Mu</th>
      <th>p9::BSP_Irms</th>
      <th>p10::MSP_n</th>
      <th>p11::UWP_Mu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>5242.000000</td>
      <td>...</td>
      <td>5242.00</td>
      <td>5242.0</td>
      <td>4447.000000</td>
      <td>4447.000000</td>
      <td>4447.000000</td>
      <td>4447.000000</td>
      <td>4447.000000</td>
      <td>4447.000000</td>
      <td>4447.000000</td>
      <td>3761.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>55.847039</td>
      <td>27.152434</td>
      <td>24.947683</td>
      <td>2.734945</td>
      <td>8.682188</td>
      <td>3.309290</td>
      <td>10.793781</td>
      <td>5.039107</td>
      <td>26.152434</td>
      <td>12.924705</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>43.437674</td>
      <td>0.540111</td>
      <td>8235.127695</td>
      <td>346.133033</td>
      <td>87.690220</td>
      <td>12.627435</td>
      <td>9521.799554</td>
      <td>88.320642</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.439244</td>
      <td>4.358982</td>
      <td>8.751453</td>
      <td>0.431730</td>
      <td>2.287659</td>
      <td>0.840682</td>
      <td>5.317152</td>
      <td>1.848228</td>
      <td>4.358982</td>
      <td>3.186535</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>11.062318</td>
      <td>0.294003</td>
      <td>5658.244776</td>
      <td>131.417512</td>
      <td>4.343265</td>
      <td>4.653808</td>
      <td>6145.426096</td>
      <td>2.934205</td>
    </tr>
    <tr>
      <th>min</th>
      <td>45.000960</td>
      <td>21.204387</td>
      <td>10.000384</td>
      <td>2.000021</td>
      <td>5.006201</td>
      <td>2.000405</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>20.204387</td>
      <td>8.007388</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>20.937265</td>
      <td>0.054076</td>
      <td>761.280320</td>
      <td>127.215630</td>
      <td>65.162410</td>
      <td>7.534755</td>
      <td>1171.984600</td>
      <td>70.238550</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>53.792861</td>
      <td>23.525393</td>
      <td>17.302182</td>
      <td>2.361072</td>
      <td>6.794815</td>
      <td>2.588997</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>22.525393</td>
      <td>10.385444</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>34.321520</td>
      <td>0.321027</td>
      <td>4266.651600</td>
      <td>227.242150</td>
      <td>86.019082</td>
      <td>7.554244</td>
      <td>5222.421750</td>
      <td>86.861221</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>56.622163</td>
      <td>26.299126</td>
      <td>24.934452</td>
      <td>2.732837</td>
      <td>8.460640</td>
      <td>3.251083</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>25.299126</td>
      <td>12.318815</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>44.167043</td>
      <td>0.477237</td>
      <td>6815.689000</td>
      <td>307.699100</td>
      <td>89.006020</td>
      <td>12.577613</td>
      <td>7998.616600</td>
      <td>89.029744</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>58.712525</td>
      <td>29.916908</td>
      <td>32.475884</td>
      <td>3.111981</td>
      <td>10.328053</td>
      <td>3.987706</td>
      <td>13.000000</td>
      <td>6.000000</td>
      <td>28.916908</td>
      <td>14.972997</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>52.863005</td>
      <td>0.703906</td>
      <td>10557.782500</td>
      <td>442.864210</td>
      <td>90.657291</td>
      <td>15.107079</td>
      <td>12100.442500</td>
      <td>90.438868</td>
    </tr>
    <tr>
      <th>max</th>
      <td>59.999232</td>
      <td>42.026714</td>
      <td>39.998003</td>
      <td>3.499768</td>
      <td>14.946449</td>
      <td>4.998505</td>
      <td>30.000000</td>
      <td>9.000000</td>
      <td>41.026714</td>
      <td>24.794923</td>
      <td>...</td>
      <td>0.11</td>
      <td>3960.0</td>
      <td>66.633388</td>
      <td>2.012437</td>
      <td>38941.723000</td>
      <td>706.812390</td>
      <td>93.531236</td>
      <td>22.702016</td>
      <td>43867.109000</td>
      <td>92.893227</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>


### 1.4. Visualización de los datos


```python
# Estilo visual
sns.set(style='whitegrid')

# Agrupar columnas por prefijo
x_cols = [col for col in df.columns if col.startswith('x') and df[col].dtype in ['float64', 'int64']]
m_cols = [col for col in df.columns if col.startswith('m') and df[col].dtype in ['float64', 'int64']]
p_cols = [col for col in df.columns if col.startswith('p') and df[col].dtype in ['float64', 'int64']]

def plot_variable_group(columns, group_name):
    if not columns:
        print(f"No hay variables para el grupo '{group_name}'")
        return

    n = len(columns)
    cols = 3  # número de columnas de subplots
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribución de {col}', fontsize=12)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)

    # Eliminar ejes vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Distribuciones del grupo "{group_name}"', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Guardar la figura en la carpeta 'Figuras_EDA/(La carpeta que corresponda)'
    figure_file = os.path.join(figure_path, f"Distribuciones del grupo_{group_name}.png")
    plt.savefig(figure_file, dpi =1080)
    plt.close()
    #plt.show()

# Generar subplots por grupo
plot_variable_group(x_cols, 'x')
plot_variable_group(m_cols, 'm')
plot_variable_group(p_cols, 'p')
```


```python
def plot_heatmap(subset_df, title, xlabel, ylabel):
    if subset_df.empty:
        print(f"No hay datos para {title}")
        return

    plt.figure(figsize=(max(10, 0.5 * subset_df.shape[1]), max(6, 0.4 * subset_df.shape[0])))
    sns.heatmap(subset_df, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, 
                cbar_kws={'label': 'Correlación'}, annot_kws={"size": 8})
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Guardar la figura en la carpeta 'Figuras_EDA/(La carpeta que corresponda)'
    figure_file = os.path.join(figure_path, f"{title}.png")
    plt.savefig(figure_file, dpi =1080)
    plt.close()
    #plt.show()

# Correlación p-x
if p_cols and x_cols:
    corr_px = df[p_cols + x_cols].corr().loc[p_cols, x_cols]
    plot_heatmap(corr_px, 'Mapa de calor_Variables p vs x', 'x', 'p')

# Correlación p-m
if p_cols and m_cols:
    corr_pm = df[p_cols + m_cols].corr().loc[p_cols, m_cols]
    plot_heatmap(corr_pm, 'Mapa de calor_Variables p vs m', 'm', 'p')

# Correlación p-p
if p_cols:
    corr_pp = df[p_cols].corr()
    plot_heatmap(corr_pp, 'Mapa de calor_Variables p vs p', 'p', 'p')
```

### 1.5. Preprocesado de los datos


```python
# Verificar y corregir tipos de datos incorrectos usando expresiones regulares
def correct_dtype_regex(df):
    numeric_regex = re.compile(r'^-?\d+(\.\d+)?$')
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].dropna().apply(lambda x: bool(numeric_regex.match(str(x)))).all():
                df[col] = pd.to_numeric(df[col])
                print(f"Columna '{col}' convertida a tipo numérico exitosamente usando regex.")
            else:
                print(f"Columna '{col}' no puede ser convertida directamente a numérico.")
    return df

df = correct_dtype_regex(df)
display(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5242 entries, 0 to 5241
    Data columns (total 25 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   x1::OSD       5242 non-null   float64
     1   x2::Dint      5242 non-null   float64
     2   x3::L         5242 non-null   float64
     3   x4::tm        5242 non-null   float64
     4   x5::hs2       5242 non-null   float64
     5   x6::wt        5242 non-null   float64
     6   x7::Nt        5242 non-null   int64  
     7   x8::Nh        5242 non-null   int64  
     8   m1::Drot      5242 non-null   float64
     9   m2::Dsh       5242 non-null   float64
     10  m3::he        5242 non-null   float64
     11  m4::Rmag      5242 non-null   float64
     12  m5::Rs        5242 non-null   float64
     13  m6::GFF       5242 non-null   float64
     14  p1::W         4447 non-null   float64
     15  p2::Tnom      5242 non-null   float64
     16  p3::nnom      5242 non-null   float64
     17  p4::GFF       4447 non-null   float64
     18  p5::BSP_T     4447 non-null   float64
     19  p6::BSP_n     4447 non-null   float64
     20  p7::BSP_Pm    4447 non-null   float64
     21  p8::BSP_Mu    4447 non-null   float64
     22  p9::BSP_Irms  4447 non-null   float64
     23  p10::MSP_n    4447 non-null   float64
     24  p11::UWP_Mu   3761 non-null   float64
    dtypes: float64(23), int64(2)
    memory usage: 1024.0 KB
    


    None



```python
# Optimización del uso de memoria reduciendo tamaño de tipos de datos
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype('int32')
        print(f"Columna '{col}' convertida de int64 a int32.")
    elif df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
        print(f"Columna '{col}' convertida de float64 a float32.")
display(df.info())
```

    Columna 'x1::OSD' convertida de float64 a float32.
    Columna 'x2::Dint' convertida de float64 a float32.
    Columna 'x3::L' convertida de float64 a float32.
    Columna 'x4::tm' convertida de float64 a float32.
    Columna 'x5::hs2' convertida de float64 a float32.
    Columna 'x6::wt' convertida de float64 a float32.
    Columna 'x7::Nt' convertida de int64 a int32.
    Columna 'x8::Nh' convertida de int64 a int32.
    Columna 'm1::Drot' convertida de float64 a float32.
    Columna 'm2::Dsh' convertida de float64 a float32.
    Columna 'm3::he' convertida de float64 a float32.
    Columna 'm4::Rmag' convertida de float64 a float32.
    Columna 'm5::Rs' convertida de float64 a float32.
    Columna 'm6::GFF' convertida de float64 a float32.
    Columna 'p1::W' convertida de float64 a float32.
    Columna 'p2::Tnom' convertida de float64 a float32.
    Columna 'p3::nnom' convertida de float64 a float32.
    Columna 'p4::GFF' convertida de float64 a float32.
    Columna 'p5::BSP_T' convertida de float64 a float32.
    Columna 'p6::BSP_n' convertida de float64 a float32.
    Columna 'p7::BSP_Pm' convertida de float64 a float32.
    Columna 'p8::BSP_Mu' convertida de float64 a float32.
    Columna 'p9::BSP_Irms' convertida de float64 a float32.
    Columna 'p10::MSP_n' convertida de float64 a float32.
    Columna 'p11::UWP_Mu' convertida de float64 a float32.
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5242 entries, 0 to 5241
    Data columns (total 25 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   x1::OSD       5242 non-null   float32
     1   x2::Dint      5242 non-null   float32
     2   x3::L         5242 non-null   float32
     3   x4::tm        5242 non-null   float32
     4   x5::hs2       5242 non-null   float32
     5   x6::wt        5242 non-null   float32
     6   x7::Nt        5242 non-null   int32  
     7   x8::Nh        5242 non-null   int32  
     8   m1::Drot      5242 non-null   float32
     9   m2::Dsh       5242 non-null   float32
     10  m3::he        5242 non-null   float32
     11  m4::Rmag      5242 non-null   float32
     12  m5::Rs        5242 non-null   float32
     13  m6::GFF       5242 non-null   float32
     14  p1::W         4447 non-null   float32
     15  p2::Tnom      5242 non-null   float32
     16  p3::nnom      5242 non-null   float32
     17  p4::GFF       4447 non-null   float32
     18  p5::BSP_T     4447 non-null   float32
     19  p6::BSP_n     4447 non-null   float32
     20  p7::BSP_Pm    4447 non-null   float32
     21  p8::BSP_Mu    4447 non-null   float32
     22  p9::BSP_Irms  4447 non-null   float32
     23  p10::MSP_n    4447 non-null   float32
     24  p11::UWP_Mu   3761 non-null   float32
    dtypes: float32(23), int32(2)
    memory usage: 512.0 KB
    


    None



```python
# Verificación de valores faltantes y duplicados
print("\nValores faltantes por columna:")
display(df.isnull().sum())
```

    
    Valores faltantes por columna:
    


    x1::OSD            0
    x2::Dint           0
    x3::L              0
    x4::tm             0
    x5::hs2            0
    x6::wt             0
    x7::Nt             0
    x8::Nh             0
    m1::Drot           0
    m2::Dsh            0
    m3::he             0
    m4::Rmag           0
    m5::Rs             0
    m6::GFF            0
    p1::W            795
    p2::Tnom           0
    p3::nnom           0
    p4::GFF          795
    p5::BSP_T        795
    p6::BSP_n        795
    p7::BSP_Pm       795
    p8::BSP_Mu       795
    p9::BSP_Irms     795
    p10::MSP_n       795
    p11::UWP_Mu     1481
    dtype: int64



```python
print("\nCantidad de filas duplicadas:")
display(df.duplicated().sum())
```

    
    Cantidad de filas duplicadas:
    


    np.int64(0)



```python
# Identifica las filas con valores NaN en cualquier matriz
rows_with_nan = df[df.isnull().any(axis=1)].index

# Obtiene el conjunto de todos los índices con NaN
all_nan_indices = set(rows_with_nan)
all_nan_indices = sorted(list(all_nan_indices))

# Elimina las filas con valores NaN.
df_cleaned = df.drop(index=all_nan_indices)

display(df_cleaned.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 3761 entries, 0 to 5241
    Data columns (total 25 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   x1::OSD       3761 non-null   float32
     1   x2::Dint      3761 non-null   float32
     2   x3::L         3761 non-null   float32
     3   x4::tm        3761 non-null   float32
     4   x5::hs2       3761 non-null   float32
     5   x6::wt        3761 non-null   float32
     6   x7::Nt        3761 non-null   int32  
     7   x8::Nh        3761 non-null   int32  
     8   m1::Drot      3761 non-null   float32
     9   m2::Dsh       3761 non-null   float32
     10  m3::he        3761 non-null   float32
     11  m4::Rmag      3761 non-null   float32
     12  m5::Rs        3761 non-null   float32
     13  m6::GFF       3761 non-null   float32
     14  p1::W         3761 non-null   float32
     15  p2::Tnom      3761 non-null   float32
     16  p3::nnom      3761 non-null   float32
     17  p4::GFF       3761 non-null   float32
     18  p5::BSP_T     3761 non-null   float32
     19  p6::BSP_n     3761 non-null   float32
     20  p7::BSP_Pm    3761 non-null   float32
     21  p8::BSP_Mu    3761 non-null   float32
     22  p9::BSP_Irms  3761 non-null   float32
     23  p10::MSP_n    3761 non-null   float32
     24  p11::UWP_Mu   3761 non-null   float32
    dtypes: float32(23), int32(2)
    memory usage: 396.7 KB
    


    None



```python
# Tabla de estadísticas descriptivas
print("\nTabla de estadísticas descriptivas finales:")
display(df_cleaned.describe().T)
```

    
    Tabla de estadísticas descriptivas finales:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x1::OSD</th>
      <td>3761.0</td>
      <td>55.744644</td>
      <td>3.460788</td>
      <td>45.003456</td>
      <td>53.590309</td>
      <td>56.496574</td>
      <td>58.670208</td>
      <td>59.999233</td>
    </tr>
    <tr>
      <th>x2::Dint</th>
      <td>3761.0</td>
      <td>26.694340</td>
      <td>4.074372</td>
      <td>21.204388</td>
      <td>23.338999</td>
      <td>25.801891</td>
      <td>29.232868</td>
      <td>41.778736</td>
    </tr>
    <tr>
      <th>x3::L</th>
      <td>3761.0</td>
      <td>23.823900</td>
      <td>8.494957</td>
      <td>10.000384</td>
      <td>16.565634</td>
      <td>23.265970</td>
      <td>30.747520</td>
      <td>39.998001</td>
    </tr>
    <tr>
      <th>x4::tm</th>
      <td>3761.0</td>
      <td>2.732233</td>
      <td>0.432805</td>
      <td>2.000021</td>
      <td>2.354096</td>
      <td>2.727570</td>
      <td>3.106093</td>
      <td>3.499768</td>
    </tr>
    <tr>
      <th>x5::hs2</th>
      <td>3761.0</td>
      <td>8.798923</td>
      <td>2.204605</td>
      <td>5.006847</td>
      <td>7.025496</td>
      <td>8.556318</td>
      <td>10.381292</td>
      <td>14.946449</td>
    </tr>
    <tr>
      <th>x6::wt</th>
      <td>3761.0</td>
      <td>3.304939</td>
      <td>0.833983</td>
      <td>2.000405</td>
      <td>2.591386</td>
      <td>3.249449</td>
      <td>3.977364</td>
      <td>4.998168</td>
    </tr>
    <tr>
      <th>x7::Nt</th>
      <td>3761.0</td>
      <td>9.603031</td>
      <td>4.167259</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>9.000000</td>
      <td>12.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>x8::Nh</th>
      <td>3761.0</td>
      <td>5.273597</td>
      <td>1.860813</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>7.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>m1::Drot</th>
      <td>3761.0</td>
      <td>25.694340</td>
      <td>4.074372</td>
      <td>20.204388</td>
      <td>22.338999</td>
      <td>24.801891</td>
      <td>28.232868</td>
      <td>40.778736</td>
    </tr>
    <tr>
      <th>m2::Dsh</th>
      <td>3761.0</td>
      <td>12.602922</td>
      <td>2.997488</td>
      <td>8.007388</td>
      <td>10.256752</td>
      <td>11.985586</td>
      <td>14.483794</td>
      <td>24.794922</td>
    </tr>
    <tr>
      <th>m3::he</th>
      <td>3761.0</td>
      <td>5.726226</td>
      <td>1.810047</td>
      <td>3.500325</td>
      <td>4.261017</td>
      <td>5.296264</td>
      <td>6.792925</td>
      <td>13.000177</td>
    </tr>
    <tr>
      <th>m4::Rmag</th>
      <td>3761.0</td>
      <td>12.164112</td>
      <td>2.035033</td>
      <td>9.361403</td>
      <td>10.492299</td>
      <td>11.708392</td>
      <td>13.434603</td>
      <td>19.883490</td>
    </tr>
    <tr>
      <th>m5::Rs</th>
      <td>3761.0</td>
      <td>22.146091</td>
      <td>2.109504</td>
      <td>15.755375</td>
      <td>20.641384</td>
      <td>22.277868</td>
      <td>23.764368</td>
      <td>26.461054</td>
    </tr>
    <tr>
      <th>m6::GFF</th>
      <td>3761.0</td>
      <td>37.111980</td>
      <td>9.556508</td>
      <td>20.004059</td>
      <td>29.085491</td>
      <td>37.159924</td>
      <td>45.031456</td>
      <td>54.999233</td>
    </tr>
    <tr>
      <th>p1::W</th>
      <td>3761.0</td>
      <td>0.569902</td>
      <td>0.155213</td>
      <td>0.255234</td>
      <td>0.445454</td>
      <td>0.555020</td>
      <td>0.683692</td>
      <td>1.005128</td>
    </tr>
    <tr>
      <th>p2::Tnom</th>
      <td>3761.0</td>
      <td>0.110000</td>
      <td>0.000000</td>
      <td>0.110000</td>
      <td>0.110000</td>
      <td>0.110000</td>
      <td>0.110000</td>
      <td>0.110000</td>
    </tr>
    <tr>
      <th>p3::nnom</th>
      <td>3761.0</td>
      <td>3960.000000</td>
      <td>0.000000</td>
      <td>3960.000000</td>
      <td>3960.000000</td>
      <td>3960.000000</td>
      <td>3960.000000</td>
      <td>3960.000000</td>
    </tr>
    <tr>
      <th>p4::GFF</th>
      <td>3761.0</td>
      <td>42.554726</td>
      <td>11.028567</td>
      <td>20.937265</td>
      <td>33.193748</td>
      <td>42.848991</td>
      <td>51.901443</td>
      <td>66.633385</td>
    </tr>
    <tr>
      <th>p5::BSP_T</th>
      <td>3761.0</td>
      <td>0.474520</td>
      <td>0.235081</td>
      <td>0.110104</td>
      <td>0.298206</td>
      <td>0.433830</td>
      <td>0.592823</td>
      <td>1.657419</td>
    </tr>
    <tr>
      <th>p6::BSP_n</th>
      <td>3761.0</td>
      <td>9024.212891</td>
      <td>5188.172852</td>
      <td>2347.505127</td>
      <td>5247.996582</td>
      <td>7634.479980</td>
      <td>11194.800781</td>
      <td>38941.722656</td>
    </tr>
    <tr>
      <th>p7::BSP_Pm</th>
      <td>3761.0</td>
      <td>367.535553</td>
      <td>129.422104</td>
      <td>138.840652</td>
      <td>259.329926</td>
      <td>351.486145</td>
      <td>459.191101</td>
      <td>706.812378</td>
    </tr>
    <tr>
      <th>p8::BSP_Mu</th>
      <td>3761.0</td>
      <td>88.922539</td>
      <td>2.745711</td>
      <td>73.908340</td>
      <td>87.576187</td>
      <td>89.512253</td>
      <td>90.878654</td>
      <td>93.531235</td>
    </tr>
    <tr>
      <th>p9::BSP_Irms</th>
      <td>3761.0</td>
      <td>13.267383</td>
      <td>4.689486</td>
      <td>7.534754</td>
      <td>10.066794</td>
      <td>12.587963</td>
      <td>17.621677</td>
      <td>22.702017</td>
    </tr>
    <tr>
      <th>p10::MSP_n</th>
      <td>3761.0</td>
      <td>10409.014648</td>
      <td>5660.459961</td>
      <td>3977.654297</td>
      <td>6298.291992</td>
      <td>8791.044922</td>
      <td>12816.920898</td>
      <td>43867.109375</td>
    </tr>
    <tr>
      <th>p11::UWP_Mu</th>
      <td>3761.0</td>
      <td>88.320641</td>
      <td>2.934205</td>
      <td>70.238548</td>
      <td>86.861221</td>
      <td>89.029747</td>
      <td>90.438866</td>
      <td>92.893227</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6. Almacenar el preprocesado


```python
# Guardar DataFrame preprocesado
print("\nDataFrame después del preprocesamiento:")
# Ruta al archivo de la base de datos
data_cleaned_file = os.path.join(db_path, 'design_DB_preprocessed.csv')
df_cleaned.to_csv(data_cleaned_file, index=False)
# Confirmación de preprocesamiento
print("\nPreprocesamiento completado exitosamente. Archivo 'datos_preprocesados.csv' guardado.")
```

    
    DataFrame después del preprocesamiento:
    
    Preprocesamiento completado exitosamente. Archivo 'datos_preprocesados.csv' guardado.
    

-------------------------------------------------------------------------------------------------------------------------
