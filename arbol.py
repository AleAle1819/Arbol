# decision_tree_spam_ham_repeated.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


# CONFIGURACIÓN
CSV_PATH = "dataset_correos.csv"
LABEL_COL = "Clasificación"             
N_RUNS = 50                     
TEST_SIZE = 0.30                
APPLY_ZSCORE = True

# CARGA Y PREPROCESO BASE
df = pd.read_csv(CSV_PATH)

# Separar X, y
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL]

# ('spam'/'ham') o 0/1
if y.dtype.kind in "biufc":  # numéricas
    # asumimos 1 = spam, 0 = ham
    y_bin = (y.astype(int)).values
    pos_label = 1
else:
    # normalizamos a minúsculas por si acaso
    y = y.astype(str).str.lower()
    # si no es exactamente 'spam'/'ham', intenta mapear
    if set(y.unique()) - {"spam", "ham"}:
        # Intento: primera clase encontrada = ham, segunda = spam
        classes_sorted = sorted(y.unique())
        mapping = {classes_sorted[0]: "ham", classes_sorted[-1]: "spam"}
        y = y.map(mapping)
    y_bin = y.values
    pos_label = "spam"

# Detectar columnas por tipo
num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler() if APPLY_ZSCORE else "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

# Clasificador: Árbol de decisión 
clf = DecisionTreeClassifier(
    criterion="entropy",     
    max_depth=None,          
    random_state=0           
)

# Pipeline completo: preprocesa dentro de cada split 
pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", clf)
])


accs, f1s = [], []
splits_info = []

for i in range(1, N_RUNS + 1):
    # StratifiedShuffleSplit con random_state distinto en cada ejecución para cambiar el 70%
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=i)
    train_idx, test_idx = next(sss.split(X, y_bin))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_bin[train_idx], y_bin[test_idx]

    # Entrenar y predecir
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    # si las etiquetas son strings 'spam'/'ham', pos_label='spam'; si son 0/1, pos_label=1
    f1 = f1_score(y_test, y_pred, pos_label=pos_label)

    accs.append(acc)
    f1s.append(f1)
    splits_info.append({
        "run": i,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "accuracy": acc,
        "f1": f1
    })


accs = np.array(accs)
f1s = np.array(f1s)
best_idx = int(np.argmax(f1s))
worst_idx = int(np.argmin(f1s))

print("============================================")
print(f"Modelo: Árbol de Decisión | Runs: {N_RUNS} | Train=70% / Test=30%")
print(f"Z-score aplicado a numéricas: {APPLY_ZSCORE}")
print("============================================")
print(f"Accuracy  -> media: {accs.mean():.4f} | desvío: {accs.std(ddof=1):.4f} | min: {accs.min():.4f} | max: {accs.max():.4f}")
print(f"F1-score  -> media: {f1s.mean():.4f} | desvío: {f1s.std(ddof=1):.4f} | min: {f1s.min():.4f} | max: {f1s.max():.4f}")

print("\n-- Mejor ejecución (por F1) --")
print(splits_info[best_idx])

print("\n-- Peor ejecución (por F1) --")
print(splits_info[worst_idx])


pd.DataFrame(splits_info).to_csv("resultados_runs.csv", index=False)
print("\nArchivo 'resultados_runs.csv' guardado con el detalle por ejecución.")




runs = np.arange(1, len(accs)+1)

# Curva por ejecución
plt.figure(figsize=(10,4))
plt.plot(runs, accs, marker="o")
plt.title("Exactitud por ejecución (orden temporal)")
plt.xlabel("Ejecución")
plt.ylabel("Exactitud")
plt.grid(True)
plt.tight_layout()
plt.show()

# Histograma
plt.figure(figsize=(6,4))
plt.hist(accs, bins=10)
plt.title("Distribución de exactitud")
plt.xlabel("Exactitud")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()