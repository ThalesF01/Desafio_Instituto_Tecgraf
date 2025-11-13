# 🌦️ **Previsão Meteorológica – Desafio Técnico Tecgraf / PUC-Rio**

### 👨‍💻 **Autor: Thales Gabriel da Silva Fiscus**

Este repositório contém a solução do desafio técnico de previsão meteorológica baseado em um histórico de **10 anos de dados horários**, com o objetivo de gerar previsões de **temperatura** e **chuva** para todas as horas do próximo ano (≈ **8760 horas**).

A solução inclui:

* ✔️ **Análise exploratória dos dados**
* ✔️ **Tratamento e normalização das features**
* ✔️ **Preparação adequada da série temporal**
* ✔️ **Treinamento de dois modelos de machine learning (LightGBM)**
* ✔️ **Validação com Time Series Cross-Validation (TSCV)**
* ✔️ **Otimização de hiperparâmetros com Random Search**
* ✔️ **Geração do arquivo final `PREVISAO.csv`**
* ✔️ **Documentação completa no Jupyter Notebook**

---

# 📂 **Arquivos do Repositório**

| Arquivo                   | Descrição                                                 |
| ------------------------- | --------------------------------------------------------- |
| `HISTORICO.csv`           | Histórico de 10 anos de dados meteorológicos (fornecido). |
| `PREVISAO.csv`            | Arquivo final contendo a previsão hora a hora para 1 ano. |
| `notebook_previsao.ipynb` | Notebook completo com todos os passos do projeto.         |
| `README.md`               | Este documento.                                           |

---

# 🧠 **Visão Geral da Solução**

### ✔️ **Modelagem Separada**

Foram criados dois modelos independentes, ambos utilizando LightGBM:

* **Modelo 1 — Temperatura**
* **Modelo 2 — Chuva**

Temperatura e chuva apresentam padrões temporais e comportamentos estatísticos muito distintos, o que faz com que modelos separados entreguem resultados mais estáveis e melhor ajustados às características de cada variável.

---

# 🔍 **Processo Metodológico**

O pipeline segue boas práticas consolidadas para séries temporais.

---

## **1. Importação e Pré-processamento**

* Carregamento do arquivo `HISTORICO.csv`
* Conversão da coluna `time` para datetime
* Separação das variáveis-alvo
* Identificação das features

Essa etapa organiza os dados e garante que a estrutura temporal seja preservada antes do início da modelagem.

```python
# === CARREGAMENTO DO HISTÓRICO ===

df = pd.read_csv("HISTORICO.csv")

# Convertendo coluna time
df["time"] = pd.to_datetime(df["time"])

df.head()

# === SEPARAÇÃO DE FEATURES E TARGETS ===

target_temp = "temperature_2m (°C)"
target_rain = "rain (mm)"

feature_cols = [c for c in df.columns if c not in ["time", target_temp, target_rain]]

X = df[feature_cols].copy()
y_temp = df[target_temp].copy()
y_rain = df[target_rain].copy()

X.head()
```

---

## **2. Normalização**

* Uso do `StandardScaler`
* Normalização aplicada apenas às features
* Targets permanecem na escala original

Essa normalização melhora estabilidade numérica no treinamento e impede que variáveis muito maiores influenciem desproporcionalmente o modelo.

```python
# === NORMALIZAÇÃO DAS FEATURES ===

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

X_scaled.head()
```

---

## **3. Divisão Temporal**

* 80% dos dados para treino
* 20% para teste final

Dentro do treino:

* 80% treino interno
* 20% validação interna

A divisão mantém a ordem cronológica para evitar vazamento de dados futuros, essencial em séries temporais.

```python
# === SPLIT TEMPORAL (80% TREINO / 20% TESTE) ===

train_size = int(len(X_scaled) * 0.8)

X_train = X_scaled.iloc[:train_size]
X_test  = X_scaled.iloc[train_size:]

y_temp_train = y_temp.iloc[:train_size]
y_temp_test  = y_temp.iloc[train_size:]

y_rain_train = y_rain.iloc[:train_size]
y_rain_test  = y_rain.iloc[train_size:]


# === VALIDAÇÃO INTERNA (20% DO TREINO) ===

valid_size = int(len(X_train) * 0.2)

X_train_internal = X_train.iloc[:-valid_size]
X_valid          = X_train.iloc[-valid_size:]

y_temp_train_internal = y_temp_train.iloc[:-valid_size]
y_temp_valid          = y_temp_train.iloc[-valid_size:]

y_rain_train_internal = y_rain_train.iloc[:-valid_size]
y_rain_valid          = y_rain_train.iloc[-valid_size:]
```

---

## **4. Treinamento**

Modelos base com:

* 3000–5000 árvores
* `learning_rate` reduzido
* **early stopping**

O uso de early stopping impede overfitting, encerrando o treinamento quando a validação deixa de melhorar, resultando em modelos mais generalizáveis.

```python
# === MODELO BASE — TEMPERATURA ===

modelo_temp = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

modelo_temp.fit(
    X_train_internal,
    y_temp_train_internal,
    eval_set=[(X_valid, y_temp_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

y_pred_train = modelo_temp.predict(X_train_internal)
y_pred_valid = modelo_temp.predict(X_valid)

mae_tr, rmse_tr = avaliar(y_temp_train_internal, y_pred_train)
mae_va, rmse_va = avaliar(y_temp_valid, y_pred_valid)

print("Treino:", mae_tr, rmse_tr)
print("Validação:", mae_va, rmse_va)

# === MODELO BASE — CHUVA (RAIN) ===

modelo_rain = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

modelo_rain.fit(
    X_train_internal,
    y_rain_train_internal,
    eval_set=[(X_valid, y_rain_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

y_rain_pred_train  = modelo_rain.predict(X_train_internal)
y_rain_pred_valid  = modelo_rain.predict(X_valid)

mae_tr_rain, rmse_tr_rain = avaliar(y_rain_train_internal, y_rain_pred_train)
mae_va_rain, rmse_va_rain = avaliar(y_rain_valid, y_rain_pred_valid)

print("RAIN — Treino:")
print(f"MAE = {mae_tr_rain:.3f}   RMSE = {rmse_tr_rain:.3f}")
print("\nRAIN — Validação Interna:")
print(f"MAE = {mae_va_rain:.3f}   RMSE = {rmse_va_rain:.3f}")

```

---

## **5. Validação Cruzada Temporal (TSCV)**

* Implementação de TimeSeriesSplit
* Avaliação com 5 folds
* Métrica principal: RMSE

Essa validação verifica a estabilidade do modelo em várias janelas temporais, simulando diferentes períodos futuros e oferecendo uma avaliação mais confiável do desempenho.

```python
# === CROSS-VALIDATION TEMPORAL — TEMPERATURA ===

tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []

print("Executando TSCV...\n")

for fold, (train_idx, test_idx) in tqdm(
    enumerate(tscv.split(X_train), start=1),
    total=5
):
    X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_tr, y_te = y_temp_train.iloc[train_idx], y_temp_train.iloc[test_idx]

    modelo = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=80)]
    )

    preds = modelo.predict(X_te)
    rmse_scores.append(np.sqrt(mean_squared_error(y_te, preds)))

print("RMSE médio:", np.mean(rmse_scores))
print("Desvio:", np.std(rmse_scores))

# === CROSS-VALIDATION TEMPORAL — RAIN ===

tscv_rain = TimeSeriesSplit(n_splits=5)
rmse_scores_rain = []

print("Executando TSCV para RAIN...\n")

for fold, (train_idx, test_idx) in tqdm(
    enumerate(tscv_rain.split(X_train), start=1),
    total=5
):
    X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_tr, y_te = y_rain_train.iloc[train_idx], y_rain_train.iloc[test_idx]

    modelo = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=80)]
    )

    preds = modelo.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    rmse_scores_rain.append(rmse)

    tqdm.write(f"Fold {fold}/5 — RMSE RAIN: {rmse:.3f}")

print("\n===== RESULTADOS CV - RAIN =====")
print(f"RMSE MÉDIO: {np.mean(rmse_scores_rain):.4f}")
print(f"DESVIO:     {np.std(rmse_scores_rain):.4f}")
```

---

## **6. Otimização dos Hiperparâmetros (Random Search)**

Avaliação de combinações de:

* `num_leaves`
* `max_depth`
* `learning_rate`
* Regularização L1/L2
* `feature_fraction`
* `bagging_fraction`
* `min_data_in_leaf`

O Random Search foi utilizado para explorar múltiplas configurações do LightGBM de forma eficiente, acelerando a busca por combinações que entregassem o menor erro sem exigir uma busca exaustiva.

```python
# === RANDOM SEARCH — TEMPERATURA ===

param_dist = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_data_in_leaf": [20, 40, 60],
    "feature_fraction": [0.7, 0.8, 0.9],
    "bagging_fraction": [0.7, 0.8, 0.9],
    "bagging_freq": [1, 3],
    "max_depth": [-1, 5, 8],
    "lambda_l1": [0, 0.1, 0.3],
    "lambda_l2": [0, 0.1, 0.3],
}

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        yield
        sys.stdout = old

param_list = list(ParameterSampler(param_dist, n_iter=20, random_state=42))

best_score = float("inf")
best_params = None

for params in tqdm(param_list):
    scores = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X_train):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_temp_train.iloc[train_idx], y_temp_train.iloc[test_idx]

        model = LGBMRegressor(**params, random_state=42, n_jobs=-1, verbosity=-1)

        with suppress_stdout():
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
            )

        rmse = np.sqrt(mean_squared_error(y_te, model.predict(X_te)))
        scores.append(rmse)

    mean_rmse = np.mean(scores)

    if mean_rmse < best_score:
        best_score = mean_rmse
        best_params = params

print("Melhores hiperparâmetros:", best_params)
print("Melhor RMSE:", best_score)

# === RANDOM SEARCH — RAIN ===

param_dist_rain = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_data_in_leaf": [20, 40, 60],
    "feature_fraction": [0.7, 0.8, 0.9],
    "bagging_fraction": [0.7, 0.8, 0.9],
    "bagging_freq": [1, 3],
    "max_depth": [-1, 5, 8],
    "lambda_l1": [0, 0.1, 0.3],
    "lambda_l2": [0, 0.1, 0.3],
}

param_list_rain = list(ParameterSampler(param_dist_rain, n_iter=20, random_state=42))

best_score_rain = float("inf")
best_params_rain = None

print("Iniciando busca de hiperparâmetros para RAIN...\n")

for params in tqdm(param_list_rain, desc="Random Search RAIN"):
    fold_scores = []

    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X_train):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_rain_train.iloc[train_idx], y_rain_train.iloc[test_idx]

        model_rain = LGBMRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        with suppress_stdout():
            model_rain.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
            )

        preds = model_rain.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        fold_scores.append(rmse)

    mean_rmse = np.mean(fold_scores)

    if mean_rmse < best_score_rain:
        best_score_rain = mean_rmse
        best_params_rain = params

print("\n===== MELHORES HIPERPARÂMETROS RAIN =====")
print(best_params_rain)
print("Melhor RMSE encontrado:", best_score_rain)
```

---

## **7. Treino Final e Avaliação**

Após encontrar os melhores hiperparâmetros, foi realizado:

* treino final dos modelos
* avaliação completa em:

  * treino
  * validação interna
  * teste final

Também foram gerados gráficos de:

* série temporal (real vs previsto)
* dispersão (scatter plot)
* distribuição dos resíduos

Essas visualizações ajudam a verificar padrões, desvios sistemáticos e confiabilidade das previsões.

```python
# === MODELO FINAL — TEMPERATURA ===

modelo_temp_final = LGBMRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)

modelo_temp_final.fit(
    X_train,
    y_temp_train,
    eval_set=[(X_test, y_temp_test)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
)

y_pred_train_final = modelo_temp_final.predict(X_train)
y_pred_valid_final = modelo_temp_final.predict(X_valid)
y_pred_test_final  = modelo_temp_final.predict(X_test)

print("Treino:", avaliar(y_temp_train, y_pred_train_final))
print("Validação:", avaliar(y_temp_valid, y_pred_valid_final))
print("Teste:", avaliar(y_temp_test, y_pred_test_final))

# === MODELO FINAL — CHUVA (RAIN) ===

modelo_rain_final = LGBMRegressor(
    **best_params_rain,
    random_state=42,
    n_jobs=-1
)

modelo_rain_final.fit(
    X_train,
    y_rain_train,
    eval_set=[(X_test, y_rain_test)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
)

# Previsões finais
y_rain_pred_train_final = modelo_rain_final.predict(X_train)
y_rain_pred_valid_final = modelo_rain_final.predict(X_valid)
y_rain_pred_test_final  = modelo_rain_final.predict(X_test)

# Métricas finais
print("\n===== RESULTADOS FINAIS - RAIN =====\n")

print("Treino:")
print("MAE  =", round(mean_absolute_error(y_rain_train, y_rain_pred_train_final), 3))
print("RMSE =", round(np.sqrt(mean_squared_error(y_rain_train, y_rain_pred_train_final)), 3))

print("\nValidação Interna:")
print("MAE  =", round(mean_absolute_error(y_rain_valid, y_rain_pred_valid_final), 3))
print("RMSE =", round(np.sqrt(mean_squared_error(y_rain_valid, y_rain_pred_valid_final)), 3))

print("\nTeste Final:")
print("MAE  =", round(mean_absolute_error(y_rain_test, y_rain_pred_test_final), 3))
print("RMSE =", round(np.sqrt(mean_squared_error(y_rain_test, y_rain_pred_test_final)), 3))

```

---

## **8. Geração das Previsões Futuras**

Para prever o próximo ano hora a hora:

* criação de `time_ref = time - 365 dias`
* reaproveitamento das features correspondentes
* previsão utilizando os dois modelos treinados
* pós-processamento:

  * temperatura com 1 casa decimal
  * chuva limitada a valores ≥ 0

Esse método mantém coerência temporal, reaproveitando padrões históricos referentes ao mesmo período anual.

```python
# === PREPARO PARA PREVISÃO DO PRÓXIMO ANO ===

# df_features_scaled terá: time + todas as features escaladas
df_features_scaled = pd.concat([df["time"], X_scaled], axis=1)

# Datas futuras
future_times = pd.date_range(
    start=df["time"].max() + pd.Timedelta(hours=1),
    periods=24 * 365,
    freq="H"
)

future_df = pd.DataFrame({"time": future_times})
future_df["time_ref"] = future_df["time"] - pd.Timedelta(days=365)

# Merge com o ano anterior
future_merged = future_df.merge(
    df_features_scaled,
    left_on="time_ref",
    right_on="time",
    how="left",
    suffixes=("_future", "_past")
).drop(columns=["time_past"]).rename(columns={"time_future": "time"})

# === PREVISÃO FINAL ===

X_future = future_merged[feature_cols].fillna(0)

temp_future = np.round(modelo_temp_final.predict(X_future), 1)
rain_future = np.round(np.clip(modelo_rain_final.predict(X_future), 0, None), 1)

df_prev = pd.DataFrame({
    "time": future_merged["time"],
    "temperature": temp_future,
    "rain": rain_future
})

df_prev.head()
```

---

## **9. Exportação Final**

O arquivo:

```
PREVISAO.csv
```

contém:

* `time` (yyyy-MM-ddTHH:mm)
* `temperature`
* `rain`

---

# 📊 **Resultados dos Modelos**

## **Temperatura**

* **MAE (Teste):** ~0.39°C
* **RMSE (Teste):** ~0.53°C
* **Erro relativo médio:** ~1.3%

O modelo apresenta bom desempenho para previsões horárias, com baixa variabilidade.

---

## **Chuva**

* **MAE (Teste):** ~0.10 mm
* **RMSE (Teste):** ~0.38 mm
* **Desvio nos folds:** ~0.02

A estabilidade entre os folds indica um modelo consistente e generalizável.

<img width="1286" height="418" alt="{251FF2DA-3A9E-4BF7-BE27-CEDD94A212BE}" src="https://github.com/user-attachments/assets/afcf6543-661a-4edd-af97-677e30ea6f83" />

<img width="1263" height="408" alt="{A5DEA6F2-F893-4459-BF3F-D0A83458BFFE}" src="https://github.com/user-attachments/assets/11f3ad42-fc93-4ff7-8029-02784ce24774" />

Esses resultados atendem plenamente ao desafio, oferecendo previsões coesas e confiáveis.
