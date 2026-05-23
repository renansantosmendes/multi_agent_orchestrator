import json
import inspect
import operator
import warnings
from typing import Annotated, List, Dict, Any

import numpy as np
import pandas as pd
import joblib
from typing_extensions import TypedDict

# LangChain / LangGraph / DeepAgents
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Evidently
from evidently import Report
from evidently.presets import DataDriftPreset

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")
print("✅ Imports OK")


# %% CELL 4 — Constantes e Store Global
DATASET_URL = (
    "https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023"
    "/master/fetal_health.csv"
)

# Store global — memória compartilhada entre as tools dos subagents
_pipeline_store: Dict[str, Any] = {}

print("✅ Constantes definidas")


# %% CELL 5 — Context Schema (Estado compartilhado do grafo)
class MLPipelineContext(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

    # Drift
    drift_detected: bool
    drift_summary: str
    drifted_columns: Annotated[List[str], operator.add]

    # Preprocessing
    preprocessing_done: bool
    feature_columns: Annotated[List[str], operator.add]
    target_column: str

    # Training
    accuracy_history: Annotated[List[float], operator.add]
    trained_models: Annotated[List[str], operator.add]
    best_model_name: str
    best_accuracy: float

    # Deploy
    model_path: str
    current_stage: str

print("✅ MLPipelineContext definido")


# %% CELL 6 — Tool: detect_data_drift (DriftAgent)
@tool
def detect_data_drift(
    reference_start: int,
    reference_end: int,
    current_start: int,
    current_end: int,
    drift_share_threshold: float = 0.5,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Detecta data drift entre uma fatia de referência e uma fatia atual
    do dataset fetal_health usando Evidently DataDriftPreset.

    Args:
        reference_start: Índice inicial dos dados de referência.
        reference_end: Índice final dos dados de referência.
        current_start: Índice inicial dos dados atuais.
        current_end: Índice final dos dados atuais.
        drift_share_threshold: Fração de colunas com drift para considerar
                               drift global (default 0.5).
    """
    df = pd.read_csv(DATASET_URL)
    reference_data = df.iloc[reference_start:reference_end]
    current_data = df.iloc[current_start:current_end]

    report = Report(metrics=[DataDriftPreset()], include_tests="True")
    result = report.run(current_data=current_data, reference_data=reference_data)
    result_dict = result.dict()

    metrics = result_dict.get("metrics", [])
    drift_count_metric = metrics[0] if metrics else {}
    drift_count = drift_count_metric.get("value", {}).get("count", 0)
    drift_share = drift_count_metric.get("value", {}).get("share", 0.0)

    drifted_cols = []
    column_details = []
    for m in metrics[1:]:
        col_name = m.get("config", {}).get("column", "unknown")
        p_value = m.get("value", 1.0)
        threshold = m.get("config", {}).get("threshold", 0.05)
        has_drift = p_value < threshold
        if has_drift:
            drifted_cols.append(col_name)
        column_details.append(
            f"  - {col_name}: p-value={p_value:.4f} | "
            f"drift={'SIM' if has_drift else 'NÃO'}"
        )

    drift_detected = drift_share >= drift_share_threshold

    summary = "\n".join([
        "📊 RELATÓRIO DE DATA DRIFT",
        f"   Referência: linhas [{reference_start}:{reference_end}] "
        f"({reference_end - reference_start} amostras)",
        f"   Atual:      linhas [{current_start}:{current_end}] "
        f"({current_end - current_start} amostras)",
        "",
        f"   Colunas com drift: {int(drift_count)} / {len(metrics)-1} "
        f"({drift_share:.1%})",
        f"   Threshold global: {drift_share_threshold:.0%}",
        f"   🚨 DRIFT GLOBAL: {'SIM' if drift_detected else 'NÃO'}",
        "",
        "   Detalhes por coluna:",
    ] + column_details)

    return Command(
        update={
            "drift_detected": drift_detected,
            "drift_summary": summary,
            "drifted_columns": drifted_cols,
            "current_stage": "drift_check",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )

print("✅ Tool: detect_data_drift")


# %% CELL 7 — Tool: preprocess_data (PreprocessAgent)
@tool
def preprocess_data(
    target_column: str = "fetal_health",
    test_size: float = 0.2,
    random_state: int = 42,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Pré-processa o dataset fetal_health: separa features/target,
    normaliza com StandardScaler e faz train/test split estratificado.

    Args:
        target_column: Nome da coluna alvo.
        test_size: Fração dos dados para teste.
        random_state: Seed de reprodutibilidade.
    """
    df = pd.read_csv(DATASET_URL)

    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols
    )

    _pipeline_store["X_train"] = X_train_scaled
    _pipeline_store["X_test"] = X_test_scaled
    _pipeline_store["y_train"] = y_train.values
    _pipeline_store["y_test"] = y_test.values
    _pipeline_store["scaler"] = scaler
    _pipeline_store["feature_cols"] = feature_cols

    summary = (
        f"⚙️ PREPROCESSING CONCLUÍDO\n"
        f"   Features: {len(feature_cols)} colunas\n"
        f"   Train: {X_train_scaled.shape[0]} amostras\n"
        f"   Test:  {X_test_scaled.shape[0]} amostras\n"
        f"   Scaler: StandardScaler\n"
        f"   Stratify: Sim"
    )

    return Command(
        update={
            "preprocessing_done": True,
            "feature_columns": feature_cols,
            "target_column": target_column,
            "current_stage": "preprocessing",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )

print("✅ Tool: preprocess_data")


# %% CELL 8 — Tool: train_model (TrainerAgent)
@tool
def train_model(
    model_name: str,
    params: Dict[str, Any] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Treina um modelo de ML no dataset fetal_health já pré-processado.

    Args:
        model_name: Nome do modelo — 'RandomForest', 'LogisticRegression'
                    ou 'GradientBoosting'.
        params: Hiperparâmetros opcionais (ex: {'n_estimators': 200}).
    """
    if "X_train" not in _pipeline_store:
        msg = "❌ Erro: execute preprocess_data antes de treinar."
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
            }
        )

    X_train = _pipeline_store["X_train"]
    X_test = _pipeline_store["X_test"]
    y_train = _pipeline_store["y_train"]
    y_test = _pipeline_store["y_test"]

    params = params or {}

    # Selecionar classe
    model_map = {
        "randomforest": (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
        "logisticregression": (LogisticRegression, {"max_iter": 2000}),
        "gradientboosting": (GradientBoostingClassifier, {"n_estimators": 100, "random_state": 42}),
    }

    key = model_name.lower().replace(" ", "").replace("_", "")
    model_class, defaults = model_map.get(key, (RandomForestClassifier, {"n_estimators": 100}))

    # Aplicar defaults e filtrar params válidos
    merged = {**defaults, **params}
    valid_args = set(inspect.signature(model_class).parameters.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_args}

    model = model_class(**filtered)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Guardar no store
    _pipeline_store["trained_model"] = model
    _pipeline_store["y_pred"] = y_pred
    _pipeline_store["last_model_name"] = model_name
    _pipeline_store["last_accuracy"] = acc

    # Guardar histórico de modelos treinados
    history = _pipeline_store.get("models_trained", [])
    history.append({"name": model_name, "accuracy": round(acc, 4), "params": filtered})
    _pipeline_store["models_trained"] = history

    summary = (
        f"🏋️ TREINAMENTO CONCLUÍDO\n"
        f"   Modelo: {model_name}\n"
        f"   Params: {filtered}\n"
        f"   Acurácia: {acc:.4f} ({acc:.1%})"
    )

    return Command(
        update={
            "accuracy_history": [round(acc, 4)],
            "trained_models": [model_name],
            "best_model_name": model_name,
            "best_accuracy": round(acc, 4),
            "current_stage": "training",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )

print("✅ Tool: train_model")


# %% CELL 9 — Tool: analyze_results (ResultAnalyzerAgent)
@tool
def analyze_results(
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Gera relatório detalhado com classification_report, feature importances
    e comparação entre modelos treinados na sessão."""

    if "trained_model" not in _pipeline_store:
        msg = "❌ Erro: nenhum modelo treinado. Execute train_model primeiro."
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
            }
        )

    y_test = _pipeline_store["y_test"]
    y_pred = _pipeline_store["y_pred"]
    model = _pipeline_store["trained_model"]
    model_name = _pipeline_store.get("last_model_name", "unknown")
    feature_cols = _pipeline_store.get("feature_cols", [])
    models_history = _pipeline_store.get("models_trained", [])

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Suspect", "Pathological"],
    )

    # Feature importances
    importance_text = ""
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        top = importances[:10]
        lines = [f"  {i+1}. {n}: {v:.4f}" for i, (n, v) in enumerate(top)]
        importance_text = "\n\n   🏆 TOP 10 FEATURES:\n" + "\n".join(lines)
    elif hasattr(model, "coef_"):
        avg_coef = np.abs(model.coef_).mean(axis=0)
        importances = sorted(
            zip(feature_cols, avg_coef),
            key=lambda x: x[1], reverse=True,
        )
        top = importances[:10]
        lines = [f"  {i+1}. {n}: {v:.4f}" for i, (n, v) in enumerate(top)]
        importance_text = (
            "\n\n   🏆 TOP 10 FEATURES (|coef| médio):\n" + "\n".join(lines)
        )

    # Comparação entre modelos treinados
    comparison_text = ""
    if len(models_history) > 1:
        sorted_models = sorted(models_history, key=lambda x: x["accuracy"], reverse=True)
        comp_lines = [
            f"  {'→' if m['name'] == model_name else ' '} "
            f"{m['name']}: {m['accuracy']:.4f}"
            for m in sorted_models
        ]
        comparison_text = (
            "\n\n   📊 COMPARAÇÃO DE MODELOS:\n" + "\n".join(comp_lines)
        )
        best = sorted_models[0]
        comparison_text += (
            f"\n\n   ✅ Melhor modelo: {best['name']} ({best['accuracy']:.4f})"
        )

    summary = (
        f"📈 ANÁLISE DE RESULTADOS — {model_name}\n\n"
        f"   Classification Report:\n{report}"
        f"{importance_text}"
        f"{comparison_text}"
    )

    return Command(
        update={
            "current_stage": "evaluation",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )

print("✅ Tool: analyze_results")


# %% CELL 10 — Tool: deploy_model (DeployAgent)
@tool
def deploy_model(
    model_path: str = "best_model.joblib",
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Serializa o modelo treinado com joblib, salva o scaler
    e gera um arquivo de metadados JSON.

    Args:
        model_path: Caminho do arquivo .joblib para salvar o modelo.
    """
    if "trained_model" not in _pipeline_store:
        msg = "❌ Erro: nenhum modelo para deploy. Execute train_model primeiro."
        return Command(
            update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
            }
        )

    model = _pipeline_store["trained_model"]
    scaler = _pipeline_store.get("scaler")
    model_name = _pipeline_store.get("last_model_name", "unknown")
    accuracy = _pipeline_store.get("last_accuracy", 0.0)
    feature_cols = _pipeline_store.get("feature_cols", [])

    # Salvar modelo
    joblib.dump(model, model_path)

    # Salvar scaler
    scaler_path = model_path.replace(".joblib", "_scaler.joblib")
    if scaler is not None:
        joblib.dump(scaler, scaler_path)

    # Metadados
    metadata = {
        "model_name": model_name,
        "accuracy": accuracy,
        "features": feature_cols,
        "model_file": model_path,
        "scaler_file": scaler_path,
    }
    meta_path = model_path.replace(".joblib", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    summary = (
        f"🚀 DEPLOY CONCLUÍDO\n"
        f"   Modelo:    {model_name} (acurácia: {accuracy:.4f})\n"
        f"   Artefatos:\n"
        f"     - Modelo:   {model_path}\n"
        f"     - Scaler:   {scaler_path}\n"
        f"     - Metadata: {meta_path}\n"
        f"   Status: ✅ Pronto para produção"
    )

    return Command(
        update={
            "model_path": model_path,
            "current_stage": "deployed",
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )

print("✅ Tool: deploy_model")


# %% CELL 11 — Definição dos SubAgents
drift_agent: SubAgent = {
    "name": "drift_detector",
    "description": (
        "Especialista em detecção de data drift. Delege a ele quando "
        "precisar verificar se houve mudança na distribuição dos dados. "
        "Ele usa Evidently para comparar dados de referência vs atuais."
    ),
    "system_prompt": (
        "Você é o DataDriftDetectorAgent, especialista em detecção de drift.\n\n"
        "SUAS RESPONSABILIDADES:\n"
        "- Usar a tool detect_data_drift para analisar drift nos dados\n"
        "- Interpretar os resultados: quais colunas driftaram, p-values\n"
        "- Emitir um parecer claro: drift detectado ou não\n\n"
        "Ao ser chamado, use detect_data_drift com os parâmetros fornecidos "
        "e retorne um resumo estruturado do resultado."
        "NUNCA inclua frases de cortesia, despedida ou oferecimento de ajuda. "
        "Seja direto e técnico."
    ),
    "tools": [detect_data_drift],
}

preprocess_agent: SubAgent = {
    "name": "preprocessor",
    "description": (
        "Especialista em pré-processamento de dados. Delege a ele para "
        "normalizar features, separar target, e fazer train/test split."
    ),
    "system_prompt": (
        "Você é o PreprocessAgent, especialista em preparação de dados.\n\n"
        "SUAS RESPONSABILIDADES:\n"
        "- Usar a tool preprocess_data para preparar o dataset\n"
        "- Garantir que os dados estejam normalizados e prontos para treino\n"
        "- Reportar: quantas features, tamanho do train/test\n\n"
        "Ao ser chamado, execute preprocess_data e retorne o resumo."
        "NUNCA inclua frases de cortesia, despedida ou oferecimento de ajuda. "
        "Seja direto e técnico."
    ),
    "tools": [preprocess_data],
}

trainer_agent: SubAgent = {
    "name": "trainer",
    "description": (
        "Especialista em treinamento de modelos de ML. Delege a ele para "
        "treinar modelos como RandomForest, LogisticRegression ou "
        "GradientBoosting. Ele pode treinar múltiplos modelos em sequência "
        "para comparar resultados."
    ),
    "system_prompt": (
        "Você é o TrainerAgent, especialista em treinamento de modelos.\n\n"
        "SUAS RESPONSABILIDADES:\n"
        "- Usar a tool train_model para treinar modelos\n"
        "- Você pode treinar MÚLTIPLOS modelos se solicitado\n"
        "- Modelos disponíveis: RandomForest, LogisticRegression, GradientBoosting\n"
        "- Reportar acurácia de cada modelo treinado\n\n"
        "Se o usuário pedir para encontrar o melhor modelo, treine pelo menos "
        "2 modelos diferentes e compare os resultados.\n"
        "Sempre retorne um resumo com a acurácia de todos os modelos treinados."
        "NUNCA inclua frases de cortesia, despedida ou oferecimento de ajuda. "
        "Seja direto e técnico."
    ),
    "tools": [train_model],
}

analyzer_agent: SubAgent = {
    "name": "result_analyzer",
    "description": (
        "Especialista em análise de resultados de ML. Delege a ele para "
        "gerar classification reports, feature importances e comparações "
        "entre modelos."
    ),
    "system_prompt": (
        "Você é o ResultAnalyzerAgent, especialista em avaliação de modelos.\n\n"
        "SUAS RESPONSABILIDADES:\n"
        "- Usar a tool analyze_results para gerar relatórios detalhados\n"
        "- Interpretar precision, recall, f1-score por classe\n"
        "- Destacar as features mais importantes\n"
        "- Comparar modelos se mais de um foi treinado\n"
        "- Dar recomendação final sobre qual modelo usar\n\n"
        "Retorne uma análise clara e acionável."
        "NUNCA inclua frases de cortesia, despedida ou oferecimento de ajuda. "
        "Seja direto e técnico."
    ),
    "tools": [analyze_results],
}

deploy_agent: SubAgent = {
    "name": "deployer",
    "description": (
        "Especialista em deploy de modelos. Delege a ele para serializar "
        "o modelo treinado, salvar scaler e metadados."
    ),
    "system_prompt": (
        "Você é o DeployAgent, especialista em deploy de modelos.\n\n"
        "SUAS RESPONSABILIDADES:\n"
        "- Usar a tool deploy_model para salvar o modelo em produção\n"
        "- Verificar que modelo, scaler e metadados foram salvos\n"
        "- Reportar os caminhos dos artefatos gerados\n\n"
        "Execute o deploy e retorne o status final."
        "NUNCA inclua frases de cortesia, despedida ou oferecimento de ajuda. "
        "Seja direto e técnico."
    ),
    "tools": [deploy_model],
}

print("✅ SubAgents definidos: drift_detector, preprocessor, trainer, result_analyzer, deployer")


# %% CELL 12 — Orquestrador com SubAgents
ORCHESTRATOR_PROMPT = """Você é o Orquestrador Principal de um pipeline de ML para o dataset fetal_health.
Você NÃO executa tarefas diretamente. Você DELEGA para subagents especializados usando a tool `task`.

SUBAGENTS DISPONÍVEIS:
- drift_detector: verifica data drift
- preprocessor: pré-processa dados
- trainer: treina modelos
- result_analyzer: analisa resultados
- deployer: faz deploy do modelo

FLUXO OBRIGATÓRIO:
1. Delegue ao `drift_detector` para verificar drift (reference_start=0, reference_end=200, current_start=1000, current_end=1200)
2. Delegue ao `preprocessor` para preparar os dados
3. Delegue ao `trainer` para treinar o(s) modelo(s) solicitado(s)
4. Delegue ao `result_analyzer` para avaliar os resultados
5. Delegue ao `deployer` para fazer o deploy

REGRAS:
- Execute TODAS as etapas em sequência, sem interrupção.
- NUNCA pare para perguntar ao usuário se deseja prosseguir. Você é autônomo.
- Se drift for detectado, registre um WARNING e continue o pipeline.
- Explique seu raciocínio brevemente ANTES de cada delegação.
- Ao final, apresente um resumo consolidado de todo o pipeline.
"""


def create_ml_orchestrator(model_name: str = "gpt-4o-mini", temperature: float = 0):
    """Cria o orquestrador com arquitetura de subagents."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    orchestrator = create_deep_agent(
        model=llm,
        tools=[],  # Orquestrador não tem tools próprias — só delega
        subagents=[
            drift_agent,
            preprocess_agent,
            trainer_agent,
            analyzer_agent,
            deploy_agent,
        ],
        context_schema=MLPipelineContext,
        system_prompt=ORCHESTRATOR_PROMPT,
    )
    return orchestrator

print("✅ Orquestrador com SubAgents configurado")


# %% CELL 13 — Execução do Pipeline
orchestrator = create_ml_orchestrator()

initial_state = {
    "messages": [
        HumanMessage(
            content=(
                "Execute o pipeline completo de ML. "
                "Verifique drift, pré-processe os dados, "
                "treine um RandomForest e um LogisticRegression, "
                "analise os resultados e faça o deploy do melhor modelo."
            )
        )
    ],
    "drift_detected": False,
    "drift_summary": "",
    "drifted_columns": [],
    "preprocessing_done": False,
    "feature_columns": [],
    "target_column": "",
    "accuracy_history": [],
    "trained_models": [],
    "best_model_name": "",
    "best_accuracy": 0.0,
    "model_path": "",
    "current_stage": "start",
}

print("🚀 Iniciando Pipeline ML com SubAgents...\n")

try:
    for event in orchestrator.stream(initial_state, stream_mode="values"):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        agent_name = tc.get("args", {}).get("subagent_type", tc.get("name", "?"))
                        print(f"🛠️  Delegando para: {agent_name}")
                elif isinstance(last_msg, ToolMessage):
                    print(f"📦 Resultado:\n{last_msg.content}\n")
                else:
                    print(f"🤖 Orquestrador:\n{last_msg.content}\n")
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
