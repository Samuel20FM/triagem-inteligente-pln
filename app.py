"""
Triagem Inteligente de Mensagens — PLN: Representação & Classificação (Grupo 13)
Disciplina de Inteligência Artificial 2026 · UP-Maputo (FET)

Interface Streamlit. A lógica está nos módulos:
  texto.py · representacoes.py · classificadores.py · avaliacao.py
  urgencia.py · agente.py · responsavel.py
"""
import random
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from agente import (TRIAGEM_HUMANA, UTILIDADE, curva_utilidade, decidir, desempenho_agente,
                    ordenar_fila)
from avaliacao import ler_csv, tabela_resultados, treinar, validar_corpus
from classificadores import CLASSIFICADORES, contribuicoes, prever
from representacoes import REPRESENTACOES, criar_vectorizador, ngramas_caracteres, ngramas_partilhados
from responsavel import (anonimizar, limiares_por_subgrupo, metricas_por_subgrupo,
                         metricas_urgencia_por_subgrupo, pesos_reponderacao,
                         taxa_urgencia_a_categoria_igual)
from texto import normalizar
from urgencia import (curva_limiar, metricas_urgencia, prob_fora_da_amostra, prob_urgente,
                      treinar_urgencia)

# ==============================================================================
# CONFIGURAÇÃO E ESTILO (funciona nos temas claro e escuro: sem cores de texto fixas)
# ==============================================================================
st.set_page_config(page_title="Triagem Inteligente · Grupo 13", layout="wide",
                   initial_sidebar_state="expanded")

CSV_PADRAO = Path(__file__).parent / "corpus_referencia.csv"
BASELINE = ("bow", "nb")  # a versão original do protótipo
PALETA = ["#2563eb", "#f59e0b", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]

st.markdown(
    """
    <style>
      .block-container {padding-top:2rem; padding-bottom:3rem; max-width:1200px;}
      h5 {font-size:0.78rem !important; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
          opacity:0.75; margin:1.4rem 0 0.6rem 0; padding-bottom:0.4rem;
          border-bottom:1px solid rgba(128,128,128,0.25);}
      div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,0.25); border-radius:10px; padding:12px 16px;}
      div[data-testid="stMetricLabel"] p {font-size:0.75rem; font-weight:600; letter-spacing:0.04em;
        text-transform:uppercase; opacity:0.75;}
      .rotulo {font-size:0.72rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; opacity:0.7;}
      .categoria {font-size:1.8rem; font-weight:700; letter-spacing:-0.02em; line-height:1.2; margin:0.15rem 0 0.6rem 0;}
      .decisao {border-left:4px solid #2563eb; background:rgba(37,99,235,0.10); padding:0.75rem 1rem;
        border-radius:0 8px 8px 0; margin:0.4rem 0;}
      .decisao.alta {border-left-color:#dc2626; background:rgba(220,38,38,0.10);}
      .decisao.humana {border-left-color:#f59e0b; background:rgba(245,158,11,0.12);}
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(v):
    return f"{v * 100:.1f}%"


def nome_modelo(chave):
    rep, clf = chave
    return f"{REPRESENTACOES[rep]} + {CLASSIFICADORES[clf]}"


def grafico_cm(cm, rotulos, titulo):
    d = pd.DataFrame([(rotulos[i], rotulos[j], int(cm[i, j]))
                      for i in range(len(rotulos)) for j in range(len(rotulos))],
                     columns=["real", "prevista", "n"])
    base = alt.Chart(d).encode(
        x=alt.X("prevista:N", sort=list(rotulos), title="Prevista", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("real:N", sort=list(rotulos), title="Real"))
    heat = base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
    txt = base.mark_text(fontSize=14).encode(
        text="n:Q", color=alt.condition(alt.datum.n > cm.max() / 2, alt.value("white"), alt.value("black")))
    return (heat + txt).properties(title=titulo, height=260)


# ==============================================================================
# CACHE (o Streamlit só existe aqui; os módulos não dependem dele)
# ==============================================================================
@st.cache_data(show_spinner=False)
def carregar(origem):
    return ler_csv(origem)


@st.cache_resource(show_spinner="A treinar e avaliar 6 modelos (3 representações × 2 classificadores)…")
def treinar_cache(_df, chave_df, usar_stop, ngram, prop, seed, alpha, C):
    return treinar(_df, usar_stop, ngram, prop, seed, alpha, C)


@st.cache_resource(show_spinner="A treinar o modelo de urgência…")
def urgencia_cache(_df, _R, chave_df, usar_stop, ngram, prop, seed, alpha):
    tr, te = _R["idx_treino"], _R["idx_teste"]
    out = {"base": treinar_urgencia(_df, tr, te, usar_stop, ngram, alpha)}
    if "registo" in _df.columns:
        w = pesos_reponderacao(_df["registo"], _df["urgente"])
        out["reponderado"] = treinar_urgencia(_df, tr, te, usar_stop, ngram, alpha, pesos=w)
        out["p_oof"] = prob_fora_da_amostra(_df, tr, usar_stop, ngram, alpha, seed)
    return out


# ==============================================================================
# DADOS + BARRA LATERAL
# ==============================================================================
st.sidebar.title("Triagem Inteligente")
st.sidebar.caption("PLN · Representação & Classificação · Grupo 13")

with st.sidebar.expander("Fonte de dados", expanded=False):
    upload = st.file_uploader("Substituir corpus (CSV)", type="csv",
                              help="Precisa das colunas `mensagem` e `categoria`; `urgente` e `registo` são opcionais.")

try:
    df = carregar(upload if upload is not None else CSV_PADRAO)
except (ValueError, FileNotFoundError, OSError) as e:
    st.error("Não foi possível ler o corpus. Coloque `corpus_referencia.csv` na mesma pasta do `app.py` "
             f"ou carregue um CSV com as colunas `mensagem` e `categoria`.\n\nDetalhe: {e}")
    st.stop()

problemas = validar_corpus(df)
if problemas:
    st.error("O corpus não serve para treinar e avaliar:\n\n- " + "\n- ".join(problemas))
    st.stop()

TEM_URG = "urgente" in df.columns and set(df["urgente"].dropna().unique()) <= {0, 1}
TEM_REG = "registo" in df.columns

with st.sidebar.expander("Parâmetros do experimento", expanded=False):
    usar_stop = st.checkbox("Remover stop-words", value=True, help="Mantém as negações (nao, sem, nem). Não afecta os n-gramas de caracteres.")
    ngram_op = st.radio("Unidade de texto (palavras)", ["Palavras (unigramas)", "Palavras + pares (bigramas)"], index=0,
                        help="Bigramas capturam expressões como 'nao validou'.")
    ngram = (1, 1) if ngram_op.startswith("Palavras (") else (1, 2)
    prop = st.select_slider("Proporção de teste", options=[0.10, 0.20, 0.25, 0.33], value=0.20,
                            format_func=lambda v: f"{v:.0%}")
    alpha = st.slider("Naive Bayes: suavização de Laplace (α)", 0.01, 2.0, 1.0, 0.01)
    C = st.select_slider("Regressão Logística: C (inverso da regularização)", options=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0], value=5.0)
    seed = int(st.number_input("Semente aleatória", 0, 9999, 42))

st.sidebar.markdown("**Agente**")
limiar = st.sidebar.slider("Confiança mínima para encaminhar", 30, 90, 55, 5,
                           help="Abaixo deste valor a mensagem vai para triagem humana.") / 100
limiar_urg = st.sidebar.slider("P(urgente) para prioridade ALTA", 10, 90, 50, 5, disabled=not TEM_URG) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("**Grupo 13** · Samuel Massinga & Arney Nhadumbuque")
st.sidebar.caption("Docente: MSc. Sheila A. Sitoe Rangel · UP-Maputo — FET\n\n"
                   "Tema claro/escuro: menu ⋮ (canto superior direito) → Settings → Theme.")

chave_df = str(pd.util.hash_pandas_object(df).sum())
R = treinar_cache(df, chave_df, usar_stop, ngram, prop, seed, alpha, C)
classes = R["classes"]
U = urgencia_cache(df, R, chave_df, usar_stop, ngram, prop, seed, alpha) if TEM_URG else None
teste = R["teste"]

# ==============================================================================
# CABEÇALHO
# ==============================================================================
st.title("Triagem Inteligente de Mensagens de Utentes")
st.caption(f"Corpus: **{len(df)} mensagens** · **{len(classes)} categorias** · treino {R['n_treino']} / "
           f"teste {R['n_teste']} (agrupado e estratificado) · melhor modelo na validação cruzada: "
           f"**{nome_modelo(R['melhor'])}**")

tabs = st.tabs(["Classificar"])
# Separadores retirados da interface (código comentado mais abaixo, para poder repor):
# tabs = st.tabs(["Classificar", "Como o texto vira números", "Comparação", "Urgência & fila",
#                 "IA responsável", "Corpus", "Teoria & PEAS"])

# ------------------------------------------------------------------------------
# 1 — CLASSIFICAR
# ------------------------------------------------------------------------------
with tabs[0]:
    if "msg" not in st.session_state:
        st.session_state.msg = "paguei a propina mas o sistema nao validou e amanha tenho exame"
    exemplos = df["mensagem"].drop_duplicates().sample(min(40, df["mensagem"].nunique()), random_state=7).tolist()

    def _escolher():
        if st.session_state.exemplo != "—":
            st.session_state.msg = st.session_state.exemplo

    def _aleatoria():
        st.session_state.msg = random.choice(df["mensagem"].tolist())

    st.markdown("##### 1 · Mensagem do utente")
    c_in, c_ex = st.columns([3, 2])
    with c_in:
        st.text_area("Mensagem do utente", key="msg", height=110, label_visibility="collapsed",
                     placeholder="Ex.: o projector da sala 14 nao funciona")
    with c_ex:
        st.selectbox("Exemplos do corpus", ["—"] + exemplos, key="exemplo", on_change=_escolher)
        st.button("Mensagem aleatória do corpus", on_click=_aleatoria, width="stretch")

    original = st.session_state.msg.strip()
    if not original:
        st.info("Escreva uma mensagem para ver a classificação.")
    else:
        msg, achados = anonimizar(original)
        if achados:
            st.info("**Privacidade:** foram mascarados dados pessoais antes de classificar — "
                    + ", ".join(f"{t.lower()}" for t, _ in achados) + f".\n\nTexto usado: `{msg}`")

        st.markdown("##### 2 · Modelo do agente")
        cr, cc = st.columns(2)
        reps, clfs = list(REPRESENTACOES), list(CLASSIFICADORES)
        rep = cr.radio("Representação", reps, index=reps.index(R["melhor"][0]),
                       format_func=REPRESENTACOES.get, horizontal=True)
        clf = cc.radio("Classificador", clfs, index=clfs.index(R["melhor"][1]),
                       format_func=CLASSIFICADORES.get, horizontal=True)
        pipe = R["modelos"][(rep, clf)]["pipe"]
        x, proba, cls, _ = prever(pipe, msg)
        pred, conf = cls[np.argmax(proba)], float(proba.max())
        p_urg = prob_urgente(U["base"]["pipe"], msg) if TEM_URG else 0.0
        d = decidir(pred, conf, p_urg, limiar, limiar_urg)

        st.markdown("##### 3 · Decisão")
        with st.container(border=True):
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='rotulo'>Categoria</div><div class='categoria'>{pred.capitalize()}</div>",
                        unsafe_allow_html=True)
            k2.metric("Confiança", pct(conf))
            k3.metric("P(urgente)", pct(p_urg) if TEM_URG else "—")
            k4.metric("Prioridade", d.prioridade if TEM_URG else "—")
            classe_css = "humana" if d.accao == "triagem_humana" else ("alta" if d.prioridade == "ALTA" else "")
            st.markdown(f"<div class='decisao {classe_css}'><b>{'Enviar para' if d.accao == 'triagem_humana' else 'Encaminhar para'} →</b> "
                        f"{d.destino}<br><small>{d.motivo}</small></div>", unsafe_allow_html=True)

        st.markdown("##### 4 · O que dizem as 6 combinações")
        grelha = pd.DataFrame(
            {CLASSIFICADORES[c]: [f"{R['modelos'][(r, c)]['pipe'].classes_[np.argmax(p := prever(R['modelos'][(r, c)]['pipe'], msg)[1])]}"
                                  f" · {pct(p.max())}" for r in reps] for c in clfs},
            index=[REPRESENTACOES[r] for r in reps])
        st.dataframe(grelha, width="stretch")
        if len(set(v.split(" · ")[0] for v in grelha.to_numpy().ravel())) > 1:
            st.warning("As combinações **discordam** — a representação e o classificador mudam a decisão.")

        ca, cb = st.columns(2)
        with ca:
            st.markdown("##### 5 · Probabilidades por categoria")
            dprob = pd.DataFrame({"categoria": cls, "probabilidade": proba})
            st.altair_chart(alt.Chart(dprob).mark_bar().encode(
                y=alt.Y("categoria:N", sort="-x", title=None),
                x=alt.X("probabilidade:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None),
                color=alt.Color("categoria:N", scale=alt.Scale(domain=classes, range=PALETA[:len(classes)]), legend=None),
                tooltip=["categoria", alt.Tooltip("probabilidade:Q", format=".1%")],
            ).properties(height=180), width="stretch")
        with cb:
            st.markdown("##### 6 · Porquê? — termos que pesaram")
            dcontr, desconhecidos, c1, c2 = contribuicoes(pipe, msg, x, proba, cls)
            if dcontr.empty:
                st.info("Nenhum termo da mensagem existe no vocabulário — a decisão baseia-se só nas frequências das categorias.")
            else:
                if rep == "char":  # o espaço que marca início/fim de palavra fica visível como "_"
                    dcontr["termo"] = dcontr["termo"].str.replace(" ", "_")
                st.caption(f"À direita empurra para **{c1}**; à esquerda para **{c2}** (2.ª mais provável)."
                           + (" `_` marca o início ou o fim de uma palavra." if rep == "char" else ""))
                st.altair_chart(alt.Chart(dcontr).mark_bar().encode(
                    y=alt.Y("termo:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelFont="monospace")),
                    x=alt.X("contribuição:Q", title=f"contribuição (log-odds {c1} vs {c2})"),
                    color=alt.condition(alt.datum["contribuição"] > 0, alt.value("#2563eb"), alt.value("#ef4444")),
                    tooltip=["termo", alt.Tooltip("contribuição:Q", format=".2f")],
                ).properties(height=max(140, 22 * len(dcontr))), width="stretch")
            if desconhecidos and rep != "char":
                st.caption("Fora do vocabulário (ignoradas): " + ", ".join(f"`{t}`" for t in desconhecidos))

        exacta = df[df["mensagem"].map(normalizar) == normalizar(msg)]["categoria"].unique()
        if len(exacta):
            st.caption("Esta mensagem existe no corpus com a(s) etiqueta(s): " + ", ".join(f"`{c}`" for c in exacta) + ".")

# ==============================================================================
# SEPARADORES 2–7 DESACTIVADOS (comentados). Para repor: descomentar este bloco
# e voltar a usar a lista completa em st.tabs(...) acima.
# ==============================================================================
# # ------------------------------------------------------------------------------
# # 2 — DO TEXTO AOS NÚMEROS
# # ------------------------------------------------------------------------------
# with tabs[1]:
#     st.markdown("Veja **passo a passo** como a mesma frase é transformada por cada representação.")
#     frase = st.text_input("Frase para analisar", value=st.session_state.get("msg", ""), key="frase_passo")
#     if frase.strip():
#         vb = R["modelos"][("bow", "nb")]["pipe"].named_steps["vec"]
#         vt = R["modelos"][("tfidf", "nb")]["pipe"].named_steps["vec"]
#         vc = R["modelos"][("char", "nb")]["pipe"].named_steps["vec"]
#         tokens = vt.build_analyzer()(frase)
#         c1, c2, c3 = st.columns(3)
#         c1.markdown("**① Original**"); c1.code(frase, language=None)
#         c2.markdown("**② Normalizada**"); c2.code(normalizar(frase), language=None)
#         c3.markdown("**③ Tokens (após stop-words)**"); c3.code(" | ".join(tokens) or "(vazio)", language=None)

#         xb, xt = vb.transform([frase]), vt.transform([frase])
#         linhas = []
#         for termo in dict.fromkeys(tokens):
#             if termo in vt.vocabulary_:
#                 j = vt.vocabulary_[termo]
#                 linhas.append({"termo": termo, "BoW: contagem": str(int(xb[0, vb.vocabulary_[termo]])),
#                                "IDF": f"{float(vt.idf_[j]):.3f}", "TF-IDF (normalizado)": f"{float(xt[0, j]):.3f}"})
#             else:
#                 linhas.append({"termo": termo, "BoW: contagem": "—", "IDF": "—", "TF-IDF (normalizado)": "fora do vocabulário"})
#         st.markdown("**④ Vectores de palavras**")
#         st.dataframe(pd.DataFrame(linhas), width="stretch", hide_index=True)

#         xc = vc.transform([frase])
#         st.markdown("**⑤ N-gramas de caracteres (3 a 5 letras dentro de cada palavra)**")
#         st.code(" ".join(g.replace(" ", "_") for g in sorted(set(vc.build_analyzer()(frase)))[:60])
#                 + (" …" if xc.nnz > 60 else ""), language=None)
#         st.caption("`_` marca o início ou o fim de uma palavra (o vectorizador `char_wb` acrescenta um espaço de cada lado).")
#         k1, k2, k3 = st.columns(3)
#         k1.metric("Dimensões · palavras", len(vt.vocabulary_))
#         k2.metric("Dimensões · caracteres", len(vc.vocabulary_))
#         k3.metric("Usadas por esta frase (caracteres)", xc.nnz)

#     st.markdown("##### Gralhas e abreviaturas: palavras vs caracteres")
#     g1, g2 = st.columns(2)
#     certa = g1.text_input("Palavra correcta", "matricula")
#     errada = g2.text_input("Com gralha / abreviada", "matricla")
#     if certa.strip() and errada.strip():
#         vt = R["modelos"][("tfidf", "nb")]["pipe"].named_steps["vec"]
#         comuns = sorted(ngramas_caracteres(certa) & ngramas_caracteres(errada))
#         a, b = st.columns(2)
#         a.metric("N-gramas de caracteres em comum", f"{ngramas_partilhados(certa, errada):.0%}")
#         b.metric(f"'{normalizar(errada)}' no vocabulário de palavras?",
#                  "sim" if normalizar(errada) in vt.vocabulary_ else "não — é ignorada")
#         st.caption("Em comum: " + (" ".join(f"`{c}`" for c in comuns) or "nenhum"))
#         st.info("Para o modelo de **palavras**, uma gralha é uma palavra nova e desconhecida. "
#                 "Para o modelo de **caracteres**, a palavra com gralha ainda partilha muitos pedaços com a correcta, "
#                 "por isso continua a dar pistas ao classificador.")

#     ka, kb, kc = st.columns(3)
#     ka.info("**BoW** → conta quantas vezes cada palavra aparece.")
#     kb.info("**TF-IDF (palavras)** → TF × IDF, normalizado (norma L2). Palavras raras pesam mais.")
#     kc.info("**TF-IDF (caracteres)** → o mesmo, mas sobre pedaços de 3–5 letras. Resiste a gralhas.")

# # ------------------------------------------------------------------------------
# # 3 — COMPARAÇÃO
# # ------------------------------------------------------------------------------
# with tabs[2]:
#     M = R["modelos"]
#     mb, mm = M[BASELINE], M[R["melhor"]]
#     st.markdown(f"#### Versão original ({nome_modelo(BASELINE)}) vs melhor ({nome_modelo(R['melhor'])})")
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("F1 CV · original", pct(mb["cv_f1"][0]), help="Validação cruzada agrupada 5 folds × 3 sementes.")
#     c2.metric("F1 CV · melhor", pct(mm["cv_f1"][0]), f"{(mm['cv_f1'][0] - mb['cv_f1'][0]) * 100:+.1f} p.p.")
#     c3.metric("ECE · original", f"{mb['calib']['ece']:.3f}", help="Erro de calibração: 0 = confiança perfeita.")
#     c4.metric("ECE · melhor", f"{mm['calib']['ece']:.3f}", f"{mm['calib']['ece'] - mb['calib']['ece']:+.3f}", delta_color="inverse")

#     st.markdown("#### Tabela de resultados no ficheiro de referência")
#     tabela = tabela_resultados(R)
#     st.dataframe(tabela, width="stretch", hide_index=True)
#     st.caption(f"Semente {seed} · teste {prop:.0%} agrupado por texto e estratificado por categoria · "
#                "CV: StratifiedGroupKFold 5 folds × 3 sementes.")
#     st.download_button("Descarregar tabela (CSV)", tabela.to_csv(index=False).encode("utf-8"),
#                        "resultados_corpus_referencia.csv", "text/csv")

#     cH, cL = st.columns([3, 2])
#     with cH:
#         st.markdown("**F1 macro na validação cruzada**")
#         heat = pd.DataFrame([{"representação": REPRESENTACOES[r], "classificador": CLASSIFICADORES[c],
#                               "F1": m["cv_f1"][0], "dp": m["cv_f1"][1]} for (r, c), m in M.items()])
#         base = alt.Chart(heat).encode(x=alt.X("classificador:N", title=None, axis=alt.Axis(labelAngle=0)),
#                                       y=alt.Y("representação:N", title=None, sort=list(REPRESENTACOES.values())))
#         st.altair_chart((base.mark_rect().encode(color=alt.Color("F1:Q", scale=alt.Scale(scheme="blues"), legend=None))
#                          + base.mark_text(fontSize=14).encode(
#                              text=alt.Text("F1:Q", format=".1%"),
#                              color=alt.condition(alt.datum.F1 > heat["F1"].mean(), alt.value("white"), alt.value("black"))))
#                         .properties(height=200), width="stretch")
#     with cL:
#         # Leitura automática: efeito da representação vs efeito do classificador
#         f1 = {k: m["cv_f1"][0] for k, m in M.items()}
#         dp = max(m["cv_f1"][1] for m in M.values())
#         por_rep = {r: np.mean([f1[(r, c)] for c in CLASSIFICADORES]) for r in REPRESENTACOES}
#         por_clf = {c: np.mean([f1[(r, c)] for r in REPRESENTACOES]) for c in CLASSIFICADORES}
#         ef_rep = max(por_rep.values()) - min(por_rep.values())
#         ef_clf = max(por_clf.values()) - min(por_clf.values())
#         st.markdown("**Leitura**")
#         st.write(f"- Efeito da **representação** (média sobre classificadores): {ef_rep * 100:.1f} p.p.; "
#                  f"melhor: *{REPRESENTACOES[max(por_rep, key=por_rep.get)]}*.")
#         st.write(f"- Efeito do **classificador** (média sobre representações): {ef_clf * 100:.1f} p.p.; "
#                  f"melhor: *{CLASSIFICADORES[max(por_clf, key=por_clf.get)]}*.")
#         st.write(f"- Desvio-padrão entre folds: até {dp * 100:.1f} p.p. Diferenças menores do que isto "
#                  "**não são conclusivas** com um corpus deste tamanho.")

#     st.markdown("#### Calibração: a confiança corresponde à taxa de acerto?")
#     cal = pd.concat([M[k]["calib"]["tabela"].assign(modelo=nome_modelo(k)) for k in (BASELINE, R["melhor"])])
#     diag = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
#     st.altair_chart((alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
#                      + alt.Chart(cal).mark_line(point=True).encode(
#                          x=alt.X("confiança média:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
#                          y=alt.Y("exatidão:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
#                          color=alt.Color("modelo:N", scale=alt.Scale(range=PALETA[:2]), legend=alt.Legend(orient="top", title=None)),
#                          tooltip=["modelo", alt.Tooltip("confiança média:Q", format=".0%"),
#                                   alt.Tooltip("exatidão:Q", format=".0%"), "n"]))
#                     .properties(height=260), width="stretch")
#     st.caption("Na diagonal tracejada, \"80 % de confiança\" acerta 80 % das vezes. Abaixo dela, o modelo é "
#                "**sobreconfiante**. O Naive Bayes assume independência entre palavras e, por isso, "
#                "acumula evidência repetida.")

#     st.markdown("#### Detalhe de uma combinação")
#     escolha = st.selectbox("Combinação", list(M), index=list(M).index(R["melhor"]), format_func=nome_modelo)
#     m = M[escolha]
#     g1, g2 = st.columns(2)
#     g1.altair_chart(grafico_cm(m["cm"], classes, "Matriz de confusão"), width="stretch")
#     f1c = pd.DataFrame([{"categoria": c, "F1": m["por_classe"][c]["f1-score"], "suporte": int(m["por_classe"][c]["support"])}
#                         for c in classes])
#     g2.altair_chart(alt.Chart(f1c).mark_bar().encode(
#         x=alt.X("categoria:N", title=None, axis=alt.Axis(labelAngle=0)),
#         y=alt.Y("F1:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
#         color=alt.Color("categoria:N", scale=alt.Scale(domain=classes, range=PALETA[:len(classes)]), legend=None),
#         tooltip=["categoria", alt.Tooltip("F1:Q", format=".1%"), "suporte"]).properties(title="F1 por categoria", height=260),
#         width="stretch")

#     with st.expander("Fuga de dados: porque não usamos um split aleatório simples"):
#         st.write(f"O corpus tem **{df['mensagem'].map(normalizar).nunique()} textos distintos** em **{len(df)} linhas**. "
#                  "Num split aleatório, o mesmo texto cai no treino e no teste e o modelo só 'decora'. "
#                  "Aqui o split agrupa por texto **e** mantém as proporções das categorias (`StratifiedGroupKFold`).")
#         st.dataframe(pd.DataFrame([{"Combinação": nome_modelo(k), "Exatidão com fuga": pct(v["fuga_acc"]),
#                                     "Exatidão sem fuga": pct(v["acc"])} for k, v in M.items()]),
#                      width="stretch", hide_index=True)

#     with st.expander("Mensagens de teste onde esta combinação erra"):
#         t = teste.assign(prevista=m["pred"], confiança=m["proba"].max(axis=1).round(2))
#         st.dataframe(t[t["prevista"] != t["categoria"]][["mensagem", "categoria", "prevista", "confiança"]],
#                      width="stretch", hide_index=True)

# # ------------------------------------------------------------------------------
# # 4 — URGÊNCIA & FILA
# # ------------------------------------------------------------------------------
# with tabs[3]:
#     if not TEM_URG:
#         st.info("O corpus carregado não tem a coluna `urgente` (0/1).")
#     else:
#         ub = U["base"]
#         mu = metricas_urgencia(ub["y_teste"], ub["p_teste"], limiar_urg)
#         st.markdown(f"Segundo alvo: prever `urgente` (TF-IDF de palavras + Naive Bayes). Limiar actual: **{limiar_urg:.0%}**.")
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Revocação dos urgentes", pct(mu["rec"]), help="Dos urgentes reais, quantos o agente apanha.")
#         c2.metric("Precisão", pct(mu["prec"]), help="Dos marcados urgentes, quantos o são.")
#         c3.metric("F1 (urgente)", pct(mu["f1"]))
#         c4.metric("Alarmes", f"{mu['alarmes']} / {len(ub['y_teste'])}")

#         a, b = st.columns(2)
#         a.altair_chart(grafico_cm(mu["cm"], ["normal", "urgente"], "Matriz de confusão (urgência)"), width="stretch")
#         curva = curva_limiar(ub["y_teste"], ub["p_teste"]).melt("limiar", var_name="métrica", value_name="valor")
#         b.altair_chart((alt.Chart(curva).mark_line(point=True).encode(
#             x=alt.X("limiar:Q", axis=alt.Axis(format="%")), y=alt.Y("valor:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
#             color=alt.Color("métrica:N", scale=alt.Scale(range=PALETA[:2]), legend=alt.Legend(orient="top", title=None)))
#             + alt.Chart(pd.DataFrame({"limiar": [limiar_urg]})).mark_rule(strokeDash=[4, 4], color="#94a3b8").encode(x="limiar:Q"))
#             .properties(title="Precisão vs revocação por limiar", height=260), width="stretch")
#         n_contra = int((df.groupby(df["mensagem"].map(normalizar))["urgente"].nunique() > 1).sum())
#         st.caption(f"Atenção: **{n_contra} textos** aparecem no corpus marcados como urgentes e não urgentes. "
#                    "O rótulo é uma decisão humana histórica, não uma verdade objectiva, e isso limita o máximo atingível.")

#         st.markdown("#### Desempenho do agente (categoria + urgência) no conjunto de teste")
#         mm = R["modelos"][R["melhor"]]
#         conf_t = mm["proba"].max(axis=1)
#         da = desempenho_agente(teste["categoria"], mm["pred"], conf_t, ub["y_teste"], ub["p_teste"], limiar, limiar_urg)
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Utilidade média", f"{da['utilidade']:+.2f}")
#         c2.metric("Triagem humana", pct(da["taxa_humana"]))
#         c3.metric("Acerto nas encaminhadas", pct(da["exatidao_encaminhadas"]))
#         c4.metric("Urgentes perdidos", f"{da['urgentes_perdidos']} / {da['urgentes']}")
#         cu = curva_utilidade(teste["categoria"], mm["pred"], conf_t, ub["y_teste"], ub["p_teste"], limiar_urg)
#         st.altair_chart((alt.Chart(cu).mark_line(point=True).encode(
#             x=alt.X("limiar:Q", title="limiar de confiança", axis=alt.Axis(format="%")),
#             y=alt.Y("utilidade:Q", title="utilidade média"),
#             tooltip=[alt.Tooltip("limiar:Q", format=".0%"), alt.Tooltip("utilidade:Q", format="+.2f"),
#                      alt.Tooltip("triagem humana:Q", format=".0%")])
#             + alt.Chart(pd.DataFrame({"limiar": [limiar]})).mark_rule(strokeDash=[4, 4], color="#94a3b8").encode(x="limiar:Q"))
#             .properties(height=220), width="stretch")
#         melhor_lim = cu.loc[cu["utilidade"].idxmax(), "limiar"]
#         st.caption(f"Utilidade: certo {UTILIDADE['certo']:+}, errado {UTILIDADE['errado']:+}, humano {UTILIDADE['humano']:+}, "
#                    f"urgente perdido {UTILIDADE['urgente_perdido']:+}. O limiar que maximiza a utilidade é **{melhor_lim:.0%}** "
#                    "(agente baseado em utilidade, tema 1).")

#         st.markdown("#### Fila simulada (15 mensagens de teste)")
#         amostra = teste.sample(min(15, len(teste)), random_state=seed).copy()
#         linhas = []
#         for _, row in amostra.iterrows():
#             _, pr, cl, _ = prever(mm["pipe"], row["mensagem"])
#             pu = prob_urgente(ub["pipe"], row["mensagem"])
#             dd = decidir(cl[np.argmax(pr)], float(pr.max()), pu, limiar, limiar_urg)
#             linhas.append({"prioridade": dd.prioridade, "p_urgente": round(pu, 2), "destino": dd.destino,
#                            "mensagem": row["mensagem"], "urgente (rótulo)": int(row["urgente"]),
#                            "categoria (rótulo)": row["categoria"]})
#         st.dataframe(ordenar_fila(pd.DataFrame(linhas)), width="stretch", hide_index=True)

# # ------------------------------------------------------------------------------
# # 5 — IA RESPONSÁVEL
# # ------------------------------------------------------------------------------
# with tabs[4]:
#     st.markdown("#### Privacidade: anonimização antes de classificar")
#     demo = st.text_input("Experimente", "sou o estudante 20231234, liguem para +258 84 123 4567 ou joao.m@exemplo.mz")
#     mascarado, enc = anonimizar(demo)
#     st.code(mascarado, language=None)
#     if enc:
#         st.dataframe(pd.DataFrame(enc, columns=["tipo", "valor encontrado"]), width="stretch", hide_index=True)
#     st.caption("Detecta email, BI (12 dígitos + letra), telemóvel moçambicano (82–87, com ou sem +258) e números longos "
#                "(NUIT, nº de estudante). **Limitação:** não detecta nomes próprios. A mensagem original nunca é guardada.")

#     if TEM_REG:
#         st.markdown("#### Viés na categoria: o modelo acerta igual em formal e informal?")
#         sub = pd.concat([metricas_por_subgrupo(teste, R["modelos"][k]["pred"]).assign(modelo=nome_modelo(k))
#                          for k in (BASELINE, R["melhor"])])
#         st.dataframe(sub.style.format({"exatidão": "{:.1%}", "F1 macro": "{:.1%}"}), width="stretch", hide_index=True)

#     if TEM_URG and TEM_REG:
#         st.markdown("#### Viés na urgência")
#         st.markdown("**1 · No rótulo histórico**: taxa de mensagens marcadas urgentes, *dentro da mesma categoria*")
#         tx = taxa_urgencia_a_categoria_igual(df).reset_index().melt("categoria", var_name="registo", value_name="taxa")
#         st.altair_chart(alt.Chart(tx).mark_bar().encode(
#             x=alt.X("categoria:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="registo:N",
#             y=alt.Y("taxa:Q", axis=alt.Axis(format="%"), title="marcadas urgentes"),
#             color=alt.Color("registo:N", scale=alt.Scale(range=PALETA[:2]), legend=alt.Legend(orient="top", title=None)),
#             tooltip=["categoria", "registo", alt.Tooltip("taxa:Q", format=".0%")]).properties(height=220), width="stretch")
#         st.caption("Em todas as categorias, as mensagens **formais** foram marcadas urgentes mais vezes do que as informais. "
#                    "Numa categoria igual, a diferença pode vir da *forma de escrever* e não da urgência real.")

#         st.markdown("**2 · No modelo, e duas medidas correctivas testadas**")
#         ub = U["base"]
#         lims = limiares_por_subgrupo(df.iloc[R["idx_treino"]]["registo"], df.iloc[R["idx_treino"]]["urgente"], U["p_oof"])
#         lim_linha = teste["registo"].map(lims).to_numpy()
#         estrategias = {
#             "Sem correcção": (ub["p_teste"], limiar_urg),
#             "Reponderação (pré-processamento)": (U["reponderado"]["p_teste"], limiar_urg),
#             "Limiar por subgrupo (pós-processamento)": (ub["p_teste"], lim_linha),
#         }
#         res = pd.concat([metricas_urgencia_por_subgrupo(teste, p, l).assign(estratégia=nome)
#                          for nome, (p, l) in estrategias.items()])
#         res = res[["estratégia", "registo", "n", "urgentes (rótulo)", "urgentes (previsto)", "revocação (TPR)", "falsos alarmes (FPR)"]]
#         st.dataframe(res.style.format({c: "{:.0%}" for c in res.columns[3:]}), width="stretch", hide_index=True)
#         gaps = res.groupby("estratégia", sort=False)["revocação (TPR)"].agg(lambda s: s.max() - s.min())
#         st.write("Diferença de revocação entre registos (quanto menor, mais justo): "
#                  + " · ".join(f"**{k}**: {v * 100:.0f} p.p." for k, v in gaps.items()))
#         st.caption(f"Limiares por subgrupo escolhidos com previsões fora-da-amostra do **treino** (sem tocar no teste), "
#                    f"para apanhar ≥ 70 % dos urgentes: " + ", ".join(f"{k} {v:.0%}" for k, v in lims.items()) + ". "
#                    "A reponderação iguala as taxas de base no treino, mas não corrige um viés que está nas palavras.")

#     st.markdown("#### Enquadramento")
#     st.markdown(
#         "- Os rótulos `categoria` e `urgente` são **decisões humanas históricas**: o modelo reproduz-lhes as virtudes e os defeitos.\n"
#         "- O agente **não decide sozinho** quando tem pouca confiança: envia para triagem humana (supervisão humana).\n"
#         "- Transparência: cada decisão mostra os termos que mais pesaram.\n"
#         "- Privacidade: minimização de dados (só o texto mascarado é processado). Confrontar com a Constituição da "
#         "República de Moçambique (art. 71.º, utilização da informática e dados pessoais) e com a Lei n.º 3/2017 "
#         "(Transacções Electrónicas). *Verifiquem o texto legal antes de o citar no artigo.*")

# # ------------------------------------------------------------------------------
# # 6 — CORPUS
# # ------------------------------------------------------------------------------
# with tabs[5]:
#     grupos_txt = df["mensagem"].map(normalizar)
#     multi = int((df.groupby(grupos_txt)["categoria"].nunique() > 1).sum())
#     k1, k2, k3, k4 = st.columns(4)
#     k1.metric("Mensagens", len(df))
#     k2.metric("Textos distintos", grupos_txt.nunique())
#     k3.metric("Textos com >1 categoria", multi, help="Mesma frase com categorias diferentes → ambiguidade.")
#     k4.metric("Comprimento médio", f"{df['mensagem'].str.split().str.len().mean():.1f} palavras")
#     if multi:
#         st.warning(f"**{multi} textos** aparecem com mais de uma categoria (ex.: *'alguem me pode ajudar pfv'*). "
#                    "Isto impõe um tecto de exatidão: nenhum classificador acerta todas.")

#     cA, cB = st.columns(2)
#     dist = df["categoria"].value_counts().rename_axis("categoria").reset_index(name="n")
#     cA.markdown("**Distribuição por categoria**")
#     cA.altair_chart(alt.Chart(dist).mark_bar().encode(
#         x=alt.X("n:Q", title="mensagens"), y=alt.Y("categoria:N", sort="-x", title=None),
#         color=alt.Color("categoria:N", scale=alt.Scale(domain=classes, range=PALETA[:len(classes)]), legend=None)
#     ).properties(height=200), width="stretch")
#     if TEM_REG:
#         cr = df.groupby(["categoria", "registo"]).size().reset_index(name="n")
#         cB.markdown("**Registo por categoria**")
#         cB.altair_chart(alt.Chart(cr).mark_bar().encode(
#             x=alt.X("categoria:N", title=None, axis=alt.Axis(labelAngle=0)), y="n:Q",
#             color=alt.Color("registo:N", scale=alt.Scale(range=PALETA[:2]), legend=alt.Legend(orient="top", title=None))
#         ).properties(height=200), width="stretch")

#     st.markdown("**Palavras mais características de cada categoria** (TF-IDF + Naive Bayes)")
#     p = R["modelos"][("tfidf", "nb")]["pipe"]
#     flp = p.named_steps["clf"].feature_log_prob_
#     nomes = p.named_steps["vec"].get_feature_names_out()
#     disc = flp - flp.mean(axis=0)
#     st.dataframe(pd.DataFrame({c: list(nomes[np.argsort(disc[i])[::-1][:8]])
#                                for i, c in enumerate(p.named_steps["clf"].classes_)}), width="stretch", hide_index=True)

#     st.markdown("**Explorar mensagens**")
#     f1_, f2_ = st.columns([1, 2])
#     cat_f = f1_.multiselect("Categoria", classes, default=classes)
#     busca = f2_.text_input("Procurar texto", placeholder="ex.: propina")
#     vista = df[df["categoria"].isin(cat_f)]
#     if busca:
#         vista = vista[vista["mensagem"].map(normalizar).str.contains(normalizar(busca), regex=False)]
#     st.dataframe(vista, width="stretch", hide_index=True, height=300)
#     st.caption(f"{len(vista)} de {len(df)} mensagens")

# # ------------------------------------------------------------------------------
# # 7 — TEORIA & PEAS
# # ------------------------------------------------------------------------------
# with tabs[6]:
#     a, b_ = st.columns(2)
#     with a:
#         st.markdown("### Quadro PEAS do agente")
#         st.markdown(
#             """
# | | |
# |---|---|
# | **P** Desempenho | Utilidade média (certo +1, errado −2, humano −0,5, urgente perdido −3); F1 macro; revocação dos urgentes; diferença de revocação entre registos |
# | **E** Ambiente | Fluxo de mensagens curtas em português, formais e informais, com gralhas, abreviaturas e dados pessoais |
# | **A** Actuadores | Encaminhar ao serviço; marcar prioridade ALTA/NORMAL; enviar a triagem humana; mascarar dados pessoais |
# | **S** Sensores | Texto da mensagem |
#             """)
#         st.caption("Tipo de agente: **baseado em utilidade**, com componente de **aprendizagem**. Escolhe o limiar "
#                    "e a prioridade que maximizam a utilidade esperada, e não apenas a classe mais provável.")
#     with b_:
#         st.markdown("### Fórmulas")
#         st.latex(r"\text{TF-IDF}(t,d)=\text{TF}(t,d)\times\left(\ln\frac{1+|D|}{1+df(t)}+1\right)")
#         st.latex(r"\text{NB: } P(C_k\mid x)\propto P(C_k)\prod_i P(x_i\mid C_k),\quad P(x_i\mid C_k)=\frac{N_{ik}+\alpha}{N_k+\alpha|V|}")
#         st.latex(r"\text{RL: } P(C_k\mid x)=\frac{e^{w_k\cdot x+b_k}}{\sum_j e^{w_j\cdot x+b_j}}")
#         st.latex(r"\text{ECE}=\sum_b \frac{n_b}{N}\,\big|\text{exatidão}_b-\text{confiança}_b\big|")
#         st.latex(r"\text{Reponderação: } w(g,y)=\frac{P(g)\,P(y)}{P(g,y)}")

#     st.markdown("### Generativo vs discriminativo")
#     st.write("O **Naive Bayes** modela como cada classe *gera* palavras e assume que são independentes dada a classe. "
#              "A **Regressão Logística** aprende directamente a fronteira entre classes e não precisa dessa hipótese. "
#              "Com palavras correlacionadas (*'propina'* e *'pagamento'* aparecem juntas), o NB conta a mesma evidência "
#              "duas vezes e fica sobreconfiante. Isso vê-se no ECE e no diagrama de calibração.")
#     st.markdown("### Onde a independência condicional falha")
#     st.write("Em *'nao validou'* ou *'propina paga mas inscricao bloqueada'*, a ordem e a negação mudam o sentido, "
#              "mas o modelo vê palavras soltas. Os **bigramas** (barra lateral) atenuam o problema. Os n-gramas de "
#              "caracteres resolvem outro: gralhas e abreviaturas.")
