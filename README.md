# Triagem Inteligente de Mensagens — Grupo 13

**Disciplina:** Inteligência Artificial 2026 · Universidade Pedagógica de Maputo (FET)
**Tema:** Processamento de Linguagem Natural — representação e classificação
**Autores:** Samuel Massinga & Arney Nhadumbuque · **Docente:** MSc. Sheila A. Sitoe Rangel

Aplicação Streamlit que classifica mensagens curtas de utentes em **financeiro, matrícula, notas ou infraestrutura**.
Sobre o `corpus_referencia.csv` (670 mensagens) **compara três representações de texto** (Bag-of-Words, TF-IDF de
palavras, TF-IDF de n-gramas de caracteres) com **dois classificadores** (Naive Bayes e Regressão Logística).
Usa ainda a coluna `urgente` para dar prioridade às mensagens, mascara dados pessoais e audita o viés.

- O que mudou face à versão anterior, e porquê: **[IMPLEMENTACAO.md](IMPLEMENTACAO.md)**
- Guia para a aula aberta e a defesa fechada: **[DEFESA.md](DEFESA.md)**
- Manual de execução e testes: **[MANUAL_TESTES.md](MANUAL_TESTES.md)**

> O `corpus_referencia.csv` é o ficheiro de referência da turma (dados fictícios), fornecido pela docente. Está
> aqui só para que a app corra; não o alterem.

---

## 1. Como executar

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Tema claro ou escuro:** no canto superior direito, menu ⋮ → *Settings* → *Theme* (Light / Dark / System).

**Testes:** `pytest -v` (67 testes, cerca de 25 s).

## 2. Estrutura

| Ficheiro | Responsabilidade | Tema da cadeira |
|---|---|---|
| `texto.py` | `normalizar()` e stop-words (mantém as negações) | 13 |
| `representacoes.py` | BoW, TF-IDF de palavras, TF-IDF de caracteres (3–5 letras) | 13 |
| `classificadores.py` | Naive Bayes, Regressão Logística, explicação por termo | 8 / 9 |
| `avaliacao.py` | leitura e validação do CSV; divisão sem fuga; 6 combinações; CV; calibração | 9 / 13 |
| `urgencia.py` | segundo alvo `urgente`; métricas; curva de limiar | 9 / 1 |
| `agente.py` | decisão (destino + prioridade + triagem humana), utilidade, fila | 1 |
| `responsavel.py` | anonimização; métricas por subgrupo; reponderação; limiares por subgrupo | 15 |
| `app.py` | só a interface (7 separadores) | — |
| `tests/` | um ficheiro de testes por módulo + interface | — |
| `.streamlit/config.toml` | temas claro e escuro | — |
| `corpus_referencia.csv` | ficheiro de referência da turma (não alterar) | — |

Só o `app.py` importa o Streamlit. Os outros módulos podem ser usados e testados sem a interface.

## 3. Separadores

> **Nota:** de momento a interface mostra **só o separador Classificar**. Os outros seis estão comentados no
> `app.py` (bloco "SEPARADORES 2–7 DESACTIVADOS"); para os repor, basta descomentar esse bloco e a lista completa
> em `st.tabs(...)`. A lógica (comparação, urgência, auditoria…) continua nos módulos e nos testes.

| Separador | Conteúdo |
|---|---|
| **Classificar** | anonimização → categoria, confiança, P(urgente), prioridade e destino; grelha com as 6 combinações; termos que pesaram |
| **Como o texto vira números** | normalização → tokens → BoW/IDF/TF-IDF → n-gramas de caracteres; demonstração com gralhas |
| **Comparação** | tabela de resultados no formato do Guia (exportável em CSV), mapa de F1 na CV, leitura automática, calibração, matriz de confusão, fuga de dados, erros |
| **Urgência & fila** | métricas dos urgentes, curva precisão/revocação, utilidade do agente por limiar, fila simulada |
| **IA responsável** | anonimização, viés por registo (categoria e urgência), duas medidas correctivas testadas, enquadramento |
| **Corpus** | distribuição, ambiguidade, palavras características, explorador |
| **Teoria & PEAS** | PEAS, tipo de agente, fórmulas, generativo vs discriminativo |

## 4. Resultados (parâmetros por omissão)

Unigramas, stop-words removidas, α = 1, C = 5, teste 20 % agrupado e estratificado, **semente 42**, CV 5 folds × 3 sementes.

| Representação | Classificador | Exatidão (hold-out) | F1 macro (hold-out) | F1 macro (CV) | ECE |
|---|---|---|---|---|---|
| Bag-of-Words | Naive Bayes *(versão original)* | 63,4 % | 65,2 % | 69,3 % ± 8,2 | 0,213 |
| Bag-of-Words | Regressão Logística | 78,4 % | 81,6 % | 75,7 % ± 6,5 | 0,166 |
| TF-IDF (palavras) | Naive Bayes | 70,9 % | 74,6 % | 71,2 % ± 9,0 | 0,156 |
| TF-IDF (palavras) | Regressão Logística | 73,9 % | 78,4 % | 74,1 % ± 8,5 | 0,136 |
| TF-IDF (caracteres) | Naive Bayes | 74,6 % | 78,6 % | 73,0 % ± 8,5 | 0,101 |
| **TF-IDF (caracteres)** | **Regressão Logística** | **86,6 %** | **88,7 %** | **78,4 % ± 7,9** | 0,121 |

> **Leitura:** a combinação de caracteres com Regressão Logística é a melhor na CV (+9,1 p.p. sobre a original). Mas o
> desvio-padrão entre folds chega aos 9 p.p., por isso as diferenças entre combinações vizinhas **não são conclusivas**.
> Os valores mudam com os parâmetros: confirmem sempre na app antes de os copiar para o artigo ou os slides.

## 5. Decisões técnicas (para a defesa)

1. **Normalização:** minúsculas, sem acentos, sem pontuação. O corpus não tem acentos.
2. **Negações preservadas:** `nao`, `sem` e `nem` não são stop-words.
3. **Divisão agrupada e estratificada** (`StratifiedGroupKFold`): o mesmo texto nunca aparece no treino e no teste
   (o corpus só tem 197 textos distintos), as proporções das categorias mantêm-se, e o teste fica com ≈ 20 % das linhas.
4. **Validação cruzada 5 × 3** com o vectorizador dentro do `Pipeline`, para que o vocabulário e o IDF só se aprendam no treino.
5. **N-gramas de caracteres (`char_wb`, 3–5):** uma gralha como "matricla" partilha 41 % dos pedaços com "matricula".
   Para o modelo de palavras, é simplesmente uma palavra desconhecida.
6. **Calibração (ECE):** mede se "90 % de confiança" acerta mesmo 90 % das vezes. É importante porque o limiar de confiança decide a triagem humana.
7. **Urgência:** TF-IDF + Naive Bayes, usando a mesma divisão da categoria. 54 textos têm rótulos de urgência contraditórios.
8. **Viés:** em todas as categorias, o rótulo histórico marca mais urgentes formais. Mas o modelo *apanha menos* urgentes formais
   (revocação 32 % contra 78 % nos informais). A reponderação não resolve. Os limiares por subgrupo, escolhidos só com o treino,
   reduzem a diferença de 46 para 8 p.p.
9. **Treino com 1 thread de BLAS** (`threadpoolctl`): baixa o tempo de ~40 s para ~4 s.

## 6. Limitações

- O corpus é pequeno e ambíguo: há 29 textos com mais de uma categoria e 54 com urgência contraditória.
- A anonimização não detecta nomes próprios.
- O hold-out de 134 mensagens é instável; usem a CV para tirar conclusões.
- Os limiares por subgrupo assentam em poucos textos formais urgentes distintos, por isso são frágeis.

## 7. Declaração de uso de IA (modelo para o anexo do artigo)

> **Ferramenta:** Claude (Anthropic). **Parte do trabalho:** reorganização do código em módulos; n-gramas de caracteres,
> Regressão Logística, calibração, modelo de urgência, agente baseado em utilidade e módulo de IA responsável; testes;
> interface; README, IMPLEMENTACAO.md e manual de testes. **Finalidade:** alargar o protótipo e documentá-lo.
> **Verificado pelos autores:** _(preencham: execução da app e dos testes, leitura do código linha a linha,
> conferência dos resultados, das fórmulas e das referências legais)_.

O programa exige que **cada membro** saiba explicar **qualquer linha** do código na defesa.
