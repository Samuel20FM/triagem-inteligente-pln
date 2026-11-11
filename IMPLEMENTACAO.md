# Implementação — o que mudou no protótipo do Grupo 13 e porquê

**Tema do grupo:** 13 — Processamento de Linguagem Natural: representação e classificação.
**Data:** 11/11/2026 · **Base:** versão anterior (BoW vs TF-IDF + Naive Bayes, um só ficheiro `app.py`).

## Resumo

O tema 13 pede para *preparar texto, representá-lo (saco de palavras ou TF-IDF), classificar e comparar duas
representações no corpus de referência*. A versão anterior fazia isso. Esta versão:

1. **acrescenta uma terceira representação** (n-gramas de caracteres), que é o centro do tema 13;
2. traz quatro temas de outros grupos, **cada um ao serviço da pergunta do tema 13** ("qual a melhor representação, e porquê?");
3. **corrige 9 erros** encontrados na instalação;
4. separa o código em módulos, cada um explicável por um membro, e passa de 27 para **67 testes**;
5. tem **tema claro e escuro** e deixou de ter elementos fora do contexto.

| # | Mudança | Tema de origem | Ficheiro |
|---|---|---|---|
| 1 | TF-IDF de n-gramas de caracteres | **13** (representação) | `representacoes.py` |
| 2 | Regressão Logística ao lado do Naive Bayes | 8 / 9 (classificação supervisionada) | `classificadores.py` |
| 3 | Calibração da confiança (ECE, Brier, diagrama) | 9 (avaliação) | `avaliacao.py` |
| 4 | Divisão agrupada **e** estratificada | 9 (treino/teste) — correcção | `avaliacao.py` |
| 5 | Segundo alvo: urgência | 9 + 1 | `urgencia.py` |
| 6 | Agente baseado em utilidade (destino + prioridade + triagem humana) | 1 (agentes, PEAS) | `agente.py` |
| 7 | Anonimização de dados pessoais | 15 (privacidade) | `responsavel.py` |
| 8 | Auditoria de viés por registo + 2 medidas correctivas | 15 (viés) | `responsavel.py` |
| 9 | Interface: 7 separadores, tema claro/escuro | — | `app.py`, `.streamlit/config.toml` |

---

## 1. Erros corrigidos

| Erro encontrado | Causa | Correcção |
|---|---|---|
| A proporção de teste não era respeitada (16–29 % em vez de 20 %) | `GroupShuffleSplit` sorteia 20 % dos *textos distintos*, e alguns repetem-se 15 vezes | `dividir()` usa o 1.º fold de `StratifiedGroupKFold(k = 1/proporção)` → teste ≈ 20 % das linhas |
| Teste desequilibrado (46 % financeiro, quando no corpus é 29 %) | A divisão não era estratificada | O mesmo `StratifiedGroupKFold` mantém as proporções das categorias |
| A app rebentava com um CSV de 1 categoria | `contribuicoes()` fazia `ordem[1]` | `validar_corpus()` recusa esse CSV; `contribuicoes()` devolve vazio com < 2 classes |
| CSV pequeno → exatidão 0 % sem aviso, ou `ValueError` | Sem validação de tamanho | `validar_corpus()` exige ≥ 2 categorias e ≥ 5 textos distintos por categoria |
| CSV com `;` → `KeyError` | Separador fixo em `,` | `pd.read_csv(sep=None, engine="python")` detecta o separador |
| Mensagens vazias viravam "nan" (pandas 2) | `astype(str)` antes de `dropna` | `dropna` primeiro, depois remover textos vazios |
| O tema visual não era aplicado | `config.toml` fora da pasta `.streamlit/` | Movido, e agora com tema claro e escuro |
| Títulos "separador )" no manual | Ícones apagados | Manual reescrito |
| Fonte do Google: a app "offline" precisava de internet | `@import` de Google Fonts | Removido |
| *(desempenho)* Treino de ~40 s | Duas cópias do OpenBLAS (numpy + scipy) a competir pelos 8 núcleos | `@threadpool_limits.wrap(limits=1)` → ~4 s |

**Efeito na defesa:** com a divisão corrigida, o BoW + NB original passa de 73,9 % para 63,4 % no hold-out. O valor
antigo era inflacionado por um teste com muito "financeiro", a categoria mais fácil. A validação cruzada quase não
muda (69,3 % de F1), o que confirma que é a medida em que devem confiar.

---

## 2. Melhorias, uma a uma

### 2.1 Terceira representação: TF-IDF de n-gramas de caracteres — *tema 13*

**O que é.** Em vez de partir a frase em palavras, parte cada palavra em pedaços de 3 a 5 letras.
`matricula` → `_ma`, `mat`, `atr`, `tri`, …, `ula_`. O `_` marca o início ou o fim da palavra
(`analyzer="char_wb"` no scikit-learn). Depois aplica TF-IDF sobre esses pedaços.

**Como.** `representacoes.criar_vectorizador("char")` → `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))`.
`ngramas_partilhados(a, b)` calcula a fracção de pedaços em comum (índice de Jaccard), para a demonstração.

**Como ajuda o tema 13.** O Guia diz que o corpus tem "abreviaturas e gralhas propositadas". Para uma representação por
palavras, `matricla` é uma palavra desconhecida e ignorada. Para a representação de caracteres, partilha 41 % dos
pedaços com `matricula`. O tema passa de "comparar duas representações" para **explicar o que cada uma consegue
ver**: BoW vê contagens, TF-IDF vê raridade, caracteres vêem a forma das palavras.

**Resultado.** É a melhor representação em média: F1 na CV de 73,0 % com NB e 78,4 % com RL.
Exemplo `quero trancar a matricla`: o original dá matrícula com 32 % (vai para triagem humana); caracteres + RL dão **87 %**.

**Pergunta provável na defesa:** *"Porque é que o vocabulário passa de 278 para 3140 dimensões e isso não é um problema?"*
Os vectores são esparsos: uma frase usa ~100 dimensões. E a RL regulariza com o parâmetro C.

### 2.2 Regressão Logística ao lado do Naive Bayes — *temas 8 e 9*

**O que é.** Um segundo classificador, **discriminativo**: aprende directamente a fronteira entre classes com uma softmax,
sem assumir que as palavras são independentes. O NB é **generativo** e assume essa independência.

**Como.** `classificadores.criar_pipeline(rep, clf)` combina qualquer representação com qualquer classificador
→ 3 × 2 = **6 combinações**, treinadas e avaliadas em `avaliacao.treinar()`.
A explicação "Porquê?" funciona para os dois. No NB usa `log P(termo|classe)`; na RL usa os coeficientes.
Em ambos, `x_j · (w[c1,j] − w[c2,j])` é a contribuição do termo para a log-odds entre as duas classes mais prováveis.
C = 5 foi escolhido por validação cruzada agrupada (C = 1: 73,6 %; 5: 74,1 %; 10: 73,6 %; 30: 74,2 %, com TF-IDF de palavras).

**Como ajuda o tema 13.** Sem um segundo classificador, não se sabe se uma diferença vem da **representação** ou do
**classificador**. A app calcula as duas coisas separadamente: o efeito da representação (média sobre os
classificadores) é ≈ 3,1 p.p., e o do classificador (média sobre as representações) é ≈ 4,9 p.p. Ambos ficam abaixo do
desvio-padrão (≈ 9 p.p.), e a app diz isso com honestidade.

### 2.3 Calibração — *tema 9*

**O que é.** Mede se "90 % de confiança" acerta mesmo 90 % das vezes.
ECE = Σ (n_b / N) · |exatidão_b − confiança_b|, com 10 intervalos de confiança.

**Como.** `avaliacao.calibracao()` → ECE, Brier e a tabela do diagrama de fiabilidade, no separador Comparação.

**Como ajuda o tema 13.** O agente usa a confiança para decidir a triagem humana, por isso a confiança tem de ser
honesta. O original (BoW + NB) tem ECE 0,213 e dava 94 % a `alguem me pode ajudar pfv`. Caracteres + RL têm ECE 0,121
e dão 47 % → triagem humana. Isto confirma, com números, a limitação da independência condicional que o artigo já discute.

### 2.4 Urgência — *temas 9 e 1*

**O que é.** O Guia indica duas tarefas possíveis no corpus: categoria **ou** urgência. Agora fazemos as duas.
O modelo de urgência é TF-IDF de palavras + NB (F1 macro na CV ≈ 75 %, melhor do que RL com palavras, 73 %, ou com caracteres, 71 %).

**Como.** `urgencia.treinar_urgencia()` usa **a mesma divisão** da categoria, para não haver fuga.
`curva_limiar()` mostra o compromisso entre precisão e revocação.

**Como ajuda o tema 13.** Mostra que **a mesma representação de texto** serve uma segunda tarefa, e que a dificuldade
depende do rótulo: 54 textos têm rótulos de urgência contraditórios, por isso o tecto é baixo.

### 2.5 Agente baseado em utilidade — *tema 1*

**O que é.** O classificador passa a ser a "percepção" de um agente que **age**:
- confiança < limiar → **triagem humana**;
- caso contrário, encaminha ao serviço com prioridade **ALTA** (P(urgente) ≥ limiar) ou **NORMAL**.

A medida de desempenho é uma utilidade explícita: certo +1, errado −2, triagem humana −0,5, urgente perdido −3.

**Como.** `agente.decidir()`, `desempenho_agente()`, `curva_utilidade()` e `ordenar_fila()`.

**Como ajuda o tema 13.** O programa pede "modelação do agente e quadro PEAS, quando aplicável". Antes, o PEAS era só
texto. Agora o **P** é calculado (utilidade média +0,05; 12,7 % para triagem humana; 87,2 % de acerto nas mensagens
encaminhadas) e o **A** tem três actuadores reais. O agente passa de "reactivo" a **baseado em utilidade**.
A curva mostra que, com estes custos, compensa encaminhar mesmo com pouca confiança. Se um erro custasse mais, o
limiar óptimo subiria. É uma boa pergunta para a turma.

### 2.6 Anonimização — *tema 15 (privacidade)*

**O que é.** Antes de classificar, `responsavel.anonimizar()` troca por marcadores: emails, BI (12 dígitos + letra),
telemóveis moçambicanos (82–87, com ou sem +258) e números longos (NUIT, nº de estudante).

**Como ajuda o tema 13.** É **preparação do texto**, a primeira etapa do tema 13, com uma razão ética: minimização de
dados. O corpus de referência não tem dados pessoais, por isso os resultados da tabela não mudam; a protecção actua
sobre o que o utente escreve na app. Limitação declarada: não detecta
nomes próprios, o que exigiria um modelo de reconhecimento de entidades.

### 2.7 Auditoria de viés por registo — *tema 15 (viés)*

**O que é.** O corpus tem a coluna `registo` (formal/informal). Medimos três coisas:
1. **Categoria:** o original acerta 42,6 % das formais contra 74,7 % das informais. O melhor modelo reduz a diferença (78,7 % vs 90,8 %).
2. **Rótulo de urgência:** dentro de *cada* categoria, as formais foram marcadas urgentes mais vezes (≈ 44 % vs 28 %).
3. **Modelo de urgência:** ao contrário do rótulo, o modelo **apanha menos** urgentes formais (revocação 32 % contra 78 %),
   porque os pedidos formais urgentes usam palavras corteses e não "ja", "!!" ou "hoje".

**Duas medidas correctivas testadas** (o tema 15 pede "propor e testar"):
- **Reponderação** (Kamiran & Calders, 2012), w(g, y) = P(g)·P(y)/P(g, y): **não resolve** (diferença 46 → 50 p.p.).
  Iguala as taxas de base, mas o viés está nas palavras.
- **Limiar por subgrupo** (Hardt et al., 2016), escolhido com previsões fora-da-amostra **do treino**:
  diferença **46 → 8 p.p.** (revocação nos formais 32 % → 74 %). Cuidado: assenta em poucos textos formais urgentes distintos.

**Como ajuda o tema 13.** O registo é uma propriedade **da linguagem**. Mostrar que a representação de texto trata
de forma diferente quem escreve formal ou informalmente é PLN aplicado com análise crítica, que é o que o objectivo
geral da cadeira pede.

---

## 3. Interface

- **Tema claro e escuro** (`.streamlit/config.toml` com `[theme.light]` e `[theme.dark]`), escolhido no menu ⋮ → Settings → Theme.
  O CSS deixou de ter cores de texto fixas: usa transparências, que funcionam nos dois temas. O menu ⋮ voltou a estar visível.
- **7 separadores:** Classificar · Como o texto vira números · Comparação · Urgência & fila · IA responsável · Corpus · Teoria & PEAS.
- **Classificar:** decisão completa num cartão (categoria, confiança, P(urgente), prioridade, destino), grelha com as 6 combinações.
- **Retirado por estar fora do contexto:** fonte externa do Google; bloco "Executar localmente" no separador de teoria;
  tempo de inferência nos cartões (fica na tabela); o rádio "modelo usado" (substituído pela escolha representação × classificador).

---

## 4. O que actualizar no artigo e nos slides

| Secção do artigo | Acrescentar |
|---|---|
| Resumo / palavras-chave | n-gramas de caracteres; Regressão Logística; calibração |
| 2. Fundamentação | 2.3: n-gramas de caracteres; 2.4: NB (generativo) vs RL (discriminativo) |
| 4. Metodologia | Divisão agrupada e estratificada; CV 5 × 3; 6 combinações |
| 5. Modelação do agente | PEAS com a utilidade; agente baseado em utilidade; triagem humana |
| 7. Resultados | Tabela da secção 4 do README (exportável na app) + ECE + efeito da representação vs do classificador |
| 8. Impacto e limitações | Viés por registo e medidas testadas; anonimização; rótulos contraditórios |
| Referências | Kamiran & Calders (2012) *Data preprocessing techniques for classification without discrimination*; Hardt, Price & Srebro (2016) *Equality of Opportunity in Supervised Learning*; Guo et al. (2017) *On Calibration of Modern Neural Networks* (ECE). **Confirmem cada referência antes de a citar.** |

**Slides sugeridos:** (1) demonstração de `matricla` com palavras vs caracteres; (2) mapa de calor 3 × 2;
(3) diagrama de calibração; (4) caso `alguem me pode ajudar pfv`; (5) viés formal/informal antes e depois.
**Actividade para a turma:** "Se reencaminhar uma mensagem errada custasse −5 em vez de −2, o limiar de confiança
óptimo subia ou descia?" (respondam com o gráfico de utilidade).

## 5. Divisão sugerida para a defesa

| Membro | Domina primeiro | Depois revê |
|---|---|---|
| A | `texto.py`, `representacoes.py`, `classificadores.py` (temas 13, 8) | `agente.py` |
| B | `avaliacao.py`, `urgencia.py`, `responsavel.py` (temas 9, 15) | `app.py` |

Na defesa, qualquer membro pode ser questionado sobre qualquer linha. Corram `pytest -v` e leiam o teste de cada função:
os testes são a forma mais curta de perceber o que cada uma faz.
