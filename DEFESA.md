# Documento de Defesa — Grupo 13

**Tema:** Processamento de Linguagem Natural — representação e classificação
**Artefacto:** Triagem Inteligente de Mensagens de Utentes
**Grupo:** Samuel Massinga · Arney Nhadumbuque · **Docente:** MSc. Sheila António Sitoe Rangel
**Disciplina:** Inteligência Artificial 2026 · UP-Maputo, Faculdade de Engenharia e Tecnologias

> Guia de preparação para a aula aberta (60 min) e a defesa fechada (30 min). Cada secção diz **o que fizemos**,
> **porque** e **como o dizer**. Os números são os da app com os parâmetros por omissão (semente 42); confirmem-nos
> na app antes de os apresentar.

---

## 1. O trabalho numa frase

> *Construímos um agente que lê mensagens curtas dirigidas aos serviços académicos, transforma o texto em números
> de três formas diferentes, classifica-o com dois algoritmos e decide para onde e com que prioridade o
> encaminhar. Medimos qual a representação de texto que funciona melhor, porquê, e com que riscos.*

## 2. Contexto: o problema e o tema 13

### 2.1 O problema social
Os serviços académicos recebem muitas mensagens escritas: pedidos de matrícula, dúvidas de notas, problemas
financeiros, avarias. Classificá-las à mão é lento, e uma mensagem urgente pode ficar perdida na fila.
**Beneficiários:** os estudantes, que recebem resposta mais depressa e do serviço certo, e os funcionários, que
recebem pedidos já triados.

### 2.2 O que o programa pede ao tema 13

| Exigência (Programa e Guia) | Onde está no protótipo |
|---|---|
| Preparar o texto | `texto.normalizar()`: minúsculas, acentos, pontuação; stop-words sem negações; mais anonimização |
| Representar por BoW **ou** TF-IDF | BoW, TF-IDF de palavras **e** TF-IDF de n-gramas de caracteres (`representacoes.py`) |
| Classificar | Naive Bayes e Regressão Logística (`classificadores.py`) |
| **Comparar duas representações no corpus de referência** | 3 representações × 2 classificadores, hold-out + CV 5 × 3 (`avaliacao.py`, separador *Comparação*) |
| Métricas do Grupo 8: exatidão, precisão, revocação, F1 | Tabela exportável no formato do Guia (ficheiro · método · conjunto · métricas) |
| Semente e divisão treino/teste indicadas | Semente 42; teste 20 % agrupado por texto e estratificado por categoria |
| Discutir o tecto razoável | 29 textos com mais de uma categoria; o desvio-padrão da CV ≈ 9 p.p. |
| PEAS, quando aplicável | Agente baseado em utilidade com medida de desempenho calculada (`agente.py`) |
| Demonstração: entradas → processamento → saídas | Separadores *Classificar* e *Como o texto vira números* |

### 2.3 O fluxo completo

```
 mensagem do utente
        │
        ▼
 ① PREPARAÇÃO ─────────── anonimizar (tema 15) → normalizar → tokens
        │
        ▼
 ② REPRESENTAÇÃO ──────── BoW │ TF-IDF palavras │ TF-IDF caracteres        ← NÚCLEO DO TEMA 13
        │
        ▼
 ③ CLASSIFICAÇÃO ──────── Naive Bayes │ Regressão Logística               (temas 8/9)
        │                     → categoria + confiança
        │                     → P(urgente)  (2.º modelo, mesma representação)
        ▼
 ④ DECISÃO DO AGENTE ──── confiança baixa → triagem humana                (tema 1)
        │                  senão → serviço + prioridade ALTA/NORMAL
        ▼
 ⑤ AVALIAÇÃO E AUDITORIA ─ CV agrupada, calibração, viés por registo     (temas 9/15)
```

**Como dizer:** *"O tema 13 são os passos ① e ②. Os outros temas estão lá para responder a perguntas que o
tema 13 levanta: com que classificador? Como sei se o resultado é fiável? Para que serve na prática? É justo?"*

---

## 3. O núcleo: PLN (tema 13)

### 3.1 Preparação do texto
- **Normalizar:** `"Não consigo pagar a PROPINA!"` → `"nao consigo pagar a propina"`. O corpus não tem acentos;
  sem normalizar, "matrícula", escrita por um utente, nunca coincidiria com "matricula" do vocabulário.
- **Stop-words sem negações:** removemos "a", "de", "o"… mas **não** "nao", "sem" e "nem", porque mudam o sentido.
- **Divisão sem fuga:** o corpus tem 670 linhas mas só **197 textos distintos**. Se o mesmo texto ficar no treino e
  no teste, o modelo só decora. Por isso agrupamos por texto.

### 3.2 As três representações

| | Bag-of-Words | TF-IDF (palavras) | TF-IDF (caracteres) |
|---|---|---|---|
| Unidade | palavra | palavra | pedaço de 3–5 letras |
| Valor | contagem | TF × IDF, norma L2 = 1 | TF × IDF, norma L2 = 1 |
| Dimensões | 278 | 278 | 3140 |
| Vê | quantas vezes | se a palavra é rara (discriminativa) | a forma da palavra |
| Não vê | importância, ordem | ordem, gralhas | ordem |

**Exemplo para o slide — gralhas:** `matricla` (com gralha) vs `matricula`
- Palavras: `matricla` não existe no vocabulário → é **ignorada**.
- Caracteres: `_ma`, `mat`, `atr`, `tri`… partilham **41 %** dos pedaços → ainda dá pistas.
- Resultado: `quero trancar a matricla` → BoW + NB **32 %** (triagem humana); caracteres + RL **87 %** matrícula.

**Ligação ao tema 14 (só para comentar):** os modelos de linguagem também partem palavras em pedaços (*subword
tokens*) pela mesma razão: palavras raras ou mal escritas continuam a ser representáveis.

### 3.3 Resultados no ficheiro de referência

Semente 42 · teste 20 % agrupado e estratificado · CV 5 folds × 3 sementes.

| Representação | Classificador | Exatidão (hold-out) | F1 macro (hold-out) | **F1 macro (CV)** | ECE |
|---|---|---|---|---|---|
| Bag-of-Words | Naive Bayes *(versão original)* | 63,4 % | 65,2 % | 69,3 % ± 8,2 | 0,213 |
| Bag-of-Words | Regressão Logística | 78,4 % | 81,6 % | 75,7 % ± 6,5 | 0,166 |
| TF-IDF (palavras) | Naive Bayes | 70,9 % | 74,6 % | 71,2 % ± 9,0 | 0,156 |
| TF-IDF (palavras) | Regressão Logística | 73,9 % | 78,4 % | 74,1 % ± 8,5 | 0,136 |
| TF-IDF (caracteres) | Naive Bayes | 74,6 % | 78,6 % | 73,0 % ± 8,5 | 0,101 |
| **TF-IDF (caracteres)** | **Regressão Logística** | **86,6 %** | **88,7 %** | **78,4 % ± 7,9** | 0,121 |

**Leitura honesta (dizer exactamente isto):**
1. A melhor combinação é caracteres + RL: **+9,1 p.p.** de F1 sobre a versão original.
2. Em média, a representação vale ≈ 3,1 p.p. e o classificador ≈ 4,9 p.p.
3. O desvio-padrão entre folds chega aos 9 p.p., por isso diferenças pequenas **não são conclusivas**.
   O corpus é pequeno e tem 29 textos com etiquetas contraditórias, o que dá um tecto abaixo de 100 %.
4. Confiamos mais na **CV** do que no hold-out: o hold-out é um único sorteio de 134 mensagens.

---

## 4. Como os outros temas se integram

Para cada tema: **porque está aqui**, **o que traz ao tema 13**, **explicação em 30 segundos** e
**perguntas prováveis**.

### 4.1 Teorema de Bayes / Naive Bayes — temas 8 e 9 (ferramenta, não tema principal)

**Porque está aqui.** Para avaliar uma representação é preciso um classificador. O Naive Bayes multinomial é o
clássico para contagens de palavras: rápido, funciona com poucos dados e é explicável. Já estava na versão original.

**Em 30 segundos.**
> "Para cada categoria perguntamos: se a mensagem fosse desta categoria, quão provável era aparecerem estas palavras?
> Escolhemos a categoria que melhor explica as palavras."

$$P(C \mid \text{msg}) \propto P(C)\prod_i P(\text{palavra}_i \mid C), \qquad P(w \mid C) = \frac{N_{w,C} + \alpha}{N_C + \alpha\,|V|}$$

**Exemplo resolvido para o slide.** Treino: financeiro = {"propina paga", "propina atrasada"};
notas = {"nota exame", "pauta exame"}. Vocabulário com 6 palavras; cada categoria tem 4 palavras; priori 0,5; α = 1.

Mensagem nova: **"propina atrasada exame"**

| | financeiro | notas |
|---|---|---|
| P(propina \| C) | 3/10 = 0,3 | 1/10 = 0,1 |
| P(atrasada \| C) | 2/10 = 0,2 | 1/10 = 0,1 |
| P(exame \| C) | 1/10 = 0,1 | 3/10 = 0,3 |
| × P(C) = 0,5 | 0,003 | 0,0015 |
| **normalizado** | **66,7 %** | 33,3 % |

Sem Laplace, P(exame | financeiro) seria 0 e anulava o produto inteiro. É para isso que serve o α.

**Perguntas prováveis.**
- *"Porque se chama ingénuo?"* → Assume que as palavras são independentes dada a categoria. Não é verdade
  ("propina" e "pagamento" aparecem juntas), mas na prática classifica bem.
- *"Onde falha?"* → Negação: "o pagamento validou" e "o pagamento nao validou" dão os dois financeiro com ≈ 97 %.
  E sobreconfiança (ver 4.3).
- *"Porque usam logaritmos?"* → Multiplicar muitas probabilidades pequenas dá um número tão pequeno que o computador
  o arredonda para zero (*underflow*). Somar logaritmos é equivalente e estável: `feature_log_prob_`.

### 4.2 Regressão Logística — temas 8 e 9

**Porque está aqui.** Com um só classificador não se sabe se uma diferença vem da **representação** ou do
**classificador**. Com dois, separa-se o efeito de cada um, e a pergunta do tema 13 fica respondida com rigor.

**Em 30 segundos.**
> "O Naive Bayes modela como cada categoria *gera* palavras: é generativo. A Regressão Logística aprende directamente
> um peso por palavra e categoria e escolhe a fronteira entre categorias: é discriminativa. Não precisa de assumir
> independência entre palavras."

$$P(C_k \mid x) = \frac{e^{w_k \cdot x + b_k}}{\sum_j e^{w_j \cdot x + b_j}}$$

**Perguntas prováveis.**
- *"O que é o C?"* → O inverso da regularização. C pequeno = pesos pequenos (modelo simples); C grande = ajusta-se
  mais ao treino (risco de sobreajuste). Escolhemos C = 5 por validação cruzada.
- *"Se a RL é melhor, porque mantêm o NB?"* → Como referência (é a versão original) e porque a comparação é o que
  dá valor ao resultado.

### 4.3 Avaliação rigorosa e calibração — tema 9

**Porque está aqui.** Uma comparação só vale se a avaliação for justa. Encontrámos e corrigimos três problemas
na versão anterior:

| Problema | Consequência | Correcção |
|---|---|---|
| Textos repetidos no treino e no teste (fuga de dados) | Exatidão inflacionada | Divisão **agrupada** por texto |
| Teste com 46 % de "financeiro" (no corpus: 29 %) | Hold-out enviesado (73,9 % → na verdade 63,4 %) | Divisão **estratificada** |
| Um só sorteio | Resultado instável | **CV 5 folds × 3 sementes**, média ± desvio-padrão |

**Calibração.** Uma confiança de "90 %" deve acertar 90 % das vezes. O ECE mede a distância a esse ideal (0 = perfeito).
É importante porque **o agente usa a confiança para decidir a triagem humana**.
- `alguem me pode ajudar pfv` (frase sem informação): BoW + NB dá **94 %** infraestrutura (ECE 0,213); caracteres + RL
  dá **47 %** → triagem humana (ECE 0,121).
- Porquê? O NB conta pistas correlacionadas como se fossem independentes e soma evidência repetida.

### 4.4 Agente e PEAS — tema 1

**Porque está aqui.** Um classificador sozinho só dá um rótulo. O programa pede para modelar o agente; aqui o
classificador é a **percepção** e o agente **age**.

| PEAS | |
|---|---|
| **P** Desempenho | Utilidade média: certo **+1**, errado **−2**, triagem humana **−0,5**, urgente perdido **−3** |
| **E** Ambiente | Mensagens curtas em português, formais e informais, com gralhas, abreviaturas e dados pessoais; parcialmente observável, estocástico |
| **A** Actuadores | Encaminhar ao serviço; prioridade ALTA/NORMAL; enviar a triagem humana; mascarar dados pessoais |
| **S** Sensores | Texto da mensagem |

**Tipo de agente: baseado em utilidade** (com aprendizagem). Não escolhe só "a classe mais provável": escolhe a acção
que maximiza a utilidade esperada. Se a confiança é baixa, encaminhar arrisca −2, e é melhor pagar −0,5 e chamar um humano.

**Resultado (teste):** utilidade +0,05 · 12,7 % para triagem humana · 87,2 % de acerto nas mensagens encaminhadas ·
19 de 46 urgentes perdidos.

**Urgência.** A coluna `urgente` do corpus é o segundo alvo, com a mesma representação (TF-IDF + NB, F1 macro ≈ 75 % na CV).
Cuidado: **54 textos** têm rótulos de urgência contraditórios.

**Pergunta para a turma (actividade obrigatória nos slides):**
> "Se encaminhar mal custasse −5 em vez de −2, o limiar de confiança óptimo subia ou descia?"
> (Resposta: subia — erros mais caros → o agente prefere chamar um humano mais vezes. Mostrem na curva de utilidade.)

### 4.5 IA responsável — tema 15

**Porque está aqui.** O objectivo geral da cadeira pede "análise crítica dos riscos e do impacto". No nosso caso
há dois riscos ligados ao **texto**: dados pessoais nas mensagens e tratamento desigual de quem escreve de forma
formal ou informal.

**Privacidade.** Antes de classificar, `anonimizar()` troca emails, BI, telemóveis (+258 8x…) e números longos por
`[EMAIL]`, `[TELEFONE]`, etc. Isto é **preparação de texto** (tema 13) com uma razão ética: minimização de dados.
Limitação: não detecta nomes próprios.

**Viés por registo (formal vs informal).**

| Onde | Formal | Informal | Leitura |
|---|---|---|---|
| Categoria — modelo original | 42,6 % | 74,7 % | o modelo original falha muito nas formais |
| Categoria — melhor modelo | 78,7 % | 90,8 % | a diferença diminui, mas continua |
| Rótulo "urgente" (dentro de cada categoria) | ≈ 44 % | ≈ 28 % | historicamente, marcaram-se mais formais |
| **Modelo de urgência — revocação** | **32 %** | **78 %** | o modelo **perde** urgentes formais |

**Porque acontece:** os pedidos formais urgentes usam linguagem cortês ("Solicito…"), enquanto os informais dizem
"ja", "hoje", "!!". O modelo aprende essas palavras. **O viés está na representação do texto, não na taxa de urgentes.**

**Duas medidas correctivas testadas:**

| Medida | Tipo | Diferença de revocação |
|---|---|---|
| Sem correcção | — | 46 p.p. |
| Reponderação (Kamiran & Calders) | pré-processamento | 50 p.p. ✗ (iguala as taxas de base, mas o problema está nas palavras) |
| **Limiar por subgrupo** (Hardt et al.) | pós-processamento | **8 p.p.** ✓ (revocação formal 32 % → 74 %) |

Os limiares são escolhidos com previsões fora-da-amostra **do treino**: o teste nunca é usado para afinar.
Ressalva: assentam em poucos textos formais urgentes distintos, por isso são frágeis.

**Enquadramento legal (verificar antes de citar):** Constituição da República de Moçambique, art. 71.º (utilização da
informática e protecção de dados pessoais); Lei n.º 3/2017 (Transacções Electrónicas).

---

## 5. Guião da aula aberta (60 min)

| Min | Bloco | Quem | Material |
|---|---|---|---|
| 0–5 | Título, grupo, objectivos; o problema dos serviços académicos | A | slides |
| 5–15 | O que é PLN; preparação do texto; demonstração passo a passo | A | app → *Como o texto vira números* |
| 15–25 | BoW, TF-IDF, n-gramas de caracteres; exemplo `matricla` | A | app → gralhas |
| 25–33 | Naive Bayes (exemplo resolvido) vs Regressão Logística | B | slides (tabela da secção 4.1) |
| 33–43 | Resultados: tabela, mapa de calor, leitura honesta; calibração | B | app → *Comparação* |
| 43–50 | Agente e PEAS; urgência; fila | A | app → *Classificar* + *Urgência & fila* |
| 50–55 | IA responsável: anonimização e viés formal/informal | B | app → *IA responsável* |
| 55–60 | Limitações, melhorias, **pergunta à turma** | A + B | slide final |

**Demonstração ao vivo (5 mensagens, por esta ordem):**
1. `o projector da sala 14 nao funciona` → infraestrutura, caso fácil.
2. `quero trancar a matricla` → mudar para BoW + NB (32 %, humano) e voltar para caracteres + RL (87 %).
3. `alguem me pode ajudar pfv` → triagem humana; explicar a calibração.
4. `Falta de energia no bloco onde decorre o exame de hoje. Solicito intervencao imediata` → prioridade ALTA.
5. `o projector avariou, liguem 84 123 4567` → `[TELEFONE]` mascarado.

---

## 6. Defesa fechada: perguntas sobre linhas concretas do código

O programa diz que *"pelo menos uma questão incide sobre uma linha concreta do código"*. Estas são as mais prováveis.

| Ficheiro:linha | Código | O que responder |
|---|---|---|
| `texto.py:17` | `unicodedata.normalize("NFKD", …)` | Separa a letra do acento ("á" → "a" + acento); a linha seguinte remove os acentos |
| `texto.py:19` | `re.sub(r"[^a-z0-9\s]", " ", t)` | Troca por espaço tudo o que não é letra, dígito ou espaço (pontuação, emojis) |
| `representacoes.py:30` | `analyzer="char_wb", ngram_range=(3, 5)` | Pedaços de 3 a 5 letras **dentro** de cada palavra; o `wb` (*word boundary*) junta um espaço em cada ponta |
| `classificadores.py:17` | `MultinomialNB(alpha=alpha)` | NB para contagens; `alpha` é a suavização de Laplace |
| `classificadores.py:20` | `LogisticRegression(C=C, max_iter=3000)` | C = inverso da regularização; `max_iter` garante que a optimização converge |
| `classificadores.py:27` | `Pipeline([("vec", …), ("clf", …)])` | Junta representação e classificador: em cada fold o vocabulário e o IDF só se aprendem no treino (evita fuga) |
| `classificadores.py:76` | `x[0, j] * (w[i1, j] - w[i2, j])` | Peso do termo na mensagem × diferença dos seus pesos entre a 1.ª e a 2.ª classe = contribuição para a log-odds |
| `avaliacao.py:62-63` | `k = round(1/proporcao_teste)`; `StratifiedGroupKFold(k)` | 20 % → k = 5 → o 1.º fold é o teste: agrupado (sem fuga) e estratificado |
| `avaliacao.py:115` | `grupos = X.map(normalizar)` | Mensagens iguais depois de normalizadas formam um grupo e ficam sempre do mesmo lado |
| `avaliacao.py:97` | `ece += sel.mean() * abs(a_b - c_b)` | Fracção das mensagens neste intervalo × diferença entre exatidão e confiança |
| `avaliacao.py:106` | `@threadpool_limits.wrap(limits=1)` | Limita o BLAS a 1 thread: duas cópias do OpenBLAS competiam e o treino demorava 10× mais |
| `agente.py:38-39` | `prioridade = "ALTA" if …`; `if confianca < limiar_conf` | Primeiro calcula a prioridade; com confiança baixa vai a triagem humana mas mantém a prioridade |
| `agente.py:61` | `u = certo * … + errado * … + humano * … + perdido * …` | Utilidade por mensagem (vectorizada com numpy); a média é a medida de desempenho P |
| `responsavel.py:14` | regex `TELEFONE` | `(?<!\d)` = não vem depois de um dígito; `+258` opcional; `8[2-7]` = operadoras; 7 dígitos com espaços ou hífens |
| `responsavel.py:101` | `p_g * p_y / p_gy` | Reponderação: combinações raras (ex.: informal + urgente) ganham peso > 1 |
| `urgencia.py:42` | `cross_val_predict(…)` | Previsões fora-da-amostra do treino, para escolher limiares sem tocar no teste |

**Outras perguntas prováveis:**
- *"Porque não 100 % de exatidão?"* → 29 textos aparecem com categorias diferentes; nenhum modelo acerta os dois rótulos.
- *"Qual é a diferença entre hold-out e validação cruzada?"* → O hold-out é um sorteio; a CV repete 15 vezes e dá média ± dp.
- *"Porque é que o TF-IDF tem norma 1?"* → Para que mensagens longas não tenham mais peso só por serem longas.
- *"O que é uma fuga de dados?"* → O modelo vê no treino o que depois é avaliado no teste. Na app, o expander mostra o efeito.
- *"Usaram IA para fazer o trabalho?"* → Sim, declarado em anexo (ver README §7). Sabemos explicar cada linha e verificámos os resultados.

---

## 7. Limitações e melhorias futuras

| Limitação | Melhoria possível |
|---|---|
| A ordem das palavras e a negação perdem-se | Marcar negações (`nao_validou`); bigramas |
| Corpus pequeno e ambíguo | Recolher mensagens reais anonimizadas; classificação multi-etiqueta |
| Diferenças dentro do desvio-padrão | Teste estatístico (McNemar) entre combinações |
| Sem stemming nem dicionário de abreviaturas | RSLP para português; `pfv` → `por favor` |
| A anonimização não detecta nomes | Modelo de reconhecimento de entidades |
| Rótulos históricos com viés | Rever rótulos com critérios explícitos de urgência |

## 8. Lista final antes da apresentação

- [ ] `pytest -v` → 67 passed
- [ ] App aberta e já treinada (o primeiro arranque demora uns segundos)
- [ ] Tema escolhido para o projector (o claro costuma ler-se melhor)
- [ ] Números dos slides conferidos na app (semente 42)
- [ ] Tabela de resultados descarregada (CSV) e no artigo
- [ ] Declaração de uso de IA em anexo ao artigo
- [ ] Cada membro já explicou em voz alta todas as linhas da secção 6
