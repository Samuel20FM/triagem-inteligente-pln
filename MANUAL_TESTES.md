# Manual de Execução e Testes — Triagem Inteligente (Grupo 13)

## Parte A — Como correr a aplicação

### A.1 Requisitos
- **Python 3.9 ou superior** (`python --version`). Testado com Python 3.12 e Streamlit 1.67.
- Internet **apenas** para instalar as bibliotecas; a app corre offline.

### A.2 Estrutura de pastas
```
triagem/
├── app.py                 interface
├── texto.py  representacoes.py  classificadores.py  avaliacao.py
├── urgencia.py  agente.py  responsavel.py
├── corpus_referencia.csv
├── requirements.txt
├── tests/                 testes automáticos
└── .streamlit/config.toml temas claro e escuro (pasta escondida)
```

### A.3 Passo a passo

**Windows**
```powershell
cd caminho\para\triagem
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Linux / macOS**
```bash
cd caminho/para/triagem
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

O browser abre em `http://localhost:8501`. Para parar: `Ctrl + C`.
**Tema:** menu ⋮ (canto superior direito) → *Settings* → *Theme* → Light / Dark / System.

### A.4 Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `'streamlit' não é reconhecido…` | Ambiente não activado | Activar o `.venv` ou usar `python -m streamlit run app.py` |
| `ModuleNotFoundError` | Dependências em falta | `pip install -r requirements.txt` |
| "Não foi possível ler o corpus" | CSV fora da pasta do `app.py` | Copiar o CSV para a pasta ou carregá-lo em **Fonte de dados** |
| Porta ocupada | Outra app Streamlit aberta | `streamlit run app.py --server.port 8502` |
| Erro com `width="stretch"` | Streamlit antigo | `pip install -U streamlit` (≥ 1.50) |
| Não aparece a opção de tema escuro | Falta `.streamlit/config.toml` | Confirmar que a pasta escondida `.streamlit` foi copiada |
| Primeiro arranque demora ~5–10 s | Treina 6 modelos + CV + urgência | Normal; depois fica em cache |

---

## Parte B — Testes automáticos

```bash
pytest -v
```
Resultado esperado: **67 passed** (cerca de 25 s).

| Ficheiro | O que verifica |
|---|---|
| `test_texto.py` | normalização; negações não são stop-words |
| `test_representacoes.py` | BoW conta; TF-IDF tem norma 1; caracteres resistem a gralhas ("matricla") |
| `test_classificadores.py` | as 6 combinações; previsões típicas com NB e RL; explicação; 1 categoria não rebenta |
| `test_avaliacao.py` | divisão sem fuga, estratificada e com ≈ 20 %; resultados coerentes; ECE; CSV com `;`, com vazios e sem coluna |
| `test_urgencia.py` | urgência treina e prevê; curva de limiar; previsão fora-da-amostra |
| `test_agente.py` | triagem humana; prioridade; ordem da fila; utilidade |
| `test_responsavel.py` | anonimização (telefone, email, BI, número); subgrupos; reponderação; limiares por subgrupo |
| `test_app.py` | 7 separadores; classificar; mensagem vazia; telefone mascarado; parâmetros |

---

## Parte C — Testes manuais (na interface)

Parâmetros por omissão: stop-words ligadas, unigramas, α = 1, C = 5, teste 20 %, semente 42, confiança mínima 55 %,
P(urgente) 50 %. Marquem OK/Falhou. "Original" = BoW + Naive Bayes; "Melhor" = TF-IDF (caracteres) + Regressão Logística.

### C.1 Arranque e aparência

| ID | Passos | Resultado esperado | Estado |
|---|---|---|---|
| T01 | Abrir a app | Cabeçalho: **670 mensagens · 4 categorias · treino 536 / teste 134** · melhor modelo **TF-IDF (caracteres) + Regressão Logística** | |
| T02 | Clicar nos 7 separadores | Todos abrem sem erro | |
| T03 | Menu ⋮ → Settings → Theme → Dark, depois Light | Toda a app muda de tema; textos, cartões e gráficos legíveis nos dois | |

### C.2 Classificar

| ID | Mensagem / acção | Resultado esperado | Estado |
|---|---|---|---|
| T04 | `o projector da sala 14 nao funciona` | **Infraestrutura** (≈ 88 %) → Manutenção | |
| T05 | `quero trancar a matricla` (com gralha) | Melhor: **Matrícula ≈ 87 %**. Escolher BoW + NB: ≈ 32 % → **triagem humana** | |
| T06 | `alguem me pode ajudar pfv` | Melhor: ≈ 47 % → **triagem humana** (o original dava ≈ 94 %, sobreconfiante) | |
| T07 | `Falta de energia no bloco onde decorre o exame de hoje. Solicito intervencao imediata` | Infraestrutura; P(urgente) ≈ 96 %; prioridade **ALTA** (caixa vermelha) | |
| T08 | `o projector avariou, liguem 84 123 4567` | Aviso de privacidade; texto usado contém `[TELEFONE]` | |
| T09 | `xyzzy qwerty` | Confiança ≈ 36 % → triagem humana; sem erro | |
| T10 | Apagar todo o texto | Aviso "Escreva uma mensagem…" | |
| T11 | Botão **Mensagem aleatória** / lista de exemplos | A caixa é preenchida e o resultado actualiza | |
| T12 | Mudar a representação/classificador nos botões do passo 2 | Decisão e gráfico "Porquê?" mudam | |
| T13 | Representação de caracteres, gráfico "Porquê?" | Pedaços como `_pag`, `prop`, `pina`; `_` marca o início ou o fim de uma palavra | |
| T14 | Grelha "6 combinações" com T05 | Pelo menos uma combinação discorda ou tem confiança muito diferente | |

### C.3 Como o texto vira números

| ID | Passos | Resultado esperado | Estado |
|---|---|---|---|
| T15 | Frase `Matrícula NÃO confirmada` | Normalizada = `matricula nao confirmada` | |
| T16 | Passo ⑤ | Lista de n-gramas de caracteres; dimensões: palavras 278, caracteres 3140 | |
| T17 | Gralhas: `matricula` vs `matricla` | **41 %** de n-gramas em comum; "matricla" **não** está no vocabulário de palavras | |

### C.4 Comparação

| ID | Passos | Resultado esperado | Estado |
|---|---|---|---|
| T18 | Cartões | F1 CV original **69,3 %**, melhor **78,4 %** (+9,1 p.p.); ECE 0,213 → 0,121 | |
| T19 | Tabela | 6 linhas; 1.ª coluna "Ficheiro de referência" | |
| T20 | **Descarregar tabela (CSV)** | `resultados_corpus_referencia.csv` com 6 linhas | |
| T21 | Leitura | Efeito da representação ≈ 3,1 p.p.; do classificador ≈ 4,9 p.p.; aviso de que diferenças < dp não são conclusivas | |
| T22 | Calibração | Duas linhas e a diagonal tracejada; a linha do original fica mais afastada da diagonal | |
| T23 | Detalhe → matriz de confusão | A soma dá **134** | |
| T24 | Expander "Fuga de dados" | Exatidão com fuga ≥ sem fuga na maioria das combinações | |

### C.5 Urgência & fila

| ID | Passos | Resultado esperado | Estado |
|---|---|---|---|
| T25 | Ver métricas | Revocação ≈ 58,7 %; alarmes 46 / 134; nota sobre **54 textos** contraditórios | |
| T26 | Baixar "P(urgente) para prioridade ALTA" para 30 % | Revocação sobe, precisão desce; a linha tracejada move-se | |
| T27 | Desempenho do agente | Utilidade ≈ +0,05; triagem humana ≈ 12,7 %; acerto nas encaminhadas ≈ 87,2 % | |
| T28 | Fila simulada | Mensagens com prioridade ALTA no topo | |

### C.6 IA responsável

| ID | Passos | Resultado esperado | Estado |
|---|---|---|---|
| T29 | Caixa "Experimente" | `[NUMERO_ID]`, `[TELEFONE]`, `[EMAIL]` e tabela com os valores | |
| T30 | Viés na categoria | Original: formal ≈ 42,6 % vs informal ≈ 74,7 %; melhor: ≈ 78,7 % vs 90,8 % | |
| T31 | Rótulo histórico | Formal > informal em todas as categorias | |
| T32 | Medidas correctivas | Diferença de revocação: sem correcção ≈ 46 p.p.; reponderação ≈ 50; limiar por subgrupo ≈ **8** | |

### C.7 Parâmetros e robustez

| ID | Passos | Resultado esperado | Estado |
|---|---|---|---|
| T33 | Bigramas / desligar stop-words | Sem erro; F1 CV original ≈ 68,2 % / 66,7 %; melhor mantém 78,4 % (caracteres não usam stop-words) | |
| T34 | Semente 7 | Teste continua com 134 mensagens (a proporção é respeitada) | |
| T35 | Proporção de teste 33 % | Cabeçalho actualiza; sem erro | |
| T36 | Mensagem de 500+ caracteres; `123 !!!` | Sem erro | |
| T37 | Mudar o nome do CSV e reiniciar | Mensagem de erro clara; o uploader repõe o funcionamento | |
| T38 | Carregar CSV sem coluna `categoria` | Erro "O CSV não tem a(s) coluna(s): categoria" | |
| T39 | Carregar CSV só com uma categoria ou com poucos textos | Erro "O corpus não serve para treinar…", sem traceback | |
| T40 | CSV separado por `;` | Lido correctamente | |

---

## Parte D — Casos para explorar na defesa

| Caso | O que mostra | Como explicar |
|---|---|---|
| `quero trancar a matricla` | Gralhas | Palavras: "matricla" é desconhecida. Caracteres: partilha `mat`, `atr`, `tri`… com "matricula" |
| `alguem me pode ajudar pfv` | Sobreconfiança | O NB multiplica evidências sob uma independência falsa; a RL não assume independência → confiança mais honesta → triagem humana |
| `o pagamento validou` vs `nao validou` | Negação | Ambos dão financeiro ≈ 97 %: a representação não capta o sentido; só os bigramas ajudam um pouco |
| Formal vs informal na urgência | Viés | O rótulo marca mais formais, mas o modelo apanha menos formais: as palavras dos pedidos formais urgentes são corteses |
| Hold-out vs CV | Instabilidade | 134 mensagens de teste são um único sorteio; por isso reportamos média ± dp |

---

## Parte E — Registo de execução (preencher)

| Campo | Valor |
|---|---|
| Data | |
| Máquina / SO | |
| Versão Python / Streamlit | |
| `pytest` (nº passed/failed) | |
| Testes manuais OK / Falhou | ___ / 40 |
| Observações | |
