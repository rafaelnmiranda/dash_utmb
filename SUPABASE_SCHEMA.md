# Estrutura de Tabelas - Supabase Dashboard UTMB

## Análise das Planilhas

Baseado na análise das planilhas Excel (2023, 2024, 2025 USD e BRL), identifiquei:
- **Total de colunas únicas**: 34 colunas diferentes entre todas as planilhas
- **Campos obrigatórios**: id, booking_reference, nomes, email, registration_date, valores financeiros
- **Campos opcionais**: Vários campos variam entre anos (ex: `information_address` vs `Adress`)
- **Total de registros 2025**: ~1.811 registros na planilha BRL

---

## 1. Tabela Principal: `inscricoes`

Esta é a tabela central onde todos os dados de inscrições processados serão armazenados.

### Campos Obrigatórios (NOT NULL)

```sql
-- Identificação
id                  UUID PRIMARY KEY DEFAULT gen_random_uuid()
original_id         TEXT NOT NULL              -- ID original da planilha (campo 'id' do Excel)
booking_reference   TEXT NOT NULL              -- Referência de reserva

-- Dados Pessoais
last_name           TEXT NOT NULL              -- Sobrenome
first_name          TEXT NOT NULL              -- Nome
email               TEXT NOT NULL              -- Email (usado para identificar duplicatas)
birthdate           DATE NOT NULL              -- Data de nascimento

-- Inscrição
registration_date   TIMESTAMP NOT NULL         -- Data de inscrição
status              TEXT NOT NULL              -- Status: 'COMPLETED', 'IN PROGRESS'
competition         TEXT NOT NULL              -- Competição padronizada (PTR 20, UTSB 100, etc.)
competition_original TEXT NOT NULL             -- Competição original da planilha

-- Valores Financeiros (sempre em BRL após processamento)
registration_amount NUMERIC(10,2) NOT NULL     -- Valor da inscrição em BRL
total_registration_amount NUMERIC(10,2) NOT NULL -- Valor total em BRL
discounts_amount    NUMERIC(10,2) NOT NULL DEFAULT 0 -- Valor de descontos em BRL

-- Localização
country             TEXT NOT NULL              -- País
city                TEXT NOT NULL              -- Cidade normalizada via IBGE
city_original       TEXT NOT NULL              -- Cidade original da planilha

-- Nacionalidade
nationality         TEXT NOT NULL              -- Nacionalidade padronizada (BR, US, etc.)
nationality_original TEXT NOT NULL             -- Nacionalidade original

-- Metadados
ano                 INTEGER NOT NULL           -- Ano da inscrição (2023, 2024, 2025, 2026)
moeda_original      TEXT NOT NULL              -- Moeda original: 'USD' ou 'BRL'
created_at          TIMESTAMP NOT NULL DEFAULT NOW()
updated_at          TIMESTAMP NOT NULL DEFAULT NOW()
```

### Campos Opcionais (NULL permitido)

```sql
-- Código de Desconto
discount_code       TEXT                       -- Código de desconto usado

-- Contato
phone_number        TEXT                       -- Telefone
address             TEXT                       -- Endereço (pode ser 'Adress' ou 'information_address')

-- Informações Adicionais
team_club           TEXT                       -- Time/Clube
empresa             TEXT                       -- Empresa
zip_code            TEXT                       -- CEP
tshirt_size_woman   TEXT                       -- Tamanho camiseta feminina
tshirt_size_man     TEXT                       -- Tamanho camiseta masculina
campaign_link        TEXT                       -- Link da campanha
information_country_code TEXT                  -- Código do país (information_countryCode)
schedule_bib_retrieval TEXT                    -- Horário de retirada do bib

-- Campos específicos de alguns anos
utmb_indexes        TEXT                       -- Índices UTMB (apenas 2023)
document_status     TEXT                       -- Status do documento (apenas 2023)
idade               INTEGER                     -- Idade calculada (apenas 2023 tem como coluna)

-- Campos Calculados (preenchidos durante processamento)
uf                  TEXT                       -- Estado (UF) - calculado via join com IBGE
regiao              TEXT                       -- Região - calculado via join com IBGE
idade_calculada     INTEGER                    -- Idade calculada a partir de birthdate
```

### Índices para Performance

```sql
-- Índices principais
CREATE INDEX idx_inscricoes_email ON inscricoes(email);
CREATE INDEX idx_inscricoes_ano ON inscricoes(ano);
CREATE INDEX idx_inscricoes_competition ON inscricoes(competition);
CREATE INDEX idx_inscricoes_registration_date ON inscricoes(registration_date);
CREATE INDEX idx_inscricoes_nationality ON inscricoes(nationality);
CREATE INDEX idx_inscricoes_city ON inscricoes(city);
CREATE INDEX idx_inscricoes_original_id_ano ON inscricoes(original_id, ano); -- Para busca de duplicatas
CREATE INDEX idx_inscricoes_country ON inscricoes(country);
CREATE INDEX idx_inscricoes_status ON inscricoes(status);

-- Índice composto para queries frequentes
CREATE INDEX idx_inscricoes_ano_competition ON inscricoes(ano, competition);
CREATE INDEX idx_inscricoes_ano_date ON inscricoes(ano, registration_date);
```

### Constraints

```sql
-- Evitar duplicatas baseado no ID original + ano
ALTER TABLE inscricoes 
ADD CONSTRAINT unique_original_id_ano 
UNIQUE (original_id, ano);

-- Valores financeiros não podem ser negativos
ALTER TABLE inscricoes 
ADD CONSTRAINT check_positive_amounts 
CHECK (registration_amount >= 0 AND discounts_amount >= 0);

-- Ano deve ser válido
ALTER TABLE inscricoes 
ADD CONSTRAINT check_valid_ano 
CHECK (ano >= 2023 AND ano <= 2030);

-- Status deve ser válido
ALTER TABLE inscricoes 
ADD CONSTRAINT check_valid_status 
CHECK (status IN ('COMPLETED', 'IN PROGRESS', 'CANCELLED', 'PENDING'));
```

---

## 2. Tabela: `upload_historico`

Controla o histórico de uploads e processamento de arquivos.

```sql
CREATE TABLE upload_historico (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name           TEXT NOT NULL,
    file_type           TEXT NOT NULL,              -- 'USD' ou 'BRL'
    ano                 INTEGER NOT NULL,
    file_size           BIGINT,                     -- Tamanho em bytes
    total_registros     INTEGER NOT NULL DEFAULT 0, -- Total de registros no arquivo
    registros_processados INTEGER NOT NULL DEFAULT 0, -- Registros processados com sucesso
    registros_inseridos INTEGER NOT NULL DEFAULT 0, -- Novos registros inseridos
    registros_atualizados INTEGER NOT NULL DEFAULT 0, -- Registros atualizados
    registros_ignorados INTEGER NOT NULL DEFAULT 0, -- Registros ignorados (ex: KIDS)
    erros               JSONB,                     -- Array de erros encontrados
    warnings            JSONB,                     -- Array de avisos
    status              TEXT NOT NULL DEFAULT 'processing', -- 'processing', 'success', 'error'
    uploaded_by         TEXT,                      -- Email ou ID do usuário (para futuro)
    uploaded_at         TIMESTAMP NOT NULL DEFAULT NOW(),
    processed_at        TIMESTAMP,                 -- Quando foi processado
    processing_time_ms  INTEGER,                   -- Tempo de processamento em ms
    
    CONSTRAINT check_valid_file_type CHECK (file_type IN ('USD', 'BRL')),
    CONSTRAINT check_valid_ano CHECK (ano >= 2023 AND ano <= 2030)
);

CREATE INDEX idx_upload_historico_ano ON upload_historico(ano);
CREATE INDEX idx_upload_historico_status ON upload_historico(status);
CREATE INDEX idx_upload_historico_uploaded_at ON upload_historico(uploaded_at DESC);
```

**Estrutura do JSONB `erros`:**
```json
[
  {
    "linha": 123,
    "campo": "email",
    "erro": "Email inválido",
    "valor": "email@invalido"
  }
]
```

---

## 3. Tabela: `configuracoes`

Armazena configurações do sistema de forma flexível.

```sql
CREATE TABLE configuracoes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chave               TEXT NOT NULL UNIQUE,
    valor               JSONB NOT NULL,
    descricao           TEXT,
    categoria           TEXT,                      -- 'financeiro', 'metas', 'datas', etc.
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_by          TEXT
);

CREATE INDEX idx_configuracoes_chave ON configuracoes(chave);
CREATE INDEX idx_configuracoes_categoria ON configuracoes(categoria);
```

**Exemplos de configurações:**

```sql
-- Taxa de câmbio
INSERT INTO configuracoes (chave, valor, descricao, categoria) VALUES
('taxa_cambio', '{"valor": 5.5, "moeda_base": "USD", "moeda_destino": "BRL"}', 
 'Taxa de câmbio USD para BRL', 'financeiro');

-- Metas 2025
INSERT INTO configuracoes (chave, valor, descricao, categoria) VALUES
('metas_2025', '{
  "FUN 7KM": 900,
  "PTR 20": 900,
  "PTR 35": 620,
  "PTR 55": 770,
  "UTSB 100": 310,
  "TOTAL": 3500
}', 'Metas de inscritos por competição - 2025', 'metas');

-- Metas 2026 (quando disponível)
INSERT INTO configuracoes (chave, valor, descricao, categoria) VALUES
('metas_2026', '{
  "FUN 7KM": 900,
  "PTR 20": 900,
  "PTR 35": 620,
  "PTR 55": 770,
  "UTSB 100": 310,
  "TOTAL": 3500
}', 'Metas de inscritos por competição - 2026', 'metas');

-- Datas de prazo de vendas
INSERT INTO configuracoes (chave, valor, descricao, categoria) VALUES
('prazo_vendas_2025', '{
  "inicio": "2024-10-28",
  "fim": "2025-08-15"
}', 'Prazo de vendas de inscrições - 2025', 'datas');
```

---

## 4. Tabela: `ibge_municipios` (Opcional - Cache)

Para melhorar performance, podemos criar uma tabela com dados do IBGE já processados.

```sql
CREATE TABLE ibge_municipios (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    city                TEXT NOT NULL UNIQUE,      -- Nome oficial da cidade
    city_norm           TEXT NOT NULL,             -- Nome normalizado (sem acentos, lowercase)
    uf                  TEXT NOT NULL,             -- Estado (UF)
    regiao              TEXT NOT NULL,             -- Região (Norte, Nordeste, etc.)
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ibge_city_norm ON ibge_municipios(city_norm);
CREATE INDEX idx_ibge_uf ON ibge_municipios(uf);
CREATE INDEX idx_ibge_regiao ON ibge_municipios(regiao);
```

**Nota:** Esta tabela pode ser populada a partir do arquivo `municipios_IBGE.xlsx` que já existe no projeto.

---

## 5. Relacionamentos e Views

### View para Estatísticas Rápidas

```sql
CREATE VIEW vw_inscricoes_resumo AS
SELECT 
    ano,
    competition,
    COUNT(*) as total_inscritos,
    COUNT(DISTINCT email) as atletas_unicos,
    SUM(registration_amount) as receita_total,
    SUM(discounts_amount) as descontos_total,
    AVG(registration_amount) as ticket_medio,
    COUNT(CASE WHEN gender = 'Female' THEN 1 END) as total_mulheres,
    COUNT(CASE WHEN nationality = 'BR' THEN 1 END) as total_brasileiros,
    COUNT(CASE WHEN nationality != 'BR' THEN 1 END) as total_estrangeiros,
    COUNT(DISTINCT nationality) as paises_diferentes,
    COUNT(DISTINCT city) as cidades_diferentes
FROM inscricoes
GROUP BY ano, competition;
```

### View para Análise Temporal

```sql
CREATE VIEW vw_inscricoes_diarias AS
SELECT 
    ano,
    DATE(registration_date) as data_inscricao,
    COUNT(*) as inscricoes_dia,
    SUM(registration_amount) as receita_dia,
    competition
FROM inscricoes
GROUP BY ano, DATE(registration_date), competition;
```

---

## 6. RLS (Row Level Security) Policies

Para MVP, todas as tabelas terão SELECT público:

```sql
-- Habilitar RLS
ALTER TABLE inscricoes ENABLE ROW LEVEL SECURITY;
ALTER TABLE upload_historico ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuracoes ENABLE ROW LEVEL SECURITY;

-- Policy: SELECT público (para MVP)
CREATE POLICY "Public read access" ON inscricoes
    FOR SELECT
    USING (true);

CREATE POLICY "Public read access" ON upload_historico
    FOR SELECT
    USING (true);

CREATE POLICY "Public read access" ON configuracoes
    FOR SELECT
    USING (true);

-- Policy: INSERT apenas para autenticados (para uploads)
-- (Será implementado quando autenticação for adicionada)
-- CREATE POLICY "Authenticated insert" ON inscricoes
--     FOR INSERT
--     WITH CHECK (auth.role() = 'authenticated');
```

---

## 7. Funções auxiliares (PostgreSQL)

### Função para calcular idade

```sql
CREATE OR REPLACE FUNCTION calcular_idade(data_nascimento DATE)
RETURNS INTEGER AS $$
BEGIN
    RETURN EXTRACT(YEAR FROM AGE(data_nascimento));
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

### Função para normalizar texto (similar ao Python norm_text)

```sql
CREATE OR REPLACE FUNCTION normalize_text(texto TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Remove acentos e normaliza para lowercase
    RETURN lower(unaccent(texto));
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

**Nota:** A função `unaccent` requer a extensão `unaccent`:
```sql
CREATE EXTENSION IF NOT EXISTS unaccent;
```

### Função para atualizar updated_at automaticamente

```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_inscricoes_updated_at
    BEFORE UPDATE ON inscricoes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

---

## 8. Considerações de Design

### Normalização vs Performance

- **Escolha:** Tabela única `inscricoes` com todos os campos (denormalizada)
- **Razão:** 
  - Facilita queries complexas do dashboard
  - Reduz necessidade de JOINs
  - Dados históricos não mudam frequentemente
  - Performance melhor para relatórios

### Tratamento de Duplicatas

- **Estratégia:** Usar `(original_id, ano)` como chave única
- **Comportamento:** 
  - Se registro existe: **UPDATE** (atualiza valores)
  - Se registro não existe: **INSERT** (cria novo)
- **Vantagem:** Permite re-upload de arquivos atualizados sem criar duplicatas

### Campos Calculados vs Armazenados

- **Campos calculados armazenados:** `idade_calculada`, `uf`, `regiao`
- **Razão:** 
  - Melhor performance nas queries
  - Consistência dos dados
  - Reduz processamento em tempo real

### Campos Originais vs Normalizados

- **Sempre manter ambos:** `city` + `city_original`, `nationality` + `nationality_original`, etc.
- **Razão:**
  - Permite rastreabilidade
  - Facilita debug
  - Permite melhorias futuras na normalização

---

## 9. Estratégia de Migração de Dados

### Fase 1: Criação das Tabelas
1. Criar todas as tabelas com migrations
2. Criar índices
3. Criar constraints
4. Criar funções auxiliares
5. Criar views

### Fase 2: População Inicial
1. Processar arquivo `municipios_IBGE.xlsx` → `ibge_municipios`
2. Processar `UTMB - 2023 - USD.xlsx` → `inscricoes` (ano=2023)
3. Processar `UTMB - 2024 - USD.xlsx` → `inscricoes` (ano=2024)
4. Processar arquivos 2025 (USD e BRL) → `inscricoes` (ano=2025)
5. Registrar cada upload em `upload_historico`

### Fase 3: Validação
1. Verificar contagens de registros
2. Validar valores financeiros
3. Verificar normalizações
4. Comparar com dados originais

---

## 10. Estimativas de Volume

- **Inscrições 2023**: ~1.500 registros
- **Inscrições 2024**: ~1.800 registros  
- **Inscrições 2025**: ~1.800 registros (atual)
- **Total estimado**: ~5.100 registros iniciais
- **Crescimento 2026**: ~2.000-3.000 registros esperados
- **Tamanho estimado por registro**: ~2-3 KB
- **Tamanho total estimado**: ~10-15 MB inicialmente

---

## 11. Próximos Passos

1. ✅ Criar migrations SQL com esta estrutura
2. ✅ Implementar Edge Function para processar Excel
3. ✅ Criar script Python para importação retroativa
4. ✅ Implementar interface de upload
5. ✅ Testar com dados reais

