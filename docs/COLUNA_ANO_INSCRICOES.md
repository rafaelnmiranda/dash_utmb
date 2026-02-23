# Coluna `ano` na Tabela `inscricoes`

## ✅ Status: Configurada e Pronta

A coluna `ano` (ano do evento) já existe na tabela `inscricoes` e está totalmente configurada.

## 📋 Detalhes da Coluna

### Definição
```sql
ano INTEGER NOT NULL
```

- **Tipo**: `INTEGER`
- **Obrigatória**: `NOT NULL` (não pode ser vazia)
- **Constraint de Validação**: `CHECK (ano >= 2023 AND ano <= 2030)`
  - Aceita apenas anos entre 2023 e 2030

## 🗂️ Índices Criados

A coluna `ano` tem múltiplos índices para otimizar queries:

1. **`idx_inscricoes_ano`**
   - Índice simples na coluna `ano`
   - Otimiza filtros por ano: `WHERE ano = 2025`

2. **`idx_inscricoes_ano_competition`**
   - Índice composto: `ano, competition`
   - Otimiza queries: `WHERE ano = 2025 AND competition = 'PTR 55'`

3. **`idx_inscricoes_ano_date`**
   - Índice composto: `ano, registration_date`
   - Otimiza análises temporais: `WHERE ano = 2025 ORDER BY registration_date`

4. **`unique_original_id_ano`**
   - Constraint UNIQUE: `original_id, ano`
   - Evita duplicatas: mesmo `original_id` não pode aparecer duas vezes no mesmo ano

5. **`idx_inscricoes_original_id_ano`**
   - Índice composto: `original_id, ano`
   - Otimiza busca de duplicatas

## 📊 Uso da Coluna

### Exemplos de Queries

```sql
-- Contar inscrições por ano
SELECT ano, COUNT(*) as total
FROM inscricoes
GROUP BY ano
ORDER BY ano;

-- Inscrições de 2025
SELECT *
FROM inscricoes
WHERE ano = 2025;

-- Comparar anos
SELECT 
    ano,
    competition,
    COUNT(*) as inscritos,
    SUM(registration_amount) as receita
FROM inscricoes
WHERE ano IN (2023, 2024, 2025)
GROUP BY ano, competition
ORDER BY ano, competition;
```

### Popular a Coluna

Ao processar um arquivo Excel, o ano será detectado automaticamente do nome do arquivo e atribuído a todos os registros:

```python
from scripts.detect_year_from_filename import detect_year

# Detectar ano do arquivo
detection = detect_year(filename='UTMB - 2025 - USD.xlsx')
ano = detection['year']  # 2025

# Aplicar a todos os registros
df['ano'] = ano
```

## 🎯 Importância da Coluna

A coluna `ano` é essencial para:

1. **Filtragem por Ano**: Separar dados de diferentes edições do evento
2. **Análises Comparativas**: Comparar performance entre anos
3. **Prevenção de Duplicatas**: Evitar que o mesmo registro apareça em múltiplos anos
4. **Queries Eficientes**: Índices otimizam queries que filtram por ano
5. **Dashboard**: Permite seleção de anos específicos para visualização

## ✅ Confirmação

A coluna está:
- ✅ Criada na tabela
- ✅ Com constraint de validação
- ✅ Com índices otimizados
- ✅ Pronta para uso no processamento de uploads


