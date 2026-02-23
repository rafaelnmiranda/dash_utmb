# Status do Projeto - Dashboard UTMB

**Última atualização:** 2025-01-XX

## ✅ O que foi feito hoje

### 1. Banco de Dados Supabase
- ✅ **Projeto conectado:** `hsmrpjzenlrcgncgexsr`
- ✅ **Tabelas criadas:**
  - `inscricoes` (40 colunas) - Tabela principal
  - `upload_historico` (17 colunas) - Histórico de uploads
  - `configuracoes` (7 colunas) - Configurações do sistema
  - `ibge_municipios` (6 colunas) - Cache IBGE para normalização

### 2. Dados Carregados
- ✅ **IBGE:** 5.289 municípios inseridos
- ✅ **Inscrições 2023:** 3.240 registros inseridos (completos)
  - 4 competições: PTR 20 (1.198), PTR 55 (863), PTR 35 (790), UTSB 100 (389)
  - Receita total: R$ 1.946.073,97
- ✅ **Inscrições 2024:** 2.404 registros inseridos (completos)
  - 4 competições: PTR 20 (892), PTR 35 (576), PTR 55 (575), UTSB 100 (361)
  - Receita total: R$ 1.386.896,51
- ✅ **Inscrições 2025 (USD):** 2.173 registros inseridos (completos)
- ✅ **Inscrições 2025 (BRL):** 1.616 registros inseridos (completos)
  - **Total 2025:** 3.789 registros
  - 5 competições: PTR 20, PTR 35, PTR 55, UTSB 100, FUN 7KM
  - Receita total 2025: R$ 2.271.680,08

### 3. Scripts Criados
- ✅ `scripts/upload_ibge_to_supabase.py` - Upload dados IBGE
- ✅ `scripts/upload_inscricoes_2023.py` - Upload inscrições 2023 (com melhorias)
- ✅ `scripts/upload_inscricoes_2024.py` - Upload inscrições 2024
- ✅ `scripts/upload_inscricoes_2025_usd.py` - Upload inscrições 2025 (USD)
- ✅ `scripts/upload_inscricoes_2025_brl.py` - Upload inscrições 2025 (BRL)
- ✅ `scripts/detect_year_from_filename.py` - Detecção de ano do arquivo
- ✅ `scripts/detect_year_from_registration_date.py` - Detecção de ano por data (fallback)

### 4. Documentação
- ✅ `SUPABASE_SCHEMA.md` - Schema completo do banco
- ✅ `docs/DETECCAO_ANO_FINAL.md` - Como detectar ano do evento
- ✅ `docs/COLUNA_ANO_INSCRICOES.md` - Documentação da coluna `ano`
- ✅ `docs/IDENTIFICACAO_ANO.md` - Estratégias de identificação de ano

### 5. Configuração MCP
- ✅ `.cursor/mcp.json` configurado para projeto correto
- ✅ MCP conectado ao projeto Supabase `hsmrpjzenlrcgncgexsr`

## 📋 Próximos Passos

### Imediato
1. **Dashboard Next.js:**
   - Migrar gráficos do Streamlit
   - Criar componentes React
   - Implementar filtros e visualizações

### Próximas Fases
4. **Sistema de Upload Web:**
   - Criar página de upload no Next.js
   - Integrar com API routes
   - Implementar processamento assíncrono

5. **Dashboard Next.js:**
   - Migrar gráficos do Streamlit
   - Criar componentes React
   - Implementar filtros e visualizações

## 🔧 Melhorias Aplicadas

1. **Tratamento de erros:**
   - ✅ Filtro de linhas vazias (NaN) antes do processamento
   - ✅ Validação de campos obrigatórios (original_id, last_name, first_name, email, country, city)
   - ✅ Valores padrão para campos obrigatórios quando vazios (country='BR', city='Não informado')
   - ✅ Limpeza de valores NaN/Inf antes da serialização JSON
   - ✅ Carregamento completo de dados IBGE com paginação
   - ✅ Verificação de registros já enviados para evitar duplicatas

2. **Processamento:**
   - ✅ Carregamento de todos os municípios IBGE (5.289)
   - ✅ Detecção automática de registros já enviados
   - ✅ Processamento em lotes de 500 registros
   - ✅ Tratamento individual de conflitos (409)

## 📊 Estatísticas Atuais

- **Total de inscrições no banco:** 9.433 registros
  - 2023: 3.240 registros
  - 2024: 2.404 registros
  - 2025: 3.789 registros (USD: 2.173 + BRL: 1.616)
- **Municípios IBGE:** 5.289
- **Competições:** 5 (PTR 20, PTR 35, PTR 55, UTSB 100, FUN 7KM)
- **Receita total:** R$ 5.604.650,56
  - 2023: R$ 1.946.073,97
  - 2024: R$ 1.386.896,51
  - 2025: R$ 2.271.680,08

## 🔑 Credenciais (já configuradas)

- **Supabase URL:** `https://hsmrpjzenlrcgncgexsr.supabase.co`
- **Anon Key:** Configurado em `.env.local` (dashboard-utmb-web)
- **MCP Token:** Configurado em `.cursor/mcp.json`

## 📁 Estrutura do Projeto

```
dashboard_utmb/
├── scripts/
│   ├── upload_ibge_to_supabase.py
│   ├── upload_inscricoes_2023.py
│   ├── upload_inscricoes_2024.py
│   ├── upload_inscricoes_2025_usd.py
│   ├── upload_inscricoes_2025_brl.py
│   ├── detect_year_from_filename.py
│   └── detect_year_from_registration_date.py
├── docs/
│   ├── DETECCAO_ANO_FINAL.md
│   ├── COLUNA_ANO_INSCRICOES.md
│   └── IDENTIFICACAO_ANO.md
├── dashboard_grok.py (referência)
├── SUPABASE_SCHEMA.md
└── STATUS_PROJETO.md (este arquivo)

dashboard-utmb-web/
├── app/
│   ├── api/
│   ├── dashboard/
│   └── upload/
├── lib/
│   └── supabase/
└── supabase/
    └── migrations/
```

## 🎯 Objetivo Final

Criar dashboard completo em Next.js/Vercel com:
- Upload de Excel via web
- Processamento automático
- Visualizações interativas
- Comparação entre anos
- Métricas em tempo real


