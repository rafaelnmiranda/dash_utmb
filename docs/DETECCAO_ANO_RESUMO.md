# Detecção Automática de Ano por Registration Date

## ✅ Conclusão

A lógica de detecção de ano pela `Registration date` **FUNCIONA**, mas precisa de ajustes dependendo do arquivo.

## 📋 Lógica Implementada

### Regra Principal
**Período de vendas: out/YY a set/YY+1 = evento YY+1**

- **Outubro a Dezembro (mês >= 10)**: Evento do ano seguinte
- **Janeiro a Setembro (mês < 10)**: Evento do mesmo ano

### Exceção: Setembro
Para alguns arquivos, setembro também faz parte do período do evento seguinte:
- **Setembro a Dezembro (mês >= 9)**: Evento do ano seguinte
- **Janeiro a Agosto (mês < 9)**: Evento do mesmo ano

## 📊 Resultados dos Testes

### ✅ Arquivos que Funcionam Perfeitamente

1. **2023** (UTMB - 2023 - USD.xlsx)
   - Range: Dec/2022 a Sep/2023
   - Detecção: 2023 ✅
   - Método: mês >= 10

2. **2025** (PARATY_BRAZIL_BY_UTMB__2025_ChatGPT_BRL.xlsx)
   - Range: Jan/2025 a Sep/2025
   - Detecção: 2025 ✅
   - Método: mês >= 10

### ⚠️ Arquivo que Precisa de Ajuste

3. **2024** (UTMB - 2024 - USD.xlsx)
   - Range: Sep/2023 a Sep/2024
   - Detecção padrão: 2023 ❌
   - Detecção com setembro: 2024 ✅
   - Método necessário: mês >= 9

## 🔧 Solução Implementada

A função `detect_year_from_file()` detecta automaticamente qual lógica usar:

1. **Analisa todas as datas** do arquivo
2. **Testa ambas as versões** (com e sem setembro)
3. **Escolhe a versão** com mais consenso (menos anos únicos)
4. **Se há datas em setembro** e v2 tem mais consenso, usa v2

### Código de Uso

```python
from scripts.detect_year_from_registration_date import detect_year_from_file, apply_year_to_dataframe

# Detectar automaticamente
detection = detect_year_from_file(df)
print(f"Ano detectado: {detection['year']}")
print(f"Confiança: {detection['confidence']:.1%}")

# Aplicar ao DataFrame
df_with_year = apply_year_to_dataframe(df)
```

## 📝 Recomendação

1. **Usar detecção automática** como padrão
2. **Permitir correção manual** se a detecção falhar
3. **Mostrar confiança** ao usuário (se < 80%, alertar)
4. **Armazenar método usado** no `upload_historico` para auditoria

## 🎯 Próximos Passos

1. Integrar no sistema de upload
2. Adicionar validação de confiança
3. Permitir override manual se necessário
4. Testar com arquivos de 2026 quando disponíveis


