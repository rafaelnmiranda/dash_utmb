# Detecção de Ano do Evento - Solução Final

## ✅ Solução Implementada

A detecção de ano é feita **prioritariamente pelo nome do arquivo**, com fallback para a data de registro apenas se necessário.

## 🎯 Estratégia de Detecção

### 1. **Nome do Arquivo (Prioridade Máxima)** ✅

Extrai o ano diretamente do nome do arquivo usando regex:

```python
from scripts.detect_year_from_filename import detect_year

detection = detect_year(filename='UTMB - 2023 - USD.xlsx')
# Resultado: {'year': 2023, 'source': 'filename', 'confidence': 1.0}
```

**Formatos suportados:**
- `UTMB - 2023 - USD.xlsx` → 2023 ✅
- `UTMB - 2024 - USD.xlsx` → 2024 ✅
- `Paraty_Brazil_by_UTMB__2025_ChatGPT_USD.xlsx` → 2025 ✅
- `inscricoes_2026.xlsx` → 2026 ✅
- `vendas_2027_BRL.xlsx` → 2027 ✅
- `evento-2028-usd.xlsx` → 2028 ✅

**Validação:**
- Apenas anos entre 2023-2030 são aceitos
- Procura por padrão `20XX` no nome do arquivo

### 2. **Data de Registro (Fallback)** 

Se não conseguir detectar pelo nome do arquivo, usa a lógica de detecção pela `Registration date`:

```python
detection = detect_year(filename=None, df=dataframe)
# Usa a função detect_year_from_file() como fallback
```

## 📋 Como Usar

### Uso Básico

```python
from scripts.detect_year_from_filename import detect_year, apply_year_to_dataframe_smart

# Opção 1: Apenas detectar
detection = detect_year(filename='UTMB - 2023 - USD.xlsx')
print(f"Ano: {detection['year']}")
print(f"Fonte: {detection['source']}")  # 'filename' ou 'registration_date'

# Opção 2: Aplicar ao DataFrame
df_with_year, detection = apply_year_to_dataframe_smart(
    df=dataframe, 
    filename='UTMB - 2023 - USD.xlsx'
)
```

### Integração no Sistema de Upload

```python
def process_uploaded_file(file, filename):
    # Detectar ano
    detection = detect_year(filename=filename)
    
    if detection['year']:
        # Processar arquivo com o ano detectado
        df = pd.read_excel(file)
        df['ano'] = detection['year']
        # ... resto do processamento
    else:
        # Pedir confirmação manual ao usuário
        # ou usar detecção pela data de registro
        pass
```

## ✅ Resultados dos Testes

Todos os arquivos existentes foram testados com sucesso:

| Arquivo | Ano Detectado | Fonte | Status |
|---------|---------------|-------|--------|
| `UTMB - 2023 - USD.xlsx` | 2023 | filename | ✅ |
| `UTMB - 2024 - USD.xlsx` | 2024 | filename | ✅ |
| `Paraty_Brazil_by_UTMB__2025_ChatGPT_USD.xlsx` | 2025 | filename | ✅ |
| `PARATY_BRAZIL_BY_UTMB__2025_ChatGPT_BRL.xlsx` | 2025 | filename | ✅ |

## 🚀 Próximos Passos

1. **Integrar no sistema de upload**: Usar `detect_year()` no endpoint de upload
2. **Feedback ao usuário**: Mostrar o ano detectado e permitir correção se necessário
3. **Validação**: Garantir que o ano está entre 2023-2030
4. **Logging**: Registrar qual método foi usado (filename ou registration_date)

## 📝 Notas

- A detecção pelo nome do arquivo tem **100% de confiança** quando encontra um ano válido
- O fallback pela data de registro só é usado se o nome do arquivo não tiver ano
- Para arquivos futuros (2026, 2027, etc.), basta incluir o ano no nome do arquivo


