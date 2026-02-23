# Como Identificar o Ano de Inscrição

## Problema
Quando um arquivo Excel é enviado via upload, precisamos identificar qual ano (2023, 2024, 2025, 2026, etc.) cada inscrição pertence.

## Soluções Propostas

### 1. **Seleção Manual no Formulário (RECOMENDADO)**
O usuário seleciona o ano no formulário de upload antes de fazer upload do arquivo.

**Vantagens:**
- ✅ Controle total sobre o ano
- ✅ Evita erros de detecção automática
- ✅ Simples de implementar
- ✅ Funciona para uploads retroativos

**Implementação:**
```html
<select name="ano">
  <option value="2023">2023</option>
  <option value="2024">2024</option>
  <option value="2025">2025</option>
  <option value="2026">2026</option>
</select>
```

### 2. **Detecção Automática do Nome do Arquivo**
Tentar inferir o ano do nome do arquivo.

**Exemplos:**
- `UTMB - 2023 - USD.xlsx` → ano = 2023
- `Paraty_Brazil_by_UTMB__2025_ChatGPT_USD.xlsx` → ano = 2025
- `inscricoes_2026.xlsx` → ano = 2026

**Implementação:**
```python
import re

def extract_year_from_filename(filename):
    """Extrai o ano do nome do arquivo"""
    match = re.search(r'(20\d{2})', filename)
    if match:
        return int(match.group(1))
    return None
```

**Limitações:**
- ⚠️ Nem todos os arquivos têm o ano no nome
- ⚠️ Pode detectar ano errado (ex: data de exportação)

### 3. **Detecção pela Data de Registro**
Extrair o ano da coluna `Registration date`.

**Exemplos:**
- `2023-08-20 00:00:00` → ano = 2023
- `09/02/2025 18:35:29` → ano = 2025
- `19/08/2025 22:46:40` → ano = 2025

**Implementação:**
```python
def extract_year_from_date(registration_date):
    """Extrai o ano da data de registro"""
    if pd.isna(registration_date):
        return None
    return pd.to_datetime(registration_date).year
```

**Limitações:**
- ⚠️ Pode haver inscrições de anos diferentes no mesmo arquivo
- ⚠️ Arquivos retroativos podem ter datas de anos anteriores

### 4. **Solução Híbrida (RECOMENDADA PARA PRODUÇÃO)**

Combinar seleção manual com detecção automática como sugestão:

1. **Detectar automaticamente** do nome do arquivo ou data
2. **Preencher o campo** no formulário com o valor detectado
3. **Permitir correção manual** pelo usuário
4. **Validar** que o ano está entre 2023-2030

**Fluxo:**
```
1. Usuário seleciona arquivo
2. Sistema detecta ano automaticamente (nome do arquivo ou data)
3. Campo "Ano" é preenchido com sugestão
4. Usuário pode corrigir se necessário
5. Ao fazer upload, todos os registros recebem o ano selecionado
```

## Implementação no Sistema de Upload

### Estrutura da Tabela `upload_historico`
A tabela já tem o campo `ano` que armazena o ano do arquivo:

```sql
CREATE TABLE upload_historico (
    ...
    ano INTEGER NOT NULL,
    ...
    CONSTRAINT check_valid_ano_upload CHECK (ano >= 2023 AND ano <= 2030)
);
```

### Estrutura da Tabela `inscricoes`
Cada registro de inscrição tem o campo `ano`:

```sql
CREATE TABLE inscricoes (
    ...
    ano INTEGER NOT NULL,
    ...
    CONSTRAINT check_valid_ano CHECK (ano >= 2023 AND ano <= 2030)
);
```

### Processamento
Ao processar um arquivo Excel:

1. **Receber o ano** do formulário de upload
2. **Processar cada linha** do Excel
3. **Atribuir o ano** a todos os registros do arquivo
4. **Validar** que as datas de registro são compatíveis com o ano (opcional)
5. **Inserir** no banco com o ano correto

## Exemplo de Código

```python
def process_excel_file(file, ano_selecionado, file_type='USD'):
    """
    Processa arquivo Excel e insere no banco
    
    Args:
        file: Arquivo Excel
        ano_selecionado: Ano selecionado pelo usuário (2023, 2024, 2025, etc.)
        file_type: Tipo de arquivo ('USD' ou 'BRL')
    """
    df = pd.read_excel(file)
    
    # Atribuir ano a todas as linhas
    df['ano'] = ano_selecionado
    
    # Processar e inserir no banco
    for _, row in df.iterrows():
        insert_inscricao({
            'ano': ano_selecionado,
            'original_id': row['id'],
            'registration_date': row['Registration date'],
            # ... outros campos
        })
```

## Recomendações Finais

1. **Formulário de Upload**: Incluir campo obrigatório para seleção do ano
2. **Validação**: Verificar se o ano está entre 2023-2030
3. **Feedback**: Mostrar ao usuário quantos registros foram inseridos com o ano selecionado
4. **Histórico**: Armazenar o ano no `upload_historico` para auditoria
5. **Detecção Automática**: Implementar como sugestão, mas sempre permitir correção manual


