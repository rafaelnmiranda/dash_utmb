"""
Detecção de ano do evento baseado no nome do arquivo
Prioriza o nome do arquivo, usa Registration date apenas como fallback
"""
import re
import pandas as pd
import sys
import os

# Adicionar o diretório pai ao path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.detect_year_from_registration_date import detect_year_from_file
except ImportError:
    # Se não conseguir importar, criar função stub
    def detect_year_from_file(df, registration_date_col='Registration date'):
        return {'year': None, 'confidence': 0.0, 'method': None}


def extract_year_from_filename(filename):
    """
    Extrai o ano do nome do arquivo usando regex
    
    Procura por padrões como:
    - 2023, 2024, 2025, 2026, etc.
    - Deve ser um ano entre 2023 e 2030
    
    Args:
        filename: Nome do arquivo (ex: "UTMB - 2023 - USD.xlsx")
        
    Returns:
        int: Ano encontrado (ex: 2023, 2024, 2025)
        None: Se não encontrar um ano válido
    """
    if not filename:
        return None
    
    # Procurar por anos de 4 dígitos começando com 20
    matches = re.findall(r'20\d{2}', str(filename))
    
    for match in matches:
        year = int(match)
        # Validar que está no range esperado (2023-2030)
        if 2023 <= year <= 2030:
            return year
    
    return None


def detect_year(file_obj=None, filename=None, df=None, registration_date_col='Registration date'):
    """
    Detecta o ano do evento usando múltiplas estratégias (em ordem de prioridade):
    
    1. Nome do arquivo (prioridade máxima)
    2. Data de registro (fallback)
    
    Args:
        file_obj: Objeto de arquivo (opcional, para obter filename)
        filename: Nome do arquivo (opcional)
        df: DataFrame com os dados (opcional, para fallback)
        registration_date_col: Nome da coluna com data de registro
        
    Returns:
        dict: {
            'year': int,  # Ano detectado
            'source': str,  # 'filename' ou 'registration_date'
            'confidence': float,  # Confiança (0-1)
            'method': str,  # Detalhes do método usado
        }
    """
    # Obter filename se não fornecido
    if not filename and file_obj:
        filename = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    
    # Estratégia 1: Extrair do nome do arquivo
    if filename:
        year_from_filename = extract_year_from_filename(filename)
        if year_from_filename:
            return {
                'year': year_from_filename,
                'source': 'filename',
                'confidence': 1.0,
                'method': f'Extraído do nome do arquivo: {filename}'
            }
    
    # Estratégia 2: Usar data de registro como fallback
    if df is not None:
        detection = detect_year_from_file(df, registration_date_col)
        if detection['year']:
            return {
                'year': detection['year'],
                'source': 'registration_date',
                'confidence': detection['confidence'],
                'method': detection['method'],
                'fallback_warning': True
            }
    
    # Se nenhuma estratégia funcionou
    return {
        'year': None,
        'source': None,
        'confidence': 0.0,
        'method': 'Nenhum método conseguiu detectar o ano'
    }


def apply_year_to_dataframe_smart(df, filename=None, registration_date_col='Registration date'):
    """
    Aplica detecção de ano a um DataFrame usando a estratégia inteligente
    
    Args:
        df: DataFrame com as inscrições
        filename: Nome do arquivo (opcional)
        registration_date_col: Nome da coluna com data de registro
        
    Returns:
        DataFrame: DataFrame com coluna 'ano' adicionada
    """
    detection = detect_year(filename=filename, df=df, registration_date_col=registration_date_col)
    
    if detection['year']:
        df = df.copy()
        df['ano'] = detection['year']
        return df, detection
    else:
        # Se não conseguiu detectar, retorna sem ano (precisa de input manual)
        return df, detection


# Teste da função
if __name__ == '__main__':
    import sys
    
    test_files = [
        'UTMB - 2023 - USD.xlsx',
        'UTMB - 2024 - USD.xlsx',
        'Paraty_Brazil_by_UTMB__2025_ChatGPT_USD 2025-10-03 01_50_37.xlsx',
        'PARATY_BRAZIL_BY_UTMB__2025_ChatGPT_BRL 2025-10-03 02_51_05.xlsx',
        'inscricoes_2026.xlsx',  # Exemplo futuro
    ]
    
    print('='*70)
    print('TESTE DE DETECÇÃO DE ANO DO NOME DO ARQUIVO')
    print('='*70)
    
    for filename in test_files:
        detection = detect_year(filename=filename)
        print(f'\n📁 {filename}')
        print(f'   Ano detectado: {detection["year"]}')
        print(f'   Fonte: {detection["source"]}')
        print(f'   Confiança: {detection["confidence"]:.1%}')
        print(f'   Método: {detection["method"]}')

