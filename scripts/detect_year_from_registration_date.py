"""
Função para detectar o ano do evento baseado na data de registro
Lógica: out/YY a set/YY+1 = evento YY+1
Mas pode variar dependendo do padrão de datas no arquivo
"""
import pandas as pd
from datetime import datetime


def detect_year_from_registration_date(date_obj, use_september=True):
    """
    Detecta o ano do evento baseado na data de registro
    
    Lógica padrão: out/YY a set/YY+1 = evento YY+1
    - Se mês >= 10 (outubro-dezembro): evento = ano + 1
    - Se mês < 10 (janeiro-setembro): evento = mesmo ano
    
    Mas se use_september=True:
    - Se mês >= 9 (setembro-dezembro): evento = ano + 1
    - Se mês < 9 (janeiro-agosto): evento = mesmo ano
    
    Args:
        date_obj: Data de registro (datetime, string ou pd.Timestamp)
        use_september: Se True, inclui setembro no período do evento seguinte
        
    Returns:
        int: Ano do evento (ex: 2023, 2024, 2025)
        None: Se a data for inválida
    """
    if pd.isna(date_obj):
        return None
    
    # Converter para datetime se necessário
    if isinstance(date_obj, str):
        date_obj = pd.to_datetime(date_obj, errors='coerce', dayfirst=True)
    
    if pd.isna(date_obj):
        return None
    
    month = date_obj.month
    year = date_obj.year
    
    # Determinar o mês de corte
    # Lógica: out/YY a set/YY+1 = evento YY+1
    # Mas setembro pode ser parte do período dependendo do contexto
    cutoff_month = 9 if use_september else 10
    
    # Se está no período de vendas (set/out até dezembro), o evento é do ano seguinte
    if month >= cutoff_month:
        event_year = year + 1
    # Se está no período inicial (janeiro até ago/set), o evento é do mesmo ano
    else:
        event_year = year
    
    return event_year


def detect_year_from_file(df, registration_date_col='Registration date'):
    """
    Detecta o ano do evento analisando o padrão de datas no arquivo
    
    Esta função analisa todas as datas do arquivo e determina qual lógica
    usar (com ou sem setembro) baseado no padrão mais comum.
    
    Args:
        df: DataFrame com as inscrições
        registration_date_col: Nome da coluna com a data de registro
        
    Returns:
        dict: {
            'year': int,  # Ano mais provável do evento
            'confidence': float,  # Confiança (0-1)
            'method': str,  # 'v1' (mês >= 10) ou 'v2' (mês >= 9)
            'dates_range': tuple,  # (min_date, max_date)
        }
    """
    if registration_date_col not in df.columns:
        return {
            'year': None,
            'confidence': 0.0,
            'method': None,
            'dates_range': (None, None)
        }
    
    # Converter datas
    df = df.copy()
    df[registration_date_col] = pd.to_datetime(
        df[registration_date_col], 
        errors='coerce', 
        dayfirst=True
    )
    
    # Remover datas inválidas
    valid_dates = df[registration_date_col].dropna()
    
    if len(valid_dates) == 0:
        return {
            'year': None,
            'confidence': 0.0,
            'method': None,
            'dates_range': (None, None)
        }
    
    # Testar ambas as versões
    v1_years = valid_dates.apply(
        lambda x: detect_year_from_registration_date(x, use_september=False)
    )
    v2_years = valid_dates.apply(
        lambda x: detect_year_from_registration_date(x, use_september=True)
    )
    
    # Contar frequência de cada ano em cada versão
    v1_counts = v1_years.value_counts()
    v2_counts = v2_years.value_counts()
    
    # Verificar se há datas em setembro
    has_september = (valid_dates.dt.month == 9).any()
    
    # Se há datas em setembro E o arquivo parece ser de um evento específico,
    # usar a versão que inclui setembro
    # Verificar se v2 tem mais consenso quando há setembro
    v1_unique_years = len(v1_counts)
    v2_unique_years = len(v2_counts)
    
    # Se há setembro e v2 tem mais consenso (menos anos únicos), usar v2
    if has_september and v2_unique_years < v1_unique_years:
        method = 'v2'
        year_counts = v2_counts
        use_september = True
    # Caso contrário, usar v1 (padrão)
    else:
        method = 'v1'
        year_counts = v1_counts
        use_september = False
    
    # Ano mais provável é o mais frequente
    most_likely_year = int(year_counts.index[0])
    confidence = year_counts.iloc[0] / len(valid_dates)
    
    return {
        'year': most_likely_year,
        'confidence': confidence,
        'method': method,
        'dates_range': (valid_dates.min(), valid_dates.max()),
        'use_september': use_september
    }


def apply_year_to_dataframe(df, registration_date_col='Registration date', 
                            use_september=None, target_year=None):
    """
    Aplica a detecção de ano a um DataFrame completo
    
    Args:
        df: DataFrame com as inscrições
        registration_date_col: Nome da coluna com a data de registro
        use_september: Se None, detecta automaticamente. Se True/False, usa valor específico
        target_year: Se fornecido, força este ano para todos os registros
        
    Returns:
        DataFrame: DataFrame com coluna 'ano' adicionada
    """
    df = df.copy()
    
    # Se target_year foi fornecido, usar diretamente
    if target_year is not None:
        df['ano'] = target_year
        return df
    
    # Detectar automaticamente se use_september não foi especificado
    if use_september is None:
        detection = detect_year_from_file(df, registration_date_col)
        use_september = detection.get('use_september', False)
        detected_year = detection.get('year')
        
        # Se a detecção tem alta confiança, usar o ano detectado
        if detection.get('confidence', 0) > 0.8:
            df['ano'] = detected_year
            return df
    
    # Aplicar detecção linha por linha
    df['ano'] = df[registration_date_col].apply(
        lambda x: detect_year_from_registration_date(x, use_september=use_september)
    )
    
    return df


# Teste da função
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        df = pd.read_excel(file_path, nrows=100)
        
        print(f"📁 Analisando: {file_path}")
        detection = detect_year_from_file(df)
        
        print(f"\n✅ Resultado da detecção:")
        print(f"   Ano detectado: {detection['year']}")
        print(f"   Confiança: {detection['confidence']:.1%}")
        print(f"   Método: {detection['method']} ({'inclui setembro' if detection.get('use_september') else 'não inclui setembro'})")
        print(f"   Range de datas: {detection['dates_range'][0]} a {detection['dates_range'][1]}")

