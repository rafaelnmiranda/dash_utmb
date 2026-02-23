#!/usr/bin/env python3
"""
Script para fazer upload dos dados do IBGE para o Supabase
Usa apenas requests (sem biblioteca supabase-py)
"""
import pandas as pd
import requests
import json
import os
from unidecode import unidecode
import re

# Configuração do Supabase
SUPABASE_URL = 'https://hsmrpjzenlrcgncgexsr.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhzbXJwanplbmxyY2duY2dleHNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIxMDc4NjEsImV4cCI6MjA3NzY4Mzg2MX0.9znq29O1OK0P6xBoll-tEw6mXyBbQv_H9BlDanpdtd0'

def norm_text(text):
    """Normaliza texto para comparação (remove acentos, espaços, etc)"""
    if pd.isna(text):
        return ""
    t = unidecode(str(text))
    t = re.sub(r'[^a-z0-9\s]', '', t.lower().strip())
    return t

def main():
    print("🚀 Iniciando upload dos dados do IBGE para o Supabase...")
    
    # Ler arquivo Excel
    file_path = 'municipios_IBGE.xlsx'
    if not os.path.exists(file_path):
        print(f"❌ Arquivo não encontrado: {file_path}")
        return
    
    print(f"📖 Lendo arquivo: {file_path}")
    df = pd.read_excel(file_path)
    
    print(f"✅ Arquivo carregado: {len(df)} municípios")
    
    # Preparar dados
    print("🔄 Processando dados...")
    df['city_norm'] = df['City'].apply(norm_text)
    
    # Remover duplicatas baseado em city_norm
    df = df.drop_duplicates(subset=['city_norm'], keep='first')
    print(f"✅ Após remoção de duplicatas: {len(df)} municípios únicos")
    
    # Preparar dados para inserção
    records = []
    for _, row in df.iterrows():
        records.append({
            'city': str(row['City']).strip(),
            'city_norm': row['city_norm'],
            'uf': str(row['UF']).strip(),
            'regiao': str(row['Região']).strip()
        })
    
    print(f"📤 Preparando para inserir {len(records)} registros...")
    
    # Headers para API do Supabase
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    
    # Inserir em lotes (Supabase tem limite de registros por request)
    batch_size = 1000
    total_inserted = 0
    total_errors = 0
    total_skipped = 0
    
    api_url = f"{SUPABASE_URL}/rest/v1/ibge_municipios"
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=batch
            )
            
            if response.status_code == 201:
                inserted = len(response.json()) if response.json() else len(batch)
                total_inserted += inserted
                print(f"✅ Lote {i//batch_size + 1}: {inserted} registros inseridos")
            elif response.status_code == 409 or 'duplicate' in response.text.lower():
                # Alguns já existem, tenta inserir individualmente
                print(f"⚠️  Lote {i//batch_size + 1}: Alguns registros já existem, inserindo individualmente...")
                for record in batch:
                    try:
                        r = requests.post(api_url, headers=headers, json=[record])
                        if r.status_code == 201:
                            total_inserted += 1
                        elif r.status_code == 409:
                            total_skipped += 1
                        else:
                            total_errors += 1
                            if total_errors <= 5:  # Mostra apenas os primeiros 5 erros
                                print(f"❌ Erro ao inserir {record['city']}: {r.status_code}")
                    except Exception as e:
                        total_errors += 1
                        if total_errors <= 5:
                            print(f"❌ Erro ao inserir {record['city']}: {str(e)[:100]}")
            else:
                print(f"❌ Erro no lote {i//batch_size + 1}: {response.status_code}")
                print(f"   Resposta: {response.text[:200]}")
                total_errors += len(batch)
        except Exception as e:
            print(f"❌ Erro no lote {i//batch_size + 1}: {str(e)[:200]}")
            total_errors += len(batch)
    
    print("\n" + "="*50)
    print(f"✅ Concluído!")
    print(f"📊 Total inserido: {total_inserted}")
    print(f"⏭️  Já existiam: {total_skipped}")
    if total_errors > 0:
        print(f"⚠️  Erros: {total_errors}")
    print("="*50)
    
    # Verificar quantos registros foram inseridos
    try:
        response = requests.get(
            f"{api_url}?select=id",
            headers={'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'},
            params={'select': 'id', 'limit': 1}
        )
        # Contar via query
        count_response = requests.get(
            f"{api_url}",
            headers={'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'},
            params={'select': 'id', 'limit': 1}
        )
        if 'content-range' in count_response.headers:
            range_header = count_response.headers['content-range']
            total = range_header.split('/')[-1] if '/' in range_header else '?'
            print(f"📈 Total de registros no banco: {total}")
    except:
        pass

if __name__ == '__main__':
    main()
