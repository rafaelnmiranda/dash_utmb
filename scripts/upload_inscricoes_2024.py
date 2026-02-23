#!/usr/bin/env python3
"""
Script para processar e enviar inscrições de 2024 para o Supabase
"""
import pandas as pd
import requests
import unicodedata
import re
import difflib
from datetime import datetime
import json
import math

# Configuração do Supabase
SUPABASE_URL = 'https://hsmrpjzenlrcgncgexsr.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhzbXJwanplbmxyY2duY2dleHNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIxMDc4NjEsImV4cCI6MjA3NzY4Mzg2MX0.9znq29O1OK0P6xBoll-tEw6mXyBbQv_H9BlDanpdtd0'

# Taxa de câmbio
TAXA_CAMBIO = 5.5

def norm_text(text):
    """Normaliza texto para comparação"""
    if pd.isna(text):
        return ""
    t = unicodedata.normalize('NFKD', str(text))
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9\s]', '', t.lower().strip())

def standardize_nationality(value):
    """Padroniza nacionalidade"""
    if pd.isnull(value):
        return value
    value = str(value).strip().upper()
    mapping = {"BRASIL": "BR", "BRAZIL": "BR"}
    return mapping.get(value, value)

def standardize_competition(value):
    """Padroniza nomes das competições"""
    if pd.isnull(value):
        return value
    competition_mapping = {
        "UTSB 110": "UTSB 100",
        "UTSB 100 (108km)": "UTSB 100",
        "PTR 55 (58km)": "PTR 55",
        "PTR 35 (34km)": "PTR 35",
        "PTR 20 (25km)": "PTR 20",
        "Fun 7km": "FUN 7KM",
        "FUN 7KM": "FUN 7KM",
        "PTR20": "PTR 20",
        "PTR35": "PTR 35",
        "PTR55": "PTR 55",
        "Kids": "KIDS",
        "Kids Race": "KIDS",
        "KIDS_EVENT": "KIDS"
    }
    value = str(value).strip().upper()
    return competition_mapping.get(value, value)

def correct_city(city, ibge_data, cutoff=0.8):
    """Corrige nome da cidade usando fuzzy matching"""
    if pd.isna(city) or str(city).strip() == '':
        return city
    
    city_norm = norm_text(city)
    city_choices = [norm_text(c['city']) for c in ibge_data]
    
    matches = difflib.get_close_matches(city_norm, city_choices, n=1, cutoff=cutoff)
    if matches:
        # Encontrar cidade original correspondente
        for ibge_item in ibge_data:
            if norm_text(ibge_item['city']) == matches[0]:
                return ibge_item['city']
    
    return city

def load_ibge_from_supabase():
    """Carrega dados IBGE do Supabase"""
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}'
    }
    
    # Carregar todos os municípios (pode precisar de múltiplas requisições)
    all_data = []
    offset = 0
    limit = 1000
    
    while True:
        response = requests.get(
            f'{SUPABASE_URL}/rest/v1/ibge_municipios?select=*&limit={limit}&offset={offset}',
            headers=headers
        )
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        offset += limit
        if len(data) < limit:
            break
    
    return all_data

def calculate_age(birthdate):
    """Calcula idade a partir da data de nascimento"""
    if pd.isna(birthdate):
        return None
    try:
        if isinstance(birthdate, str):
            birthdate = pd.to_datetime(birthdate, errors='coerce')
        if pd.isna(birthdate):
            return None
        today = datetime.now()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age
    except:
        return None

def get_ibge_info(city, ibge_data):
    """Obtém UF e região do IBGE"""
    if pd.isna(city):
        return None, None
    
    city_norm = norm_text(city)
    for ibge_item in ibge_data:
        if norm_text(ibge_item['city']) == city_norm:
            return ibge_item.get('uf'), ibge_item.get('regiao')
    
    return None, None

def clean_for_json(obj):
    """Remove valores NaN e inf de um objeto para serialização JSON"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if pd.isna(obj) or math.isinf(obj):
            return None
        return obj
    elif pd.isna(obj):
        return None
    elif isinstance(obj, str):
        # Verificar se string vazia ou 'nan'
        if obj.strip() == '' or obj.lower() == 'nan':
            return None
        return obj
    else:
        return obj

def map_row_to_inscricao(row, ano, ibge_data, moeda_original='USD'):
    """Mapeia uma linha do Excel para o formato da tabela inscricoes"""
    
    # Corrigir cidade
    city_corrected = correct_city(row.get('City', ''), ibge_data)
    uf, regiao = get_ibge_info(city_corrected, ibge_data)
    
    # Normalizar valores monetários
    registration_amount = pd.to_numeric(row.get('Registration amount', 0), errors='coerce')
    discounts_amount = pd.to_numeric(row.get('Discounts amount', 0), errors='coerce')
    
    # Substituir NaN por 0
    registration_amount = 0.0 if pd.isna(registration_amount) else float(registration_amount)
    discounts_amount = 0.0 if pd.isna(discounts_amount) else float(discounts_amount)
    
    # Converter USD para BRL
    if moeda_original == 'USD':
        registration_amount *= TAXA_CAMBIO
        discounts_amount *= TAXA_CAMBIO
    
    total_registration_amount = registration_amount + discounts_amount
    
    # Processar data de nascimento
    birthdate = row.get('Birthdate')
    if birthdate and pd.notna(birthdate):
        # Tentar diferentes formatos
        birthdate = pd.to_datetime(birthdate, errors='coerce', dayfirst=True)
        if pd.isna(birthdate):
            # Tentar formato específico dd-mm-yyyy
            try:
                birthdate_str = str(row.get('Birthdate', ''))
                if isinstance(birthdate_str, str) and '-' in birthdate_str:
                    parts = birthdate_str.split('-')
                    if len(parts) == 3:
                        birthdate = pd.to_datetime(f"{parts[2]}-{parts[1]}-{parts[0]}", errors='coerce')
            except:
                pass
        if pd.isna(birthdate):
            birthdate = None
    else:
        birthdate = None
    
    idade_calculada = calculate_age(birthdate)
    
    # Processar data de registro
    registration_date = row.get('Registration date')
    if registration_date:
        registration_date = pd.to_datetime(registration_date, errors='coerce')
        if pd.isna(registration_date):
            registration_date = None
    else:
        registration_date = None
    
    # Validar campos obrigatórios
    original_id = str(row.get('id', '')).strip()
    if not original_id or original_id == 'nan' or original_id == 'None':
        raise ValueError(f"original_id vazio ou inválido na linha {row.name if hasattr(row, 'name') else 'N/A'}")
    
    booking_ref = str(row.get('Booking reference', '')).strip()
    if not booking_ref or booking_ref == 'nan':
        booking_ref = f"REF_{original_id}"  # Gerar referência se vazia
    
    last_name = str(row.get('Last name', '')).strip()
    if not last_name or last_name == 'nan':
        raise ValueError(f"last_name vazio na linha {row.name if hasattr(row, 'name') else 'N/A'}")
    
    first_name = str(row.get('First name', '')).strip()
    if not first_name or first_name == 'nan':
        raise ValueError(f"first_name vazio na linha {row.name if hasattr(row, 'name') else 'N/A'}")
    
    email = str(row.get('Email', '')).strip()
    if not email or email == 'nan':
        raise ValueError(f"email vazio na linha {row.name if hasattr(row, 'name') else 'N/A'}")
    
    # Mapear campos
    inscricao = {
        'original_id': original_id,
        'booking_reference': booking_ref,
        'last_name': last_name,
        'first_name': first_name,
        'email': email,
        'birthdate': birthdate.strftime('%Y-%m-%d') if birthdate and not pd.isna(birthdate) else None,
        'registration_date': registration_date.strftime('%Y-%m-%d %H:%M:%S') if registration_date and not pd.isna(registration_date) else None,
        'status': str(row.get('Status', 'COMPLETED')).upper(),
        'competition': standardize_competition(row.get('Competition', '')),
        'competition_original': str(row.get('Competition', '')),
        'registration_amount': registration_amount,
        'total_registration_amount': total_registration_amount,
        'discounts_amount': discounts_amount,
        'country': str(row.get('Country', '')),
        'city': str(city_corrected),
        'city_original': str(row.get('City', '')),
        'nationality': standardize_nationality(row.get('Nationality', '')),
        'nationality_original': str(row.get('Nationality', '')),
        'ano': ano,
        'moeda_original': moeda_original,
        'discount_code': str(row.get('Discount code', '')) if pd.notna(row.get('Discount code')) else None,
        'phone_number': str(row.get('Phone number', '')) if pd.notna(row.get('Phone number')) else None,
        'address': str(row.get('information_address', '')) if pd.notna(row.get('information_address')) else str(row.get('Address', '')) if pd.notna(row.get('Address')) else None,
        'team_club': str(row.get('Team / Club', '')) if pd.notna(row.get('Team / Club')) else None,
        'zip_code': str(row.get('Zip code', '')) if pd.notna(row.get('Zip code')) else None,
        'tshirt_size_woman': str(row.get('T-shirt size (woman)', '')) if pd.notna(row.get('T-shirt size (woman)', '')) else None,
        'tshirt_size_man': str(row.get('T-shirt size (man)', '')) if pd.notna(row.get('T-shirt size (man)', '')) else None,
        'utmb_indexes': str(row.get('UTMB indexes', '')) if pd.notna(row.get('UTMB indexes')) else None,
        'document_status': str(row.get('Document status', '')) if pd.notna(row.get('Document status')) else None,
        'schedule_bib_retrieval': str(row.get('Schedule your Bib retrieval time', '')) if pd.notna(row.get('Schedule your Bib retrieval time')) else None,
        'idade_calculada': idade_calculada if idade_calculada is not None else None,
        'uf': uf,
        'regiao': regiao
    }
    
    return inscricao

def load_existing_ids(ano):
    """Carrega todos os original_id já enviados para o ano especificado"""
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}'
    }
    
    all_ids = set()
    offset = 0
    limit = 1000
    
    while True:
        response = requests.get(
            f'{SUPABASE_URL}/rest/v1/inscricoes?ano=eq.{ano}&select=original_id&limit={limit}&offset={offset}',
            headers=headers
        )
        if response.status_code != 200:
            break
        data = response.json()
        if not data:
            break
        all_ids.update([str(r['original_id']) for r in data])
        offset += limit
        if len(data) < limit:
            break
    
    return all_ids

def main():
    print("🚀 Iniciando processamento de inscrições de 2024...")
    
    # Carregar IBGE do Supabase
    print("📖 Carregando dados IBGE do Supabase...")
    ibge_data = load_ibge_from_supabase()
    print(f"✅ Carregados {len(ibge_data)} municípios do IBGE")
    
    # Carregar IDs já enviados
    print("\n📖 Verificando registros já enviados...")
    existing_ids = load_existing_ids(2024)
    print(f"✅ Já existem {len(existing_ids)} registros no banco")
    
    # Ler arquivo Excel
    file_path = 'UTMB - 2024 - USD.xlsx'
    print(f"\n📖 Lendo arquivo: {file_path}")
    df = pd.read_excel(file_path, sheet_name=0)
    print(f"✅ Arquivo carregado: {len(df)} registros")
    
    # Detectar ano do arquivo
    ano = 2024
    moeda_original = 'USD'
    
    # Padronizar competições
    df['Competition'] = df['Competition'].apply(standardize_competition)
    
    # Excluir KIDS
    df = df[~df['Competition'].str.contains("KIDS", na=False, case=False)]
    print(f"✅ Após excluir KIDS: {len(df)} registros")
    
    # Filtrar linhas vazias (sem ID válido)
    df = df[df['id'].notna()].copy()
    df = df[df['id'].astype(str).str.strip() != ''].copy()
    df = df[df['id'].astype(str).str.lower() != 'nan'].copy()
    print(f"✅ Após excluir linhas vazias: {len(df)} registros")
    
    # Filtrar apenas registros que ainda não foram enviados
    df['id_str'] = df['id'].astype(str)
    df_new = df[~df['id_str'].isin(existing_ids)].copy()
    print(f"✅ Registros novos para enviar: {len(df_new)} (de {len(df)} total)")
    
    if len(df_new) == 0:
        print("\n✅ Todos os registros já foram enviados!")
        return
    
    # Processar e mapear registros
    print("\n🔄 Processando registros...")
    inscricoes = []
    errors = []
    
    for idx, row in df_new.iterrows():
        try:
            inscricao = map_row_to_inscricao(row, ano, ibge_data, moeda_original)
            inscricoes.append(inscricao)
        except Exception as e:
            errors.append({
                'row': idx,
                'error': str(e),
                'id': row.get('id', 'N/A')
            })
    
    print(f"✅ Processados: {len(inscricoes)} registros")
    if errors:
        print(f"⚠️  Erros: {len(errors)} registros")
        if len(errors) <= 10:
            for err in errors:
                print(f"   - Linha {err['row']}: {err['error']}")
    
    # Criar registro de upload
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    
    upload_record = {
        'file_name': file_path,
        'file_type': moeda_original,
        'ano': ano,
        'file_size': None,
        'total_registros': len(df_new),
        'status': 'processing'
    }
    
    print("\n📤 Criando registro de upload...")
    response = requests.post(
        f'{SUPABASE_URL}/rest/v1/upload_historico',
        headers=headers,
        json=upload_record
    )
    
    if response.status_code == 201:
        upload_id = response.json()[0]['id']
        print(f"✅ Upload ID: {upload_id}")
    else:
        print(f"❌ Erro ao criar registro de upload: {response.status_code}")
        print(f"   Resposta: {response.text[:200]}")
        return
    
    # Inserir inscrições em lotes
    batch_size = 500
    total_inserted = 0
    total_updated = 0
    total_errors = 0
    
    print(f"\n📤 Inserindo {len(inscricoes)} registros em lotes de {batch_size}...")
    
    for i in range(0, len(inscricoes), batch_size):
        batch = inscricoes[i:i + batch_size]
        # Limpar valores NaN antes de enviar
        batch_clean = [clean_for_json(insc) for insc in batch]
        try:
            response = requests.post(
                f'{SUPABASE_URL}/rest/v1/inscricoes',
                headers=headers,
                json=batch_clean
            )
            
            if response.status_code == 201:
                inserted = len(response.json()) if response.json() else len(batch)
                total_inserted += inserted
                print(f"✅ Lote {i//batch_size + 1}: {inserted} registros inseridos")
            elif response.status_code == 409:
                # Alguns já existem, tentar inserir individualmente
                print(f"⚠️  Lote {i//batch_size + 1}: Alguns registros já existem, processando individualmente...")
                for record in batch:
                    try:
                        record_clean = clean_for_json(record)
                        r = requests.post(
                            f'{SUPABASE_URL}/rest/v1/inscricoes',
                            headers=headers,
                            json=[record_clean]
                        )
                        if r.status_code == 201:
                            total_inserted += 1
                        elif r.status_code == 409:
                            total_updated += 1
                        else:
                            total_errors += 1
                            if total_errors <= 5:
                                print(f"❌ Erro: {r.status_code} - {r.text[:100]}")
                    except Exception as e:
                        total_errors += 1
                        if total_errors <= 5:
                            print(f"❌ Erro ao inserir: {str(e)[:100]}")
            else:
                print(f"❌ Erro no lote {i//batch_size + 1}: {response.status_code}")
                print(f"   Resposta: {response.text[:200]}")
                total_errors += len(batch)
        except Exception as e:
            print(f"❌ Erro no lote {i//batch_size + 1}: {str(e)[:200]}")
            total_errors += len(batch)
    
    # Atualizar registro de upload
    update_data = {
        'status': 'completed' if total_errors == 0 else 'completed_with_errors',
        'registros_processados': len(inscricoes),
        'registros_inseridos': total_inserted,
        'registros_atualizados': total_updated,
        'registros_ignorados': 0,
        'processed_at': datetime.now().isoformat(),
        'erros': json.dumps(errors[:10]) if errors else None
    }
    
    requests.patch(
        f'{SUPABASE_URL}/rest/v1/upload_historico?id=eq.{upload_id}',
        headers=headers,
        json=update_data
    )
    
    print("\n" + "="*70)
    print("✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"📊 Total processado: {len(inscricoes)}")
    print(f"✅ Inseridos: {total_inserted}")
    print(f"🔄 Atualizados: {total_updated}")
    if total_errors > 0:
        print(f"❌ Erros: {total_errors}")
    print("="*70)

if __name__ == '__main__':
    main()

