from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import psycopg


DATA_CORTE_MOCK = "2024-10-31"

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "clientes": PROCESSED / "clientes_seed.csv",
    "veiculos": PROCESSED / "veiculos_seed_com_clientes.csv",
    "vin_share_servicos": PROCESSED / "vin_share_servicos_seed.csv",
}

COLUMN_MAPS = {
    "vin_share_servicos": {
        "Country": "country",
        "ScheduleID": "schedule_id",
        "MaintenanceID": "maintenance_id",
        "ServiceOrder": "service_order",
        "ServiceDate": "service_date",
        "ServiceOpenDate": "service_open_date",
        "ServiceClosedDate": "service_closed_date",
        "ServiceDeptCode": "service_dept_code",
        "ServiceRepairTypeCode": "service_repair_type_code",
        "ServiceType": "service_type",
        "ServiceCode": "service_code",
        "MaintenanceNumber": "maintenance_number",
        "DealerCode": "dealer_code",
        "MainSource": "main_source",
        "IsAgendaSchedule": "agenda_flag",
        "StatusUSA": "status_usa",
        "VIN_Hash": "vin_hash",
        "ModelYear": "model_year",
        "ModelName": "model_name",
        "KM": "km",
        "InvoiceDate": "invoice_date",
        "SalesDate": "sales_date",
        "DeliveryDate": "delivery_date",
        "RegistrationDate": "registration_date",
        "WarrantyStartDate": "warranty_start_date",
    }
}


def read_headers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError(f"CSV vazio: {path}")

    return [h.strip() for h in headers]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def target_columns(table: str, headers: list[str]) -> list[str]:
    mapping = COLUMN_MAPS.get(table)
    if mapping is None:
        return headers

    missing = sorted(header for header in headers if header not in mapping)
    if missing:
        raise ValueError(f"Colunas sem mapeamento para public.{table}: {missing}")

    return [mapping[header] for header in headers]


def copy_csv(conn: psycopg.Connection, table: str, path: Path) -> None:
    headers = read_headers(path)
    columns = target_columns(table, headers)

    columns_sql = ", ".join(quote_ident(c) for c in columns)
    copy_sql = f"""
        COPY public.{quote_ident(table)} ({columns_sql})
        FROM STDIN
        WITH (
            FORMAT csv,
            HEADER true,
            NULL '',
            ENCODING 'UTF8'
        )
    """

    print(f"Importando {path.name} -> public.{table} ({len(columns)} colunas)")

    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                while chunk := f.read(1024 * 1024):
                    copy.write(chunk)


def truncate_mock_data(conn: psycopg.Connection) -> None:
    print("Limpando dados mock atuais...")

    sql = """
        truncate table public.predicao_resultados restart identity cascade;
        truncate table public.ml_feature_refresh_queue restart identity cascade;
        truncate table public.vin_share_feature_snapshots restart identity cascade;
        truncate table public.vin_share_servicos restart identity cascade;
        truncate table public.veiculos restart identity cascade;
        truncate table public.clientes restart identity cascade;
    """

    with conn.cursor() as cur:
        cur.execute(sql)


def reset_sequences(conn: psycopg.Connection) -> None:
    print("Ajustando sequences...")

    sql = """
        select setval(
            pg_get_serial_sequence('public.clientes', 'id'),
            coalesce((select max(id) from public.clientes), 1),
            true
        );

        select setval(
            pg_get_serial_sequence('public.veiculos', 'id'),
            coalesce((select max(id) from public.veiculos), 1),
            true
        );
    """

    with conn.cursor() as cur:
        cur.execute(sql)


def enqueue_vins(conn: psycopg.Connection) -> None:
    print(f"Enfileirando VINs com data_corte={DATA_CORTE_MOCK}...")

    sql = """
        insert into public.ml_feature_refresh_queue (
            vin_hash,
            data_corte,
            status,
            tentativas,
            criado_em,
            atualizado_em
        )
        select distinct
            v.vin_hash,
            %(data_corte)s::date,
            'pending_features',
            0,
            now(),
            now()
        from public.veiculos v
        where v.vin_hash is not null
        on conflict (vin_hash, data_corte)
        do update set
            status = 'pending_features',
            erro = null,
            atualizado_em = now();
    """

    with conn.cursor() as cur:
        cur.execute(sql, {"data_corte": DATA_CORTE_MOCK})


def refresh_features(conn: psycopg.Connection, limit: int) -> None:
    print(f"Gerando snapshots/features no banco, limit={limit}...")

    sql = """
        select ml.refresh_pending_features(%(data_corte)s::date, %(limit)s);
    """

    with conn.cursor() as cur:
        cur.execute(sql, {"data_corte": DATA_CORTE_MOCK, "limit": limit})


def validate_counts(conn: psycopg.Connection) -> None:
    print("\nValidação:")

    sql = """
        select 'clientes' as tabela, count(*)::bigint as linhas from public.clientes
        union all
        select 'veiculos', count(*)::bigint from public.veiculos
        union all
        select 'vin_share_servicos', count(*)::bigint from public.vin_share_servicos
        union all
        select 'vin_share_feature_snapshots', count(*)::bigint from public.vin_share_feature_snapshots
        union all
        select 'ml_feature_refresh_queue', count(*)::bigint from public.ml_feature_refresh_queue
        union all
        select 'predicao_resultados', count(*)::bigint from public.predicao_resultados
        order by tabela;
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        for tabela, linhas in cur.fetchall():
            print(f"- {tabela}: {linhas}")

    queue_sql = """
        select status, count(*)::bigint
        from public.ml_feature_refresh_queue
        group by status
        order by status;
    """

    with conn.cursor() as cur:
        cur.execute(queue_sql)
        rows = cur.fetchall()

    print("\nFila:")
    for status, total in rows:
        print(f"- {status}: {total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-truncate", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Defina a variável DATABASE_URL com a connection string do Supabase.")

    for path in FILES.values():
        if not path.exists():
            raise FileNotFoundError(f"Arquivo esperado não encontrado: {path}")

    with psycopg.connect(database_url, autocommit=False) as conn:
        try:
            if not args.no_truncate:
                truncate_mock_data(conn)

            copy_csv(conn, "clientes", FILES["clientes"])
            copy_csv(conn, "veiculos", FILES["veiculos"])
            copy_csv(conn, "vin_share_servicos", FILES["vin_share_servicos"])

            reset_sequences(conn)
            enqueue_vins(conn)

            if not args.no_refresh:
                refresh_features(conn, args.limit)

            validate_counts(conn)

            conn.commit()
            print("\nCarga concluída com sucesso.")

        except Exception:
            conn.rollback()
            print("\nErro durante a carga. Rollback executado.")
            raise


if __name__ == "__main__":
    main()
