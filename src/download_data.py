"""Snapshot Snowflake tables to parquet (read-only: only SELECTs are issued)."""

from pathlib import Path

from metaflow import Snowflake

SCHEMA = "pattern_db.data_science_stage"
RAW_DIR = Path("data/raw")
EXTERNAL_DIR = Path("data/external")

QUERIES = {
    "core_daily": """
        SELECT
            marketplace_id,
            partner_id,
            page_id,
            date,
            units,
            oos,
            promo_pct_off,
            avg_price_paid,
            buybox_pct,
            buybox_suppression_pct,
            ad_spend,
            prime_day,
            big_deals,
            black_fri,
            cyber_mon,
            christmas,
            ny_day
        FROM {core_table}
    """,
    # from Nikhil: https://patterninc.slack.com/archives/C0BAH9Q2PUG/p1781597229517349
    "page_views_daily": """
        WITH universe AS (
            SELECT DISTINCT cd.marketplace_id, cd.page_id, mrkt.country_code
            FROM {core_table} cd
            JOIN analytics_db.stg_catalog.stg_catalog__marketplaces mrkt
                ON cd.marketplace_id = mrkt.id
        )
        SELECT
            u.marketplace_id AS marketplace_id,
            u.page_id AS page_id,
            pv.date AS date,
            pv.total_page_views AS page_views
        FROM analytics_db.core.page_views pv
        JOIN universe u
            ON pv.asin = u.page_id AND pv.country_code = u.country_code
        WHERE pv.date >= '2019-01-01'
    """,
    # from Brad: https://patterninc.slack.com/archives/C0BAH9Q2PUG/p1781123500805869
    "hourly_sales_daily": """
        SELECT
            mrkt.id AS marketplace_id,
            prt.id AS partner_id,
            hr.order_date AS date,
            SUM(hr.quantity) AS partner_units
        FROM pattern_db.public.hourly_sales hr
        LEFT JOIN analytics_db.stg_catalog.stg_catalog__listings lst
            ON hr.listing_id = lst.listing_id
        LEFT JOIN analytics_db.stg_catalog.stg_catalog__marketplaces mrkt
            ON lst.marketplace_id = mrkt.id
        LEFT JOIN pc_fivetran_db.amaczar_public.vendors vnd
            ON hr.vendor_id = vnd.id
        LEFT JOIN analytics_db.stg_catalog.stg_catalog__partners prt
            ON vnd.vendor_name = prt.name
        WHERE mrkt.id IS NOT NULL
          AND prt.id IS NOT NULL
          AND hr.order_date >= '2019-01-01'
        GROUP BY 1, 2, 3
    """,
    # from Umesh: https://patterninc.slack.com/archives/C0BAH9Q2PUG/p1781159310398959?thread_ts=1781127601.348779&cid=C0BAH9Q2PUG
    # ASIN -> browse-node category path, US marketplace; the GBM keeps the
    # deepest node per page as a LightGBM categorical. DISTINCT collapses rows
    # that differ only by parent_node_id (dropped here).
    "jbe_page_nodes": """
        SELECT DISTINCT page_id, browse_node_id, depth
        FROM pattern_db.data_science_stage.jbe_browse_node
        WHERE marketplace_id = 1
          AND page_id IN (SELECT DISTINCT page_id FROM {core_table})
    """,
}


def download_query(query: str, dest: Path) -> None:
    """Run a SELECT and save the result as parquet."""
    con = Snowflake(
        integration="snowflake-default", schema="DATA_SCIENCE_STAGE", database="PATTERN_DB"
    ).cn
    try:
        df = con.cursor().execute(query).fetch_pandas_all()
    finally:
        con.close()
    df.columns = df.columns.str.lower()
    df.to_parquet(dest, index=False)
