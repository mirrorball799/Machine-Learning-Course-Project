"""
法国月度天气数据下载脚本

数据来源: Météo-France / data.gouv.fr
数据集: "Données climatologiques de base mensuelles"
时间范围: 2006年12月 至 2010年11月
目标变量: RR, NBJRR1, NBJRR5, NBJRR10, NBJBROU

使用方法:
  方案 A (推荐): 在 https://meteo.data.gouv.fr 手动下载 CSV
  方案 B: 注册 Météo-France API 后运行本脚本
    python weather_download.py --api-key YOUR_TOKEN

输出: weather_monthly_200612_201011.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ============ 配置 ============
# 家庭用电数据集来自法国一户家庭，大致位于巴黎地区
# 巴黎最近的 Météo-France 主要站点
STATION_ID = "75114001"  # Paris-Montsouris (巴黎最长历史记录站)
# 备选站点:
#   "07156001" - Lille-Lesquin
#   "75114001" - Paris-Montsouris
#   "33281001" - Bordeaux-Mérignac

START_DATE = "2006-12-01T00:00:00Z"
END_DATE = "2010-11-30T23:59:59Z"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_FILE = "weather_monthly.csv"

# ============ 方案 A: 手动下载说明 ============

MANUAL_DOWNLOAD_GUIDE = """
============================================================================
方案 A: 手动下载天气数据 (无需编程)
============================================================================

1. 打开浏览器访问:
   https://meteo.data.gouv.fr

2. 在搜索框搜索 "Base mensuelles" 或 "données climatologiques mensuelles"

3. 选择数据集 "Données climatologiques de base mensuelles"

4. 筛选条件:
   - Station: Paris-Montsouris (或选择离目标家庭最近的站点)
   - Période: 2006-12 到 2010-11
   - Format: CSV

5. 下载 CSV 文件

6. 将下载的文件命名为 weather_monthly.csv
   放在 data/ 目录下

7. 文件应包含以下列:
   - DATE: 日期 (YYYY-MM)
   - RR: 月累计降水量 (mm)
   - NBJRR1: 日降水 ≥ 1mm 的天数
   - NBJRR5: 日降水 ≥ 5mm 的天数
   - NBJRR10: 日降水 ≥ 10mm 的天数
   - NBJBROU: 雾天数

============================================================================
"""

# ============ 方案 B: API 自动下载 ============


def get_api_token(client_id: str, client_secret: str) -> str:
    """获取 Météo-France API OAuth2 token"""
    resp = requests.post(
        "https://portail-api.meteofrance.fr/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def list_stations(token: str, department: str = "75") -> list:
    """列出指定省份的气象站点"""
    resp = requests.get(
        "https://portail-api.meteofrance.fr/clim/v1/stations",
        params={"departement": department, "step": "1mo"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def order_data(token: str, station_id: str, date_deb: str, date_fin: str) -> str:
    """提交数据订单,返回订单号 (id_cmde)"""
    resp = requests.post(
        "https://portail-api.meteofrance.fr/clim/v1/ordre",
        json={
            "station": station_id,
            "step": "1mo",
            "date_deb": date_deb,
            "date_fin": date_fin,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "id_cmde" not in data:
        print(f"  返回数据: {data}")
    return data.get("id_cmde") or data.get("id_ordre")


def check_order_status(token: str, order_id: str) -> dict:
    """查询订单状态"""
    resp = requests.get(
        f"https://portail-api.meteofrance.fr/clim/v1/ordre/{order_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def download_order_csv(token: str, order_id: str, output_path: str) -> str:
    """下载已完成的订单 CSV"""
    resp = requests.get(
        f"https://portail-api.meteofrance.fr/clim/v1/ordre/{order_id}/csv",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return output_path


def download_with_api(token: str, station_id: str, output_path: str):
    """完整的 API 下载流程"""
    print(f"  站点: {station_id}")
    print(f"  时间: {START_DATE} → {END_DATE}")

    # 1. 提交订单
    print("  提交数据订单...")
    order_id = order_data(token, station_id, START_DATE, END_DATE)
    print(f"  订单号: {order_id}")

    # 2. 轮询等待完成
    print("  等待数据处理...", end="", flush=True)
    for _ in range(30):  # 最多等 5 分钟
        time.sleep(10)
        status = check_order_status(token, order_id)
        print(".", end="", flush=True)
        if status.get("statut") == "termine":
            print(" 完成!")
            break
    else:
        print("\n  超时。请稍后手动下载。")
        return None

    # 3. 下载 CSV
    print("  下载 CSV...")
    download_order_csv(token, order_id, output_path)
    print(f"  已保存: {output_path}")
    return output_path


# ============ 数据处理 ============


def process_weather_csv(csv_path: str, output_path: str) -> pd.DataFrame:
    """处理原始天气 CSV,提取所需列"""
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")

    # 标准化列名
    df.columns = [c.strip().upper() for c in df.columns]

    print(f"  原始列: {list(df.columns)}")
    print(f"  原始行数: {len(df)}")

    # 需要的列
    target_cols = {
        "DATE": "date",
        "RR": "RR",
        "NBJRR1": "NBJRR1",
        "NBJRR5": "NBJRR5",
        "NBJRR10": "NBJRR10",
        "NBJBROU": "NBJBROU",
    }

    # 只保留目标列
    available = {k: v for k, v in target_cols.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # RR 列: 原始数据以十分之一毫米为单位,需除以 10
    if "RR" in df.columns:
        df["RR"] = pd.to_numeric(df["RR"], errors="coerce") / 10.0

    # 数值转换
    for col in ["NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(output_path, index=False)
    print(f"  处理结果: {df.shape}, 列: {list(df.columns)}")
    return df


# ============ 主入口 ============


def main():
    parser = argparse.ArgumentParser(
        description="下载法国月度天气数据 (2006-12 至 2010-11)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Météo-France API Bearer Token (从 portail-api.meteofrance.fr 获取)",
    )
    parser.add_argument(
        "--station",
        type=str,
        default=STATION_ID,
        help=f"站点 ID (默认: {STATION_ID} Paris-Montsouris)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = args.output or os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    if args.api_key:
        # 方案 B: API 下载
        print("=" * 60)
        print("方案 B: API 自动下载")
        print("=" * 60)
        csv_path = download_with_api(args.api_key, args.station, out_path)
        if csv_path:
            process_weather_csv(csv_path, out_path)
    else:
        # 方案 A: 提示手动下载
        print(MANUAL_DOWNLOAD_GUIDE)
        print(f"\n下载完成后将 CSV 放入: {OUTPUT_DIR}")
        print(f"或指定路径: python {__file__} --output <path>")


if __name__ == "__main__":
    main()
