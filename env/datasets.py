"""
Deterministic messy datasets for each task.
All datasets are hardcoded so grader scores are fully reproducible.
"""

import copy
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# TASK EASY — Customer Records
# Issues: age stored as strings, 3 "N/A" ages, 1 duplicate row, 2 trailing spaces
# ---------------------------------------------------------------------------
_TASK_EASY_RAW: List[Dict[str, Any]] = [
    {"name": "Alice ",   "age": "28",  "email": "alice@email.com",   "city": "New York"},
    {"name": "Bob",      "age": "N/A", "email": "bob@email.com",     "city": "London"},
    {"name": "Charlie",  "age": "35",  "email": "charlie@email.com", "city": "Paris"},
    {"name": "Diana",    "age": "N/A", "email": "diana@email.com",   "city": "Tokyo"},
    {"name": "Eve ",     "age": "22",  "email": "eve@email.com",     "city": "Sydney"},
    {"name": "Alice ",   "age": "28",  "email": "alice@email.com",   "city": "New York"},  # dup
    {"name": "Frank",    "age": "31",  "email": "frank@email.com",   "city": "Berlin"},
    {"name": "Grace",    "age": "N/A", "email": "grace@email.com",   "city": "Rome"},
    {"name": "Henry",    "age": "45",  "email": "henry@email.com",   "city": "Madrid"},
    {"name": "Iris",     "age": "29",  "email": "iris@email.com",    "city": "Amsterdam"},
]

# ---------------------------------------------------------------------------
# TASK MEDIUM — Product Inventory
# Issues: mixed-case categories, 3 negative prices, 5 null stocks, 3 null product codes
# ---------------------------------------------------------------------------
_TASK_MEDIUM_RAW: List[Dict[str, Any]] = [
    {"product_code": "P001", "name": "Laptop",        "category": "Electronics", "price":  999.99, "stock": 5},
    {"product_code": "P002", "name": "Mouse",         "category": "electronics", "price":   29.99, "stock": None},
    {"product_code": "P003", "name": "Keyboard",      "category": "ELECTRONICS", "price":   79.99, "stock": 12},
    {"product_code": None,   "name": "Monitor",       "category": "Electronics", "price":  -50.00, "stock": 3},
    {"product_code": "P005", "name": "Headphones",    "category": "electronics", "price":  149.99, "stock": None},
    {"product_code": "P006", "name": "Webcam",        "category": "ELECTRONICS", "price":   59.99, "stock": 8},
    {"product_code": "P007", "name": "Desk Chair",    "category": "Furniture",   "price":  299.99, "stock": 2},
    {"product_code": "P008", "name": "Standing Desk", "category": "furniture",   "price":  -10.00, "stock": None},
    {"product_code": None,   "name": "Lamp",          "category": "FURNITURE",   "price":   45.99, "stock": 15},
    {"product_code": "P010", "name": "Notebook",      "category": "Stationery",  "price":    4.99, "stock": 100},
    {"product_code": "P011", "name": "Pen Set",       "category": "stationery",  "price":   12.99, "stock": None},
    {"product_code": "P012", "name": "Stapler",       "category": "STATIONERY",  "price":    8.99, "stock": 25},
    {"product_code": None,   "name": "Tape",          "category": "stationery",  "price":    3.49, "stock": None},
    {"product_code": "P014", "name": "Scissors",      "category": "Stationery",  "price":   -2.00, "stock": 30},
    {"product_code": "P015", "name": "Ruler",         "category": "stationery",  "price":    1.99, "stock": None},
]

# ---------------------------------------------------------------------------
# TASK HARD — Orders
# Issues: mixed date formats, negative quantities, invalid emails,
#          currency symbols in price, discounts > 100, 1 duplicate order_id
# ---------------------------------------------------------------------------
_TASK_HARD_RAW: List[Dict[str, Any]] = [
    {"order_id": "O001", "date": "2024-01-15",  "customer_email": "alice@example.com",   "product": "Laptop",      "quantity":  1,  "price": "$999.99", "discount_pct": 10},
    {"order_id": "O002", "date": "02/14/2024",  "customer_email": "bob@example.com",     "product": "Mouse",       "quantity": -1,  "price": "€29.99",  "discount_pct": 5},
    {"order_id": "O003", "date": "Mar 15 2024", "customer_email": "charlie.example.com", "product": "Keyboard",    "quantity":  2,  "price": "$79.99",  "discount_pct": 150},
    {"order_id": "O004", "date": "2024-04-20",  "customer_email": "diana@example.com",   "product": "Monitor",     "quantity":  1,  "price": "£499.99", "discount_pct": 20},
    {"order_id": "O005", "date": "05/05/2024",  "customer_email": "eve @example.com",    "product": "Headphones",  "quantity":  3,  "price": "$149.99", "discount_pct": 0},
    {"order_id": "O006", "date": "Jun 6 2024",  "customer_email": "frank@example.com",   "product": "Webcam",      "quantity": -2,  "price": "€59.99",  "discount_pct": 110},
    {"order_id": "O007", "date": "2024-07-07",  "customer_email": "grace@example.com",   "product": "Chair",       "quantity":  1,  "price": "$299.99", "discount_pct": 15},
    {"order_id": "O008", "date": "08/08/2024",  "customer_email": "henry@example.com",   "product": "Desk",        "quantity":  1,  "price": "£799.99", "discount_pct": 25},
    {"order_id": "O009", "date": "Sep 9 2024",  "customer_email": "iris.example.com",    "product": "Lamp",        "quantity":  4,  "price": "$45.99",  "discount_pct": 5},
    {"order_id": "O001", "date": "2024-01-15",  "customer_email": "alice@example.com",   "product": "Laptop",      "quantity":  1,  "price": "$999.99", "discount_pct": 10},  # dup
    {"order_id": "O010", "date": "10/10/2024",  "customer_email": "jack@example.com",    "product": "Notebook",    "quantity": 10,  "price": "€4.99",   "discount_pct": 0},
    {"order_id": "O011", "date": "Nov 11 2024", "customer_email": "kate@example.com",    "product": "Pen Set",     "quantity":  5,  "price": "$12.99",  "discount_pct": 200},
    {"order_id": "O012", "date": "2024-12-12",  "customer_email": "leo@example.com",     "product": "Stapler",     "quantity":  2,  "price": "£8.99",   "discount_pct": 10},
    {"order_id": "O013", "date": "01/13/2024",  "customer_email": "mia example.com",     "product": "Tape",        "quantity": -3,  "price": "$3.49",   "discount_pct": 5},
    {"order_id": "O014", "date": "Feb 14 2024", "customer_email": "noah@example.com",    "product": "Scissors",    "quantity":  1,  "price": "€5.99",   "discount_pct": 300},
    {"order_id": "O015", "date": "2024-03-15",  "customer_email": "olivia@example.com",  "product": "Ruler",       "quantity":  5,  "price": "$1.99",   "discount_pct": 0},
    {"order_id": "O016", "date": "04/16/2024",  "customer_email": "peter@example.com",   "product": "Binder",      "quantity": -1,  "price": "£6.99",   "discount_pct": 10},
    {"order_id": "O017", "date": "May 17 2024", "customer_email": "quinn example.com",   "product": "Folder",      "quantity":  3,  "price": "$2.99",   "discount_pct": 5},
    {"order_id": "O018", "date": "2024-06-18",  "customer_email": "rachel@example.com",  "product": "Highlighter", "quantity":  8,  "price": "€1.99",   "discount_pct": 120},
    {"order_id": "O019", "date": "07/19/2024",  "customer_email": "sam@example.com",     "product": "Eraser",      "quantity": 10,  "price": "$0.99",   "discount_pct": 0},
]


def get_dataset(task_id: str) -> List[Dict[str, Any]]:
    """Return a fresh deep-copy of the named task's dataset."""
    datasets = {
        "task_easy":   _TASK_EASY_RAW,
        "task_medium": _TASK_MEDIUM_RAW,
        "task_hard":   _TASK_HARD_RAW,
    }
    if task_id not in datasets:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(datasets)}")
    return copy.deepcopy(datasets[task_id])
