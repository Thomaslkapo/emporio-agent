"""Tools the agent can call. Everything here reads the CSVs in data/."""
import csv
import re
import unicodedata
from pathlib import Path

from langchain_core.tools import tool

DATA = Path(__file__).parent / "data"
MAX_RESULTS = 15


def _read(name):
    with open(DATA / f"{name}.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


PRODUCTS = _read("products")
CATEGORIES = {c["category_id"]: c["name"] for c in _read("categories")}
ORDERS = {o["order_id"]: o for o in _read("orders")}
ORDER_ITEMS = _read("order_items")
CUSTOMERS = {c["customer_id"]: c for c in _read("customers")}
# Expired promotions are dropped here, so the agent can never offer them.
PROMOTIONS = {p["product_id"]: p for p in _read("promotions") if p["is_active"] == "1"}


def _normalize(text):
    """Lowercase and remove accents, so 'violão' matches 'violao'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _digits(text):
    return re.sub(r"\D", "", text)


def _describe(product):
    """Turn a CSV row into what the agent sees. Prices are computed here, not by the LLM."""
    price = float(product["price_brl"])
    stock = int(product["stock_quantity"])
    result = {
        "nome": product["name"],
        "categoria": CATEGORIES[product["category_id"]],
        "descricao": product["description"],
        "especificacoes": product["specs"],
        "status": product["status"],
        "estoque": stock,
        "disponivel": product["status"] == "active" and stock > 0,
        "preco_tabela": price,
    }
    promo = PROMOTIONS.get(product["product_id"])
    if promo:
        percent = float(promo["discount_percent"])
        result["promocao"] = promo["description"]
        result["desconto_percentual"] = percent
        result["preco_atual"] = round(price * (1 - percent / 100), 2)
    else:
        result["preco_atual"] = price
        # Store policy: the PIX discount does not stack with promotions.
        result["preco_pix"] = round(price * 0.95, 2)
    return result


@tool
def search_products(query: str = "", max_price: float | None = None) -> dict:
    """Busca instrumentos no catálogo. Use SEMPRE antes de falar de preço, estoque ou promoção.

    Args:
        query: palavras do produto, marca ou categoria (ex.: "Takamine GD20", "violões").
            Vazio lista todo o catálogo.
        max_price: preço atual máximo em reais, se o cliente informar um limite.
    """
    words = _normalize(query).split()
    found = []
    for product in PRODUCTS:
        text = _normalize(
            f"{product['name']} {product['description']} {CATEGORIES[product['category_id']]}"
        )
        if all(word in text for word in words):
            described = _describe(product)
            if max_price is None or described["preco_atual"] <= max_price:
                found.append(described)
    found.sort(key=lambda p: (not p["disponivel"], p["preco_atual"]))
    return {"total_encontrado": len(found), "produtos": found[:MAX_RESULTS]}


@tool
def get_order(order_id: int, phone_or_email: str) -> dict:
    """Consulta um pedido. Exige o número do pedido E o telefone ou e-mail do cadastro do cliente.

    Args:
        order_id: número do pedido.
        phone_or_email: telefone ou e-mail informado pelo cliente nesta conversa.
    """
    # Same answer for "no such order" and "wrong contact", so nobody can probe which orders exist.
    denied = {"erro": "Pedido não encontrado ou dados de contato não conferem."}
    order = ORDERS.get(str(order_id))
    if not order:
        return denied
    customer = CUSTOMERS[order["customer_id"]]
    given = phone_or_email.strip().lower()
    phone = _digits(given)
    if given != customer["email"].lower() and (not phone or phone != _digits(customer["phone"])):
        return denied

    products = {p["product_id"]: p["name"] for p in PRODUCTS}
    return {
        "cliente": customer["name"],
        "data_pedido": order["order_date"],
        "status": order["status"],
        "total": float(order["total_brl"]),
        "pagamento": order["payment_method"],
        "rastreio": order["tracking_code"] or None,
        "previsao_entrega": order["estimated_delivery"] or None,
        "observacoes": order["notes"] or None,
        "itens": [
            {"produto": products[i["product_id"]], "quantidade": int(i["quantity"])}
            for i in ORDER_ITEMS
            if i["order_id"] == order["order_id"]
        ],
    }


TOOLS = [search_products, get_order]
