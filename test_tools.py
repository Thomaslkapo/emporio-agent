"""Checks the rules that must not depend on the LLM. Run: python test_tools.py"""
from tools import CUSTOMERS, ORDERS, PRODUCTS, get_order, search_products


def by_id(product_id):
    name = next(p["name"] for p in PRODUCTS if p["product_id"] == product_id)
    return search_products.invoke({"query": name})["produtos"][0]


# Accents don't matter when searching
assert search_products.invoke({"query": "violao"})["total_encontrado"] > 0

# Price limit is respected
cheap = search_products.invoke({"query": "violões", "max_price": 1000})["produtos"]
assert cheap and all(p["preco_atual"] <= 1000 for p in cheap)

# Exact price lookup, with the 5% PIX discount
gd20 = search_products.invoke({"query": "Takamine GD20"})["produtos"]
assert len(gd20) == 1 and gd20[0]["preco_atual"] == 2199 and gd20[0]["preco_pix"] == 2089.05

# Active promotion applies and doesn't stack with PIX
promo = by_id("121")
assert promo["desconto_percentual"] == 20 and "preco_pix" not in promo
assert promo["preco_atual"] == round(promo["preco_tabela"] * 0.8, 2)

# Expired promotion is never shown
assert "promocao" not in by_id("102")

# Discontinued products are not available
discontinued = next(p for p in PRODUCTS if p["status"] == "discontinued")
assert by_id(discontinued["product_id"])["disponivel"] is False

# Order needs the right contact: phone (any format) or email
order = ORDERS["8"]
customer = CUSTOMERS[order["customer_id"]]
assert get_order.invoke({"order_id": 8, "phone_or_email": customer["phone"]})["status"] == "shipped"
only_digits = "".join(c for c in customer["phone"] if c.isdigit())
assert "status" in get_order.invoke({"order_id": 8, "phone_or_email": only_digits})
assert "status" in get_order.invoke({"order_id": 8, "phone_or_email": customer["email"].upper()})

# Wrong, empty and someone else's contact get the same denial as a missing order
denied = get_order.invoke({"order_id": 999, "phone_or_email": customer["phone"]})
assert "erro" in denied
for wrong in ["", "   ", "(67) 00000-0000", CUSTOMERS["1"]["email"]]:
    assert get_order.invoke({"order_id": 8, "phone_or_email": wrong}) == denied

print("ok")
