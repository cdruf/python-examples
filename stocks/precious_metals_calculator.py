from dataclasses import dataclass

GRAMM_PER_OUNCE = 31.1034768
online_price_usd_per_ounce = 1684.29
eur_per_usd = 1.082


def get_price_in_eur_oz(oz):
    return oz * online_price_usd_per_ounce / eur_per_usd


def get_price_in_eur_gr(gramm):
    return gramm / GRAMM_PER_OUNCE * online_price_usd_per_ounce / eur_per_usd


def gold_summary(options):
    for o in options:
        preis = get_price_in_eur_gr(o.gramm)
        aufschlag = (o.price - preis) / preis * 100
        out = (f'Theoretischer Preis: {get_price_in_eur_gr(o.gramm) :.2f} EUR pro {o.gramm} Gramm; '
               f'Ladenpreis: {o.price:.2f} EUR, Aufschlag {aufschlag:.2f} %')
        print(out)


@dataclass
class PurchaseOption:
    gramm: float
    price: float


if __name__ == '__main__':
    options = [PurchaseOption(5, 285.5),
               PurchaseOption(10, 540.5),
               PurchaseOption(20, 1174.0)]
    gold_summary(options)
