from dataclasses import dataclass
from datetime import datetime

import gurobipy as gb
import openpyxl
import pandas as pd
import pulp as pl
from pulp import lpSum, LpBinary as Bin


def get_positive_expr_values_int(dct, tolerance=0.001):
    ret = {}
    for k, expr in dct.items():
        val = pl.value(expr)
        assert abs(val - round(val)) <= tolerance
        if val > tolerance:
            ret[k] = round(val)
    return ret


class Instance:

    def __init__(self):
        workbook = openpyxl.load_workbook(filename='./Karsana.xlsx', data_only=True)  # read values, not formulas
        dfs = pd.read_excel(workbook, engine='openpyxl', skiprows=1,
                            sheet_name=['Produkte', 'Packstuecke'])

        df = dfs['Produkte']
        self.products = df['Produkt'].to_list()
        df.set_index('Produkt', inplace=True)
        self.gewicht_p = df['kg pro Karton'].to_dict()
        self.laenge_p = df['Länge'].to_dict()
        self.breite_p = df['Breite'].to_dict()
        self.hoehe_p = df['Höhe'].to_dict()
        self.anzahl_p = df['Menge in Kartons'].to_dict()

        df = dfs['Packstuecke']
        self.packstuecke = df['Typ ID'].to_list()
        df.set_index('Typ ID', inplace=True)
        self.gewicht_k = df['Max kg'].to_dict()
        self.laenge_k = df['Länge'].to_dict()
        self.breite_k = df['Breite'].to_dict()
        self.hoehe_k = df['Höhe'].to_dict()
        self.preis_k = df['Versandkosten pro Packstück'].to_dict()


class Solution:
    def __init__(self, instance: Instance, mdl: 'PackModell', y_kj_vals, x_pikj_vals):
        rows = []
        for (p, i, k, j), x in x_pikj_vals.items():
            assert (k, j) in y_kj_vals
            gewicht = instance.gewicht_p[p]
            row = (k, j, p, i, gewicht)
            rows.append(row)

        df = pd.DataFrame(rows, columns=['Packstueck', 'j', 'Produkt', 'Anzahl', 'Gewicht'])
        self.packing = df.groupby(['Packstueck', 'j', 'Produkt']).sum()
        self.packing.sort_values(['Packstueck', 'j', 'Produkt'], inplace=True)
        print('Solution ready')

    def write_csv(self):
        self.packing.to_csv('./out.csv')


@dataclass()
class PackModell:
    solve_secs: float

    def __init__(self, instance: Instance):
        self._i = instance

        n_products = sum(self._i.anzahl_p.values())
        n_k = {k: n_products for k in self._i.packstuecke}

        print("")
        print("Build model")
        self._mdl = pl.LpProblem("PackingModel", pl.LpMinimize)

        print("Variables")
        idx = ((k, j) for k in self._i.packstuecke for j in range(n_k[k]))
        self.y_kj = gb.tupledict(pl.LpVariable.dicts(name='y_kj', indices=idx, cat=Bin))

        idx = ((p, i, k, j) for p in self._i.products for i in range(self._i.anzahl_p[p])
               for k in self._i.packstuecke for j in range(n_k[k]))
        self.x_pikj = gb.tupledict(pl.LpVariable.dicts(name='x_pikj', indices=idx, cat=Bin))

        print('Objective')
        self.obj_expr = lpSum(self._i.preis_k[k] * y for (k, j), y in self.y_kj.items())
        self._mdl += self.obj_expr

        print("Constraints")
        print("\tGewicht")
        for (k, j), y in self.y_kj.items():
            lhs_expr = lpSum(self._i.gewicht_p[p] * self.x_pikj[p, i, k, j]
                             for p in self._i.products
                             for i in range(self._i.anzahl_p[p]))
            self._mdl += lhs_expr <= self._i.gewicht_k[k] * y, f"c_gewicht_{k},{j}"

        print("\tVerpacke alles")
        for p in self._i.products:
            for i in range(self._i.anzahl_p[p]):
                lhs_expr = lpSum(self.x_pikj[p, i, k, j] for k in self._i.packstuecke for j in range(n_k[k]))
                self._mdl += lhs_expr == 1, f"c_verpacke_{p},{i}"

        print("Model built")

    def _build_solution(self) -> Solution:
        y_kj_vals = get_positive_expr_values_int(self.y_kj)
        x_pikj_vals = get_positive_expr_values_int(self.x_pikj)
        return Solution(self._i, self, y_kj_vals, x_pikj_vals)

    def solve(self, timeout_sec=60 * 5, mip_gap=0.001) -> Solution:
        print('Solve model')
        print(f"\tTimeout set to {timeout_sec} sec")
        start = datetime.now()
        self._mdl.writeLP('./packing_model.lp')
        solver = pl.getSolver('PULP_CBC_CMD', timeLimit=timeout_sec, msg=True, gapRel=mip_gap, presolve=True)
        self._mdl.solve(solver=solver)
        self.solve_secs = round((datetime.now() - start).seconds, 2)
        print(f"Model solved in {self.solve_secs:.0f} secs")
        print(f"Objective value = {pl.value(self._mdl.objective):.0f}")

        return self._build_solution()


if __name__ == '__main__':
    instance = Instance()
    model = PackModell(instance)
    solution = model.solve()
    solution.write_csv()
    print('Fertig')
