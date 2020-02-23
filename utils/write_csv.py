import pandas as pd
import numpy as np


def write_csv(id_, p1, p2, p3, p4, p5, p6, csv_name):
    id_ = pd.Series(id_, name='id')
    p1 = pd.Series(p1, name='p1')
    p2 = pd.Series(p2, name='p2')
    p3 = pd.Series(p3, name='p3')
    p4 = pd.Series(p4, name='p4')
    p5 = pd.Series(p5, name='p5')
    p6 = pd.Series(p6, name='p6')
    con = pd.concat([id_, p1, p2, p3, p4, p5, p6], axis=1)
    con.to_csv(csv_name, index=0)
