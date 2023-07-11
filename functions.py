import math


class stable_data:
    def __init__(self, x, y, n):
        self.n = n
        self.lx = x
        self.ly = y
        self.min_x = 0
        self.min_y = 0
        self.max_x = 0
        self.max_y = 0
    
    def Minn(self):
        self.min_x = min(self.lx)
        self.min_y = min(self.ly)
        return self.min_x, self.min_y

    def Maxx(self):
        self.max_x = max(self.lx)
        self.max_y = max(self.ly)
        return self.max_x, self.max_y

    def razmah(self):
        self.rx = round(self.max_x - self.min_x, 2)
        self.ry = round(self.max_y - self.min_y, 2)
        return self.rx, self.ry
        
    def col_integ(self):
        self.col_integrals = round(1 + 3.2 * math.log10(self.n))
        return self.col_integrals

    def lens_integ(self):
        self.hx = round(self.rx / self.col_integrals, 2)
        self.hy = round(self.ry / self.col_integrals, 2)
        return self.hx, self.hy

    def gran_integ(self):
        self.a = [round(self.min_x - self.hx / 2, 2)]
        self.b = [round(self.min_y - self.hy / 2, 2)]
        for _ in range(8):
            self.a.append(round(self.a[-1] + self.hx, 2))
            self.b.append(round(self.b[-1] + self.hy, 2))
        return self.a, self.b
