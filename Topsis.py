import numpy as np

#https://github.com/shivambehl/TopsisPy
#author: Shivam Behl

class Topsis():

    def __init__(self):
        pass

    def floater(self, a):  # .astype() can be used but is not reliable
        b = []
        for i in a:
            try:
                ix = []
                for j in i:
                    ix.append(float(j))
            except:
                ix = float(i)
                pass
            b.append(ix)
        b = np.array(b)
        return b


    def normalize(self, matrix, r, n, m):
        for j in range(m):
            sq = np.sqrt(sum(matrix[:, j]**2))
            for i in range(n):
                r[i, j] = matrix[i, j]/sq
        return r


    def weight_product(self, matrix, weight):
        r = matrix*weight
        return r


    def calc_ideal_best_worst(self, sign, matrix, n, m):
        ideal_worst = []
        ideal_best = []
        for i in range(m):
            if sign[i] == 1:
                ideal_worst.append(min(matrix[:, i]))
                ideal_best.append(max(matrix[:, i]))
            else:
                ideal_worst.append(max(matrix[:, i]))
                ideal_best.append(min(matrix[:, i]))
        return (ideal_worst, ideal_best)


    def euclidean_distance(self, matrix, ideal_worst, ideal_best, n, m):
        diw = (matrix - ideal_worst)**2
        dib = (matrix - ideal_best)**2
        dw = []
        db = []
        for i in range(n):
            dw.append(sum(diw[i, :])**0.5)
            db.append(sum(dib[i, :])**0.5)
        dw = np.array(dw)
        db = np.array(db)
        return (dw, db)


    def performance_score(self, distance_best, distance_worst, n, m):
        score = []
        score = distance_worst/(distance_best + distance_worst)
        return score


    def topsis(self, a, w, sign, custom_ideal_worst=None, custom_ideal_best=None):
        a = self.floater(a)
        n = len(a)
        m = len(a[0])
        # print('n:', n, '\nm:', m)
        r = np.empty((n, m), np.float64)
        r = self.normalize(a, r, n, m)
        t = self.weight_product(r, w)
        (ideal_worst, ideal_best) = self.calc_ideal_best_worst(sign, t, n, m)
        if custom_ideal_worst != None:
            ideal_worst = custom_ideal_worst
        if custom_ideal_best != None:
            ideal_best = custom_ideal_best
        (distance_worst, distance_best) = self.euclidean_distance(
            t, ideal_worst, ideal_best, n, m)
        score = self.performance_score(distance_best, distance_worst, n, m)
        return (np.argmax(score), score)
        # returns a tupple with index of best data point as first element and score array(numpy) as the other
