""" Module containing functions to calculate scores """

def hitrate(a, d, n):
    return (a + d) / n


def pod_clr(b, d):
    return d / (b + d)


def pod_cld(a, c):
    return a / (a + c)


def far_clr(c, d):
    return c / (c + d)


def far_cld(a, b):
    return b / (a + b)


def pofd_clr(a, c):
    return c / (a + c)


def pofd_cld(b, d):
    return b / (b + d)


def heidke(a, b, c, d):
    num = 2 * ((a * d) - (b * c))
    denom = ((a + c) * (c + d) + (a + b) * (b + d))
    denom[denom == 0] = 1
    return num / denom


def kuiper(a, b, c, d):
    num = (a * d - b * c)
    denom = (a + c) * (b + d)
    denom[denom == 0] = 1
    kuipers = num / denom
    return kuipers


def bias(b, c, n):
    return (b - c) / n


def mean(x, y, n):
    return (x + y) / n