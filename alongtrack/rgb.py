"""
Module calculating RGB array from passive instrument satellite channels.

Code by Martin Stengel (DWD)

"""

import numpy as np


def calc_rgb_with016(data_vis006, data_vis008, data_nir016,
                     data_tir108, data_sunzen):
    """ Generate RGB from VIS006, VIS008, NIR016, IR108 and solar zenith. """

    truefact = np.where((data_sunzen > 0) & (data_sunzen < 88),
                        np.cos(np.radians(data_sunzen)), 1)

    ir = data_tir108
    ir = np.where(ir < 0, 0, ir)

    term = (ir-302.)
    term = np.where(term < 0, 0, term )
    term = term / 30. * 40.

    r = data_nir016/truefact / 100. * 0.5
    r = np.where(r < 0, 0, r)
    r = np.where(r > 1, 1, r)
    r = r*255

    g = data_vis008/truefact / 138. * 0.8
    g = np.where(g < 0, 0, g)
    g = np.where(g > 1, 1, g)
    g = g*255

    b = data_vis006/truefact / 122. * 0.5
    b = np.where(b < 0, 0, b)
    b = np.where(b > 1, 1, b)
    b = b*255

    r = r * 2.5*0.8
    g = g * 2.5*0.8
    b = b * 3.2*0.8

    c1 = 298.
    c2 = 10.

    r_weight = np.arctan((ir - c1) / c2 * np.pi) / np.pi + 0.5
    multi = np.where(g > b, g , b)
    r = r_weight * r + ((1 - r_weight) * multi)

    r = np.where(r > 255, 255, r)
    g = np.where(g > 255, 255, g)
    b = np.where(b > 255, 255, b)

    ir_min = 180.
    ir_max = 310.
    irb = np.where(ir > ir_max, ir_max, ir)
    irb = np.where(ir < ir_min, ir_min, ir)
    irb = ir_max - irb
    irb = (irb * 255.) / (ir_max - ir_min)
    irb =  np.where(irb > 255, 255, irb)
    irb =  np.where(irb < 0, 0, irb)

    r = np.where(data_sunzen >= 88, irb, r)
    g = np.where(data_sunzen >= 88, irb, g)
    b = np.where(data_sunzen >= 88, irb, b)

    weight = (data_sunzen-70)/18.
    r = np.where((data_sunzen > 70) & (data_sunzen < 88),
                 weight*irb + (1-weight)*r, r)
    g = np.where((data_sunzen > 70) & (data_sunzen < 88),
                 weight*irb + (1-weight)*g, g)
    b = np.where((data_sunzen > 70) & (data_sunzen < 88),
                 weight*irb + (1-weight)*b, b)

    rgb_tmp = np.dstack((r,g,b))

    rgb_tmp = rgb_tmp.astype(np.int)
    rgb_tmp = np.where(rgb_tmp > 255, 255, rgb_tmp)
    return rgb_tmp

