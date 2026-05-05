import numpy as np


def gga_functional(name, rho, grad_rho, alpha):
    # evaluate the Perdwe, Burke and Ernzerhof Generalized Gradient Approximation functional

    # deal with zeros for divide
    epsilon = 1e-30
    rho = rho + epsilon

    if name in ["PBE", "PBE0"]:
        # parameters
        MU, KAPPA = 0.2195149727645171, 0.804
        GAMMA, BETA = 0.031090690869655, 0.066724550603149
        PW = np.array([0.0310907, 0.2137000, 7.5957000, 3.5876000, 1.6382000, 0.4929400])

        def interpolate_LSD_energy(r, PW):
            # correlation energy interpolation from PW91

            q = [0.0, 0.0, 0.0, 0.0]
            q[0] = -2 * PW[0] * (1 + PW[1] * r * r)
            q[1] = 2 * PW[0] * r * (PW[2] + r * (PW[3] + r * (PW[4] + r * PW[5])))
            q[2] = np.log(1.0 + 1.0 / q[1])
            gg = q[0] * q[2]

            q[3] = PW[0] * (PW[2] / r + 2.0 * PW[3] + r * (3.0 * PW[4] + r * 4.0 * PW[5]))
            gg_rs = -2.0 * PW[0] * PW[1] * q[2] - q[0] * q[3] / (q[1] * (1.0 + q[1]))

            return gg, gg_rs

        def gga_pbe_exchange(rho, grad_rho):
            # local density approximation exchange energy
            ex_lda = -0.75 * (3.0 * rho / np.pi) ** (1.0 / 3.0)

            kf = (3.0 * np.pi * np.pi * rho) ** (1.0 / 3.0)

            # dimensionless gradient
            s = grad_rho / (2.0 * kf * rho)
            p = 1.0 + (MU / KAPPA) * s * s

            # enhancement factor
            f = 1.0 + KAPPA - KAPPA / p

            # apply enhancement
            ex = ex_lda * f

            # derivative of enhancement factor with respect to s
            fs = 2.0 * MU / (p * p) * s

            # exchange potential - rho and sigma components
            v_rho_x = (4.0 / 3.0) * (ex - ex_lda * fs * s)
            v_sigma_x = ex_lda * MU / (p * p * kf * kf * rho * 4 + epsilon)

            return ex, v_rho_x, v_sigma_x, kf

        def gga_pbe_correlation(rho, grad_rho, kf):
            # Seitz radius
            rs = (0.75 / (np.pi * rho)) ** (1.0 / 3.0)
            rrs = np.sqrt(rs)

            # Thomas-Fermi wavenumber
            ks = np.sqrt(4.0 * kf / np.pi)

            # dimensionless gradient
            t = grad_rho / (2.0 * ks * rho)

            ec_lda, ec_lda_rs = interpolate_LSD_energy(rrs, PW)
            b = -ec_lda / GAMMA
            b = BETA / ((np.exp(b) - 1.0) * GAMMA + epsilon)

            t_pow = [t * t, pow(t, 4), pow(t, 6)]

            # intermediates
            q = [0.0, 0.0, 0.0, 0.0]
            q[0] = 1.0 + b * t_pow[0]
            q[1] = q[0] + b * b * t_pow[1]

            # correlation rho potential
            b_ec = b / GAMMA + b * b / BETA

            q[2] = q[1] * q[1] + BETA * q[0] * q[1] * t_pow[0] / GAMMA
            q[3] = 1.0 + 2.0 * b * t_pow[0]

            hb = -BETA * b * t_pow[2] * (2.0 + b * t_pow[0]) / q[2]
            hr = hb * b_ec * ec_lda_rs
            ht = 2.0 * BETA * q[3] * t / q[2]

            h = GAMMA * np.log(1.0 + BETA * q[0] * t_pow[0] / (q[1] * GAMMA))
            ec = h + ec_lda
            v_rho_c = (
                ec - (1.0 / 3.0) * rs * ec_lda_rs - (1.0 / 3.0) * rs * hr - (7.0 / 6.0) * ht * t
            )

            return ec, v_rho_c

        def gga_pbe_sigma_potential(rho, sigma, epsilon):
            # Seitz radius
            rs = (3.0 / (4.0 * np.pi)) ** (1 / 3) / rho ** (1.0 / 3.0)
            rrs = np.sqrt(rs)

            # get interpolated LDA correlation
            ec_lda, _ec_lda_rs = interpolate_LSD_energy(rrs, PW)

            b = -ec_lda / GAMMA
            b = 1.0 / ((np.exp(b) - 1.0) + epsilon)

            p = 1.0 / (rho ** (1.0 / 3.0) * rho**2)
            tr = 2 ** (1.0 / 3.0) * p / (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

            # intermediates
            q = [0.0] * 8
            q[0] = BETA * b * sigma**2 / GAMMA
            q[1] = sigma * tr / 32.0 + q[0] * tr * tr / 1024.0
            q[2] = BETA * b * q[1] / GAMMA + 1.0
            q[3] = BETA * q[1] / (q[2] * GAMMA) + 1.0
            q[4] = tr / 32.0 + BETA * b * sigma * tr * tr / (512.0 * GAMMA)

            q[5] = BETA * BETA * q[1] / (GAMMA * GAMMA)
            q[6] = b * q[4] / (q[2] * q[2])

            q[7] = BETA * q[4] / (q[2] * GAMMA) - q[5] * q[6]

            return rho * GAMMA * q[7] / q[3]

        # get exchange energy, rho-potential and sigma-potential
        ex, v_rho_x, v_sigma_x, kf = gga_pbe_exchange(rho, grad_rho)

        # get correlation energy and rho-potential
        ec, v_rho_c = gga_pbe_correlation(rho, grad_rho, kf)

        # get correlation sigma-potential
        v_sigma_c = gga_pbe_sigma_potential(rho, grad_rho**2, epsilon)

        # total energy
        exc = (alpha * ex) + ec

        # total rho and sigma-potentials
        v_rho_xc = (v_rho_x * alpha) + v_rho_c
        v_sigma_xc = (v_sigma_x * alpha) + v_sigma_c

    if name == "B3LYP":
        sigma = grad_rho * grad_rho + epsilon

        def local_density_approximation(rho):
            # LDA exchange

            ex = -0.75 * pow(3 / np.pi, 1 / 3) * pow(rho, 1 / 3)
            vx = 4 / 3 * ex

            return ex, vx

        def becke_88_exchange(rho, sigma, ex_lda, vx_lda):
            # Becke 88 exchange functional

            beta = 0.0042
            gamma = 6

            # helper intermediates
            q = []
            q.append(np.sqrt(sigma) * pow(2, 1 / 3) / (rho * pow(rho, 1 / 3)))
            q.append(beta * gamma * q[0] * np.log(q[0] + np.sqrt(1.0 + q[0] * q[0])))
            q.append(q[0] * q[0])
            q.append(q[1] + 1)

            # exchange energy
            ex_b = -pow(rho, 1 / 3) * beta * q[0] * q[0] / (q[3] * pow(2, 1 / 3))

            # rho derivatives
            vx_rho_b = 4 / 3 * ex_b
            vx_rho_b += (
                4
                / 3
                * beta
                * q[2]
                / pow(2, 1 / 3)
                * pow(rho, 1 / 3)
                * (2 / q[3] - (q[1] + gamma * beta * q[2] / np.sqrt(1 + q[2])) / pow(q[3], 2))
            )

            # sigma derivatives
            vx = (
                2 / q[3]
                - q[1] / pow(q[3], 2)
                - gamma * beta * q[2] / np.sqrt(q[2] + 1) / pow(q[3], 2)
            )
            vx_sigma_b = -0.5 * vx * beta * q[0] / np.sqrt(sigma)

            return ex_lda + ex_b, vx_lda + vx_rho_b, vx_sigma_b

        def vwn_3_correlation(rho):
            # The Vosko, Wilk and Nusair 3 functional

            # VWN3 parameterisation
            A = 0.0310907
            x0 = -0.409286
            b = 13.0720
            c = 42.7198

            # helper intermediates
            q = []
            q.append(np.sqrt(pow(3.0 / (4.0 * np.pi * rho), 1.0 / 3.0)))
            q.append(np.sqrt(4 * c - b * b))

            def X(y):
                return y * y + b * y + c

            # correlation energy
            ec_vwn = A * (
                np.log(pow(q[0], 2.0) / X(q[0]))
                + 2.0 * b * np.arctan(q[1] / (2.0 * q[0] + b)) * pow(q[1], -1.0)
                - b
                * x0
                * (
                    np.log(pow(q[0] - x0, 2.0) / X(q[0]))
                    + 2.0 * (b + 2.0 * x0) * np.arctan(q[1] / (2.0 * q[0] + b)) * pow(q[1], -1.0)
                )
                * pow(X(x0), -1.0)
            )

            vc_rho_vwn = ec_vwn - (A / 3) * (c * (q[0] - x0) - b * q[0] * x0) / (
                (q[0] - x0) * (q[0] * q[0] + b * q[0] + c)
            )

            return ec_vwn, vc_rho_vwn

        def lyp_correlation(rho, sigma):
            # The Lee, Yang and Parr correlation functional

            a = 0.04918
            b = 0.132
            c = 0.2533
            d = 0.349

            # helper intermediates
            q = []
            q.append((d / (d * pow(rho, -1 / 3) + 1) + c) / pow(rho, 1 / 3))
            q.append(0.3 * pow(3, 2 / 3) * pow(np.pi * np.pi, 2 / 3))

            ec_lyp = (
                a
                * (
                    b
                    * np.exp(-c * pow(rho, -1 / 3))
                    * ((3 + 7 * q[0]) * sigma / 72 / (pow(rho, 2 / 3) * rho * rho) - q[1])
                    - 1
                )
                / (d * pow(rho, -1 / 3) + 1)
            )

            q.append(d * pow(rho, -1 / 3) + 0.1e1)
            q.append(b * np.exp(-c * pow(rho, -1 / 3)) / q[2])
            q.append(sigma / (pow(rho, 2 / 3) * rho * rho) * (1 / 24 + 7 / 72 * q[0]) - q[1])
            q.append(-d / pow(q[2], 2) + c * q[3] * q[4] + d * q[3] * q[4] / q[2])
            q.append(1 / 9 + 7 / 24 * q[0] - 7 / 216 * d * d / pow(q[2], 2) / pow(rho, 2 / 3))

            vc_rho_lyp = a * q[5] / 3 / pow(rho, 1 / 3) - a * q[3] * q[6] * sigma / pow(
                rho, 2 / 3
            ) / pow(rho, 2)
            vc_rho_lyp += a * (q[3] * q[4] - 1 / q[2])

            vc_sigma_lyp = a * q[3] * (1 + 7 / 3 * q[0]) / (pow(rho, 2 / 3) * rho) / 24

            return ec_lyp, vc_rho_lyp, vc_sigma_lyp

        # component energies and potentials
        ex_lda, vrho_lda = local_density_approximation(rho)
        ex_b88, vrho_b88, vsigma_b88 = becke_88_exchange(rho, sigma, ex_lda, vrho_lda)
        ec_vwn, vrho_vwn = vwn_3_correlation(rho)
        ec_lyp, vrho_lyp, vsigma_lyp = lyp_correlation(rho, sigma)

        # B3LYP definition - using VWN3 as Gaussian - .20HF + ,08LDA + .72B88 + ,19VWN + ,81LYP
        exc = 0.08 * ex_lda + 0.72 * ex_b88 + 0.81 * ec_lyp + 0.19 * ec_vwn
        v_rho_xc = 0.08 * vrho_lda + 0.72 * vrho_b88 + 0.81 * vrho_lyp + 0.19 * vrho_vwn
        v_sigma_xc = 0.72 * vsigma_b88 + 0.81 * vsigma_lyp

    # numerical clean-up
    exc[abs(exc) < 1.0e-10] = 0
    v_rho_xc[abs(v_rho_xc) < 1.0e-10] = 0
    v_sigma_xc[abs(v_sigma_xc) > 1.0e5] = 0

    return exc, (v_rho_xc, v_sigma_xc)
