import numpy as np


def wind_stress(u, v):
    """
    Tau=Rho*Cd*(speed)^2; Tx=Rho*Cd*Speed*u; Ty=Rho*Cd*Speed*v

    % INPUTS:
    % u = Zonal wind component [m/s], must be 2D
    % v = Meridional wind component [m/s], must be 2D

    % OUTPUT:
    % Tx = Zonal wind stress [N/m^2]
    % Ty = Meridional wind stress [N/m^2]

    :param u: Zonal wind component [m/s], must be 2D
    :param v: Meridional wind component [m/s], must be 2D
    :return:
    """
    # Defining Constant
    roh = 1.2   # kg/m^3, air density
    #
    Tx = np.full_like(u, np.nan)
    Ty = np.full_like(u, np.nan)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            U = np.sqrt(u[i, j] ** 2 + v[i, j] ** 2)
            if U <= 1:
                Cd = 0.00218
            elif 1 < U <= 3:
                Cd = (0.62+1.56/U)*0.001
            elif 3 < U < 10:
                Cd = 0.00114
            else:
                Cd = (0.49 + 0.065 * U) * 0.001
            Tx[i, j] = Cd * roh * U * u[i, j]
            Ty[i, j] = Cd * roh * U * v[i, j]
    return Tx, Ty


if __name__ == '__main__':
    u_ = np.random.random((10, 10))
    v_ = np.random.random((10, 10))
    Tx_, Ty_ = wind_stress(u_, v_)
    print(Tx_.shape)