def gmsl_model(paramaters, temperatures, deltat):
    alpha, Teq, S0 = paramaters[0], paramaters[1], paramaters[2]
    S = [0]*(len(temperatures)+1)
    S[0] = S0
    for i in range(1,len(temperatures)+1):
        S[i] = S[i-1] + deltat * alpha * (temperatures[i-1] - Teq)
    return S[1:]