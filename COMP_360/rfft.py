def rfft(a):
    """
    Recursive algorith for the fast fourier transform (FFT)
    """

    n = len(a)
    # repeat until we have n time signals of length 1
    if (n == 1):
        return a

    # e^(2pi*i/n)
    w_n = np.exp(2*np.pi*1j/n)
    w = 1

    a0 = a[0::2] #select even indices
    a1 = a[1::2] #select odd indices
    
    y0 = rfft(a0)
    y1 = rfft(a1)

    y = np.zeros(n)
    t = int(n/2)
    for k in range (0, t):
        y[k] = y0[k] + w*y1[k]
        y[k + t] = y0[k] - w*y1[k]
        w = w * w_n

    return y