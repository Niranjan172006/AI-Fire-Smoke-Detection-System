def interest(p,r,t):
    si = (p*r*t)/100
    return si

print(interest(1000,5,2))

def fact(n):
    if n == 1:
        return 1
    return n * fact(n-1)

print(fact(5))