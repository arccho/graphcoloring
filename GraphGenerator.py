import random

def id_generator(number_id, size=12):
    alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    liste_id = []
    for x in range(number_id):
        liste_id.append(''.join(random.choice(alphanum) for _ in range(size)))
    return liste_id

print id_generator(10)