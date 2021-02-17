'''
Simulation of Dominion benchmark strategy.
'''
import numpy as np
import pandas as pd


class TreasureCard:
    
    def __init__(self, val):
        self.val = val
    
    def __str__(self):
        return "Treasure " + str(self.val)


class VictoryCard:
    
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return "Victory " + str(self.val)


class Player:
    
    def __init__(self):
        self.nachziehstapel = []
        self.hand = []
        self.ablagestapel = []
        
        for _n in range(7):
            self.ablagestapel.append(TreasureCard(1))
        for _n in range(3):
            self.ablagestapel.append(VictoryCard(1))

    def mische(self):
        nachzieh = self.nachziehstapel
        ablage = self.ablagestapel
        assert len(nachzieh) == 0, "Es gibt noch Karten, mischen unnötig"
        n = len(ablage)
        print('mische ' + str(n) + ' Karten')
        while len(ablage) > 0:
            rand = np.random.randint(0, high=len(ablage))
            nachzieh.append(ablage[rand])
            del ablage[rand]
        assert len(nachzieh) == n

    def zieheNach(self):
        print('ziehe nach')
        while len(self.hand) < 5:
            if len(self.nachziehstapel) == 0:
                self.mische()
            self.hand.append(self.nachziehstapel.pop())

    def getCash(self):
        ret = 0
        for karte in self.hand:
            if type(karte) is TreasureCard:
                ret += karte.val
        return ret

    def cleanUp(self):
        self.ablagestapel.extend(self.hand[:]) 
        self.hand.clear()

    def __str__(self):
        return "\n" + str(self.nachziehstapel) + "\n" \
            +str(self.hand) + "\n" \
            +str(self.ablagestapel) + "\n" \



def sim():
    '''
    Simulation
    '''

    provinzen = []
    for _n in range(5):
        provinzen.append(VictoryCard(6))
    
    player = Player()
    
    nRounds = 0
    while len(provinzen) > 0:
        nRounds += 1
        print("Round " + str(nRounds))
        player.zieheNach()
        cash = player.getCash()
        if cash >= 8:
            player.ablagestapel.append(provinzen.pop())
        elif cash >= 6:
            player.ablagestapel.append(TreasureCard(3))
        elif cash >= 3:
            player.ablagestapel.append(TreasureCard(2))
        else:
            print("zu wenig Kohle")
        player.cleanUp()
            
    print(str(nRounds) + " Runden")
    return nRounds


vals = []
for i in range(1000):
    n = sim()
    vals.append(n)

series = pd.Series(vals)
df = pd.DataFrame({'nRounds' : series})
print(df)
print(df.describe())
