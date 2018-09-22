from enum import IntEnum, auto
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from operator import add


class Position(IntEnum):
    '''蟻の位置を表す定数クラス'''
    change = 0
    stay = auto()
    home = auto()


class Select(IntEnum):
    '''蟻の選択を表す定数クラス'''
    change = 0
    stay = auto()


class Path():
    '''経路'''
    def __init__(self, evaporation_rate):
        # 蒸発率
        self.evaporation_rate = evaporation_rate
        
        # フェロモン2サイクル分
        self.lp = len(Position)
        self.p_pheromone = np.full((self.lp, self.lp), 0.001)
        self.pheromone = np.full((self.lp, self.lp), 0.001)
    
    def get_dests(self, position):
        '''目的地の候補と、1サイクル前のフェロモンを返す'''
        if position == Position.home:
            dests = [Position.change, Position.stay]
            pheromones = [self.p_pheromone[Position.home][Position.change],
                          self.p_pheromone[Position.home][Position.stay]]
        
        elif position == Position.change:
            dests = [Position.stay]
            pheromones = [self.p_pheromone[Position.change][Position.stay]]
        
        elif position == Position.stay:
            dests = [Position.change]
            pheromones = [self.p_pheromone[Position.stay][Position.change]]
        
        return dests, pheromones
    
    def append_pheromone(self, from_, to, pheromone):
        '''フェロモンを上塗りする'''
        self.pheromone[from_][to] += pheromone
    
    def update(self):
        '''フェロモンを蒸発、上塗りする'''
        self.p_pheromone *= self.evaporation_rate
        self.p_pheromone += self.pheromone
        self.pheromone = np.full((self.lp, self.lp), 0.001)


class Monty:
    '''モンティ'''
    def __init__(self):
        self.reset()
    
    def reset(self):
        '''状況を初期化する'''
        self.boxes = np.zeros(3)
        self.choices = np.zeros(3)
        self.key = np.zeros(3)
        self.key[np.random.randint(3)] = 1.
    
    def open(self):
        '''箱を開ける'''
        # 最初に選択する箱はランダム
        self.choices[np.random.randint(3)] = 1.
        
        # はずれの箱を一つ開ける
        boxes = self.key + self.choices
        choices = np.where(boxes == 0.)[0]
        self.boxes[np.random.choice(choices)] = 1.
        
        # stayとchangeのインデックスを保持
        self.stay = self.choices.argmax()
        self.change = (self.boxes + self.choices).argmin()
    
    def judge(self, select):
        '''蟻の選択を判定'''
        # 蟻の選択を反映
        if select == Select.change:
            self.choices = np.zeros(3)
            self.choices[self.change] = 1.
        
        elif select == Select.stay:
            self.choices = np.zeros(3)
            self.choices[self.stay] = 1.
        
        # 正解判定
        return self.choices.argmax() == self.key.argmax()


class Ant():
    '''蟻'''
    def __init__(self, path, action_rate):
        self.path = path
        self.action_rate = action_rate
        self.monty = Monty()
        self.pheromone = 1.
        self.position = Position.home
        self.hasFood = False
        self.memory = []
    
    def action(self):
        '''行動を起こす'''
        if np.random.random() < self.action_rate:
            self.walk()
        else:
            self.sleep()
    
    def sleep(self):
        '''何もしない'''
        pass
    
    def walk(self):
        '''歩く'''
        # 往路
        if not self.hasFood:
            if self.position == Position.home:
                # モンティに箱を開けさせる
                self.monty.reset()
                self.monty.open()
            
            # 目的地の候補とフェロモンを取得し、目的地（選択）を決定
            dests, pheromones = self.path.get_dests(self.position)
            sp = sum(pheromones)
            probability = [p / sp for p in pheromones]
            self.select = np.random.choice(dests, p=probability)
            
            # 選択した位置に移動
            self.memory.append(self.position)
            self.position = self.select
            
            # 餌を得られたか判定
            self.hasFood = self.monty.judge(self.select)
        
        # 復路
        else:
            if self.position == Position.home:
                # 餌を巣に保管し、フェロモン充電
                self.hasFood = False
                self.pheromone = 1.
            
            else:
                # フェロモン放出量を計算
                lm = len(self.memory)
                pheromone = self.pheromone / lm
                
                # フェロモンを放出しつつ、帰る
                dest = self.memory.pop()
                self.path.append_pheromone(dest, self.position, pheromone)
                self.pheromone -= pheromone
                self.position = dest


def visualize1(title, history):
    '''履歴をグラフ化する'''
    x = range(1, len(history[0]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(x, history[0], marker='.', label='p_home2change')
    plt.plot(x, history[1], marker='.', label='p_change2stay')
    plt.plot(x, history[2], marker='.', label='p_home2stay')
    plt.plot(x, history[3], marker='.', label='p_stay2change')
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=10)
    plt.grid()
    plt.xlabel('cycles')
    plt.ylabel('pheromone')
    plt.title(title)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.show()


def visualize2(title, history):
    '''履歴をグラフ化する'''
    x = range(1, len(history[0]) + 1)
    p_change = list(map(add, history[0], history[1]))
    p_stay = list(map(add, history[2], history[3]))
    plt.figure(figsize=(8, 4))
    plt.plot(x, p_change, marker='.', label='p_change')
    plt.plot(x, p_stay, marker='.', label='p_stay')
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=10)
    plt.grid()
    plt.xlabel('cycles')
    plt.ylabel('pheromone')
    plt.title(title)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.show()


class AntController():
    '''管理者'''
    def start(self, cycles, ants_num, action_rate, evaporation_rate):
        '''シミュレーションを開始する'''
        path = Path(evaporation_rate)
        ants = [Ant(path, action_rate) for _ in range(ants_num)]
        history = [[] for _ in range(4)]
        
        for _ in range(cycles):
            for ant in ants:
                ant.action()
            path.update()
            history[0].append(path.p_pheromone[Position.home][Position.change])
            history[1].append(path.p_pheromone[Position.change][Position.stay])
            history[2].append(path.p_pheromone[Position.home][Position.stay])
            history[3].append(path.p_pheromone[Position.stay][Position.change])
        
        title = (f'ants_num = {ants_num}, '
                 f'action_rate = {action_rate:.1f}, '
                 f'evaporation_rate = {evaporation_rate:.1f}')
        
        print('1/2')
        visualize1(title, history)
        print('2/2')
        visualize2(title, history)


if __name__ == '__main__':
    ac = AntController()
    print('wait a moment...')
    ac.start(cycles=200, ants_num=10000, action_rate=0.3, evaporation_rate=0.7)
