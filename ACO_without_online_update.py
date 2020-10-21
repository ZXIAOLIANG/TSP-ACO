import numpy as np
from random import randint
import random
import matplotlib.pyplot as plt
import copy
import csv
import math


def construct_distance_matrix(coordinate_list):
    Distance = [[0 for x in range(len(coordinate_list))] for y in range(len(coordinate_list))]
    for i in range(len(coordinate_list)):
        for j in range(len(coordinate_list)):
            if i < j:
                # we only need to calulate the upper triangle
                # calculate Euclidean distance
                Distance[i][j] = math.sqrt((float(coordinate_list[i][0]) - float(coordinate_list[j][0]))**2 
                                            + (float(coordinate_list[i][1]) - float(coordinate_list[j][1]))**2)
    return Distance

def evaporation(ph_map, roh):
    for i in range(len(ph_map)):
        for j in range(len(ph_map)):
            if i < j:
                ph_map[i][j] = (1.0-roh)*ph_map[i][j]


class Ant:
    def __init__(self, source, Distance, heuristic, ph_map, q0,alpha,beta, Q, tao_0,roh):
        self.source = source
        self.current_city = copy.deepcopy(source)
        self.tao_0 = tao_0
        self.roh = roh
        self.ph_map = ph_map
        self.Distance = Distance
        self.heuristic = heuristic
        self.q0 = q0
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.greeedy_list = []
        self.solution = []
        self.solution.append(self.current_city)
        self.unvisisted = []
        self.current_cost = 0
        self.unvisisted = [x for x in range(len(self.ph_map))]
        self.unvisisted.remove(self.current_city)
    
    def select_next_city(self):
        q = random.uniform(0,1)
        selected_city = 0
        if q < self.q0:
            # greedy
            max_val = 0
            for i in range(len(self.unvisisted)):
                val = self.ph_map[min(self.unvisisted[i], self.current_city)][max(self.unvisisted[i], self.current_city)]*\
                    pow(self.heuristic[min(self.unvisisted[i], self.current_city)][max(self.unvisisted[i], self.current_city)], self.beta)
                if val > max_val:
                    selected_city = self.unvisisted[i]
                    max_val = val
        else:
            # select based on probability
            weight_list = []
            for i in range(len(self.unvisisted)):
                weight =  pow(self.ph_map[min(self.unvisisted[i], self.current_city)][max(self.unvisisted[i], self.current_city)], self.alpha)*\
                        pow(self.heuristic[min(self.unvisisted[i], self.current_city)][max(self.unvisisted[i], self.current_city)],self.beta)
                weight_list.append(weight)
            Sum = sum(weight_list)
            try:
                weight_list[:] = [x /Sum for x in weight_list ]
            except ZeroDivisionError:
                print(weight_list)
                print("################")
                print(self.ph_map)

            selected_city = np.random.choice(self.unvisisted, p=weight_list)
        try:
            self.unvisisted.remove(selected_city)
        except ValueError:
            print(selected_city)
            print(self.unvisisted)
            print(q)
            print(self.q0)
        self.solution.append(selected_city)
        self.last_edge = self.Distance[min(selected_city, self.current_city)][max(selected_city, self.current_city)]
        self.current_cost += self.last_edge
        self.last_city = self.current_city
        self.current_city = selected_city

    def offline_update_ph(self):
        for i in range(len(self.solution)-1):
            self.ph_map[min(self.solution[i], self.solution[i+1])][max(self.solution[i], self.solution[i+1])] += self.Q/self.current_cost
    
    def online_update_ph(self):
        for i in range(len(self.solution)-1):
            self.ph_map[min(self.solution[i], self.solution[i+1])][max(self.solution[i], self.solution[i+1])] += self.Q/self.current_cost
        
    def go_back_to_source(self):
        self.solution.append(self.source)
        self.last_edge = self.Distance[min(self.source, self.current_city)][max(self.source, self.current_city)]
        self.current_cost += self.last_edge
        self.last_city = self.current_city
        self.current_city = self.source

def run_ACO(max_itr, ant_count, Distance, heuristic, city_list,
            alpha, beta, Q, roh, tao_0, q0):
    # print(roh)
    best_solution_per_itr = []
    ph_map = [[tao_0 for x in range(len(coordinate_list))] for y in range(len(coordinate_list))]
    for m in range(max_itr):
        random.shuffle(city_list)
        ant_list = []
        # initialize ants to different city
        for i in range(ant_count):
            ant = Ant(city_list[i%28],Distance, heuristic, ph_map, q0, alpha, beta, Q, tao_0,roh)
            ant_list.append(ant)

        for i in range(ant_count):
            for n in range(len(city_list)-1):
                ant_list[i].select_next_city()
            ant_list[i].go_back_to_source()

        
        # check for best solution per iteration
        best_ant_index = 0
        best_cost = float('inf')
        for i in range(ant_count):
            if ant_list[i].current_cost < best_cost:
                best_cost = ant_list[i].current_cost
                best_ant_index = i
        
        best_solution_per_itr.append(best_cost)
        evaporation(ph_map, roh)
        ant_list[best_ant_index].offline_update_ph()

        # check for stagnation
        stagnation_flag = True
        for i in range(ant_count):
            if ant_list[i].current_cost != best_cost:
                stagnation_flag = False
        if stagnation_flag == True:
            break

    return best_solution_per_itr

if __name__ == "__main__":
    alpha = 1
    beta = 1
    Q = 10000.0
    tao_0 = 1

    q0_list = [0.2, 0.5, 0.8]
    roh_list = [0.05, 0.3, 0.6]
    ant_count_list = [5, 10, 20]

    max_itr = 300
    best_solution_per_itr = []
    ant_list = []

    with open('cities.csv') as f:
        reader = csv.reader(f)
        coordinate_list = list(reader)
    
    for i in range(len(coordinate_list)):
        coordinate_list[i] = coordinate_list[i][1:]

    Distance = construct_distance_matrix(coordinate_list)
    heuristic = [[0 for x in range(len(coordinate_list))] for y in range(len(coordinate_list))]
    for i in range(len(coordinate_list)):
        for j in range(len(coordinate_list)):
            if i < j:
                heuristic[i][j] = 1.0/Distance[i][j]

    city_list = list(range(len(coordinate_list)))
    
    plt.figure(1)
    for roh in roh_list: 
        best_solution_per_itr = run_ACO(max_itr, 20, Distance, heuristic, city_list,
                alpha, beta, Q, roh, tao_0, 0.5)

        x = [i+1 for i in range(len(best_solution_per_itr))]

        plt.plot(x, best_solution_per_itr)
        plt.title("ACO without online update with different pheromone persistence constant (roh)")
        plt.xlabel("Iteration")
        plt.ylabel("best solution cost per iteration")
        plt.legend(['roh = 0.05', 'roh = 0.3', 'roh = 0.6'])
    plt.show()

    plt.figure(2)
    for q0 in q0_list: 
        best_solution_per_itr = run_ACO(max_itr, 10, Distance, heuristic, city_list,
                alpha, beta, Q, 0.05, tao_0, q0)

        x = [i+1 for i in range(len(best_solution_per_itr))]

        plt.plot(x, best_solution_per_itr)
        plt.title("ACO without online update with different state transition control parameter (q0)")
        plt.xlabel("Iteration")
        plt.ylabel("best solution cost per iteration")
        plt.legend(['q0 = 0.2', 'q0 = 0.5', 'q0 = 0.8'])
    plt.show()  
    
    plt.figure(3)
    for ant_count in ant_count_list: 
        best_solution_per_itr = run_ACO(max_itr, ant_count, Distance, heuristic, city_list,
                alpha, beta, Q, 0.05, tao_0, 0.5)

        x = [i+1 for i in range(len(best_solution_per_itr))]

        plt.plot(x, best_solution_per_itr)
        plt.title("ACO without online update with different state population size parameter (# of ants)")
        plt.xlabel("Iteration")
        plt.ylabel("best solution cost per iteration")
        plt.legend(['# of ants = 5', '# of ants = 10', '# of ants = 20'])
    plt.show()  


    

